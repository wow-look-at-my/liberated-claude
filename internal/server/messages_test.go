package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// proxyFixture wires a gateway in front of a stub upstream and records requests.
type proxyFixture struct {
	handler  http.Handler
	upstream *httptest.Server
	gotBody  []byte
	gotPath  string
	gotHdr   http.Header
}

func newProxyFixture(t *testing.T, kind, cache string, reply func(w http.ResponseWriter)) *proxyFixture {
	t.Helper()
	f := &proxyFixture{}
	f.upstream = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		f.gotBody, f.gotPath, f.gotHdr = body, r.URL.Path, r.Header.Clone()
		reply(w)
	}))
	t.Cleanup(f.upstream.Close)

	xml := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server><listen>127.0.0.1:0</listen><publicURL>http://x</publicURL><apiKey>%s</apiKey></server>
	<providers>
		<provider name="p" kind="%s" cache="%s">
			<baseURL>%s</baseURL>
			<apiKey>upstream-key</apiKey>
			<models>
				<model id="z-ai/glm-5.3-flash" label="GLM" tier="opus"
				       contextWindow="1310720" maxOutputTokens="1000"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`, testKey, kind, cache, f.upstream.URL)

	cfg, err := config.Parse([]byte(xml))
	require.NoError(t, err, "fixture config should parse")
	f.handler = New(cfg, f.upstream.Client(), slog.New(slog.DiscardHandler)).Handler()
	return f
}

func (f *proxyFixture) post(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", stringReader(body))
	req.Header.Set("x-api-key", testKey)
	rec := httptest.NewRecorder()
	f.handler.ServeHTTP(rec, req)
	return rec
}

// modelRef is the advertised ID for the fixture model, which is what Claude
// Desktop sends back on a request.
func modelRef(t *testing.T) string {
	t.Helper()
	cfg, err := config.Parse([]byte(testConfigXML))
	require.NoError(t, err, "config should parse")
	m, ok := cfg.Resolve("z-ai/glm-5.3-flash")
	require.True(t, ok, "fixture model should resolve")
	return m.AliasID()
}

func TestProxyOpenAITranslatesReplyAndCacheUsage(t *testing.T) {
	f := newProxyFixture(t, "openai", "implicit", func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"cc-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}}`)
	})

	body := fmt.Sprintf(`{"model":%q,"max_tokens":50,"messages":[{"role":"user","content":"hi"}]}`, modelRef(t))
	rec := f.post(t, body)
	require.Equal(t, http.StatusOK, rec.Code, "proxy should succeed")

	assert.Equal(t, "/chat/completions", f.gotPath, "OpenAI providers take chat completions")
	assert.Equal(t, "Bearer upstream-key", f.gotHdr.Get("Authorization"), "upstream key should be sent")

	var sent wire.OARequest
	require.NoError(t, json.Unmarshal(f.gotBody, &sent), "upstream body should decode")
	assert.Equal(t, "z-ai/glm-5.3-flash", sent.Model, "upstream must receive the real model ID")
	require.NotNil(t, sent.MaxTokens, "max_tokens should be set")
	assert.Equal(t, 50, *sent.MaxTokens, "max_tokens should pass through under the model cap")

	var got wire.MessagesResponse
	require.NoError(t, json.NewDecoder(rec.Body).Decode(&got), "reply should decode")
	assert.Equal(t, "end_turn", got.StopReason, "stop maps to end_turn")
	require.Len(t, got.Content, 1, "one text block expected")
	assert.Equal(t, "hi there", got.Content[0].Text, "text should carry through")

	// Anthropic input_tokens excludes cache hits (80 cached of 100 leaves 20 billed).
	assert.Equal(t, 20, got.Usage.InputTokens, "cached tokens must be subtracted from input")
	require.NotNil(t, got.Usage.CacheReadInputTokens, "cache reads should be reported")
	assert.Equal(t, 80, *got.Usage.CacheReadInputTokens, "cache read count should carry through")
}

// An implicit-cache provider must not be sent cache_control: it caches prefixes
// on its own, and some reject the unknown field outright.
func TestProxyOpenAIStripsCacheControlWhenImplicit(t *testing.T) {
	f := newProxyFixture(t, "openai", "implicit", func(w http.ResponseWriter) {
		io.WriteString(w, `{"id":"c","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	})
	body := fmt.Sprintf(`{"model":%q,"max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"big","cache_control":{"type":"ephemeral"}}]}]}`, modelRef(t))
	require.Equal(t, http.StatusOK, f.post(t, body).Code, "proxy should succeed")
	assert.NotContains(t, string(f.gotBody), "cache_control",
		"implicit caching must not forward cache_control")
}

// An Anthropic upstream gets the original bytes, so cache_control survives
// exactly as Claude Desktop sent it.
func TestProxyAnthropicPreservesCacheControl(t *testing.T) {
	f := newProxyFixture(t, "anthropic", "explicit", func(w http.ResponseWriter) {
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":1}}`)
	})
	body := fmt.Sprintf(`{"model":%q,"max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"big","cache_control":{"type":"ephemeral"}}]}]}`, modelRef(t))
	require.Equal(t, http.StatusOK, f.post(t, body).Code, "proxy should succeed")

	assert.Equal(t, "/v1/messages", f.gotPath, "Anthropic providers take the messages path")
	assert.Equal(t, "upstream-key", f.gotHdr.Get("x-api-key"), "key goes in x-api-key")
	assert.Equal(t, "2023-06-01", f.gotHdr.Get("anthropic-version"), "version header is required")
	assert.Contains(t, string(f.gotBody), `"cache_control"`, "cache_control must survive untouched")
	assert.Contains(t, string(f.gotBody), `"z-ai/glm-5.3-flash"`, "model should be rewritten to upstream ID")
}

func TestProxyStreamingEmitsAnthropicEvents(t *testing.T) {
	f := newProxyFixture(t, "openai", "implicit", func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\ndata: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n")
	})
	body := fmt.Sprintf(`{"model":%q,"max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, modelRef(t))
	rec := f.post(t, body)
	require.Equal(t, http.StatusOK, rec.Code, "stream should start")

	out := rec.Body.String()
	for _, want := range []string{
		"event: message_start",
		"event: content_block_start",
		"event: content_block_delta",
		"event: content_block_stop",
		"event: message_delta",
		"event: message_stop",
	} {
		assert.Contains(t, out, want, "stream should contain %s", want)
	}
	assert.Contains(t, out, "Hel", "first fragment should reach the client")
	assert.Contains(t, out, "lo", "second fragment should reach the client")

	var sent wire.OARequest
	require.NoError(t, json.Unmarshal(f.gotBody, &sent), "upstream body should decode")
	require.NotNil(t, sent.StreamOptions, "usage must be requested while streaming")
	assert.True(t, sent.StreamOptions.IncludeUsage,
		"without include_usage the final cache accounting is lost")
}

// An upstream failure reaches Claude Desktop with its own status and message,
// rather than being flattened into a generic gateway error.
func TestProxyRelaysUpstreamError(t *testing.T) {
	f := newProxyFixture(t, "openai", "implicit", func(w http.ResponseWriter) {
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, `{"error":{"message":"rate limited upstream","type":"rate_limit"}}`)
	})
	body := fmt.Sprintf(`{"model":%q,"max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`, modelRef(t))
	rec := f.post(t, body)

	assert.Equal(t, http.StatusTooManyRequests, rec.Code, "upstream status should be preserved")
	assert.Contains(t, rec.Body.String(), "rate limited upstream", "upstream message should survive")
}

func TestMalformedRequestBodyIsRejected(t *testing.T) {
	f := newProxyFixture(t, "openai", "implicit", func(w http.ResponseWriter) {})
	rec := f.post(t, `{"model":`)
	assert.Equal(t, http.StatusBadRequest, rec.Code, "unparseable JSON should be a 400")
	assert.True(t, strings.Contains(rec.Body.String(), "parse request"),
		"the error should say what failed")
}

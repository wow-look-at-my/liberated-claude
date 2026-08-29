package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// postCount sends a count_tokens request to h and returns the reported count.
func postCount(t *testing.T, h http.Handler, body string) (int, int) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		return 0, w.Code
	}
	var got struct {
		InputTokens int `json:"input_tokens"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	return got.InputTokens, w.Code
}

const countBody = `{"model":"m","messages":[{"role":"user","content":"The quick brown fox."}]}`

func TestCountTokensPrefersUpstream(t *testing.T) {
	var path string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Contains(t, string(raw), `"model":"m"`, "the upstream model id is substituted")
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"input_tokens":1234}`)
	}))
	defer upstream.Close()

	h := New(throttleConfig(t, upstream.URL, 2), upstream.Client(), slog.New(slog.DiscardHandler)).Handler()
	count, code := postCount(t, h, countBody)
	require.Equal(t, http.StatusOK, code)
	require.Equal(t, 1234, count, "the provider's own count wins over any estimate")
	require.Equal(t, "/messages/count_tokens", path)
}

func TestCountTokensProbesUpstreamOnlyOnce(t *testing.T) {
	var calls int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt64(&calls, 1)
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"error":"not found"}`)
	}))
	defer upstream.Close()

	h := New(throttleConfig(t, upstream.URL, 2), upstream.Client(), slog.New(slog.DiscardHandler)).Handler()
	for i := 0; i < 3; i++ {
		count, code := postCount(t, h, countBody)
		require.Equal(t, http.StatusOK, code, "a provider without the endpoint still gets an answer")
		require.Positive(t, count)
	}
	require.EqualValues(t, 1, atomic.LoadInt64(&calls),
		"an upstream that refuses once is not asked again")
}

func TestCountTokensRejectsUnknownModel(t *testing.T) {
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()

	h := New(throttleConfig(t, upstream.URL, 2), upstream.Client(), slog.New(slog.DiscardHandler)).Handler()
	_, code := postCount(t, h, `{"model":"nope","messages":[]}`)
	require.Equal(t, http.StatusNotFound, code)
}

func TestFallbackUsesTheRealTokenizer(t *testing.T) {
	tok, err := bpe()
	require.NoError(t, err)

	// cl100k_base splits this into exactly two tokens, so a count that merely
	// scaled the byte length would not land here.
	n, err := tok.CountTokens("Hello World")
	require.NoError(t, err)
	require.Equal(t, 2, n)

	req := &wire.MessagesRequest{
		Messages: []wire.Message{{Role: "user", Content: json.RawMessage(`"Hello World"`)}},
	}
	total, err := countInputTokens(req)
	require.NoError(t, err)

	// The turn's content is JSON-quoted, so it costs its own tokens plus the
	// quotes, plus the per-message framing.
	quoted, err := tok.CountTokens(`"Hello World"`)
	require.NoError(t, err)
	require.Equal(t, quoted+perMessageOverhead, total)
}

func TestFallbackCountsToolsAndSystem(t *testing.T) {
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()
	h := New(throttleConfig(t, upstream.URL, 2), upstream.Client(), slog.New(slog.DiscardHandler)).Handler()

	bare, _ := postCount(t, h, countBody)
	withSystem, _ := postCount(t, h,
		`{"model":"m","system":"you are a helpful assistant with a long preamble",`+
			`"messages":[{"role":"user","content":"The quick brown fox."}]}`)
	withTools, _ := postCount(t, h,
		`{"model":"m","messages":[{"role":"user","content":"The quick brown fox."}],`+
			`"tools":[{"name":"read_file","description":"Read a file from disk",`+
			`"input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]}`)

	require.Greater(t, withSystem, bare, "the system prompt is billed as input")
	require.Greater(t, withTools, bare, "tool schemas are billed as input")
}

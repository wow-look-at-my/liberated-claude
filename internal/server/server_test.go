package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/pricing"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

const testKey = "test-key"

// testConfigXML holds one model above the 1M threshold and one below, so the
// discovery assertions can tell the two apart.
const testConfigXML = `<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:0</listen>
		<publicURL>http://127.0.0.1:8787</publicURL>
		<apiKey>` + testKey + `</apiKey>
	</server>
	<bootstrap>
		<deploymentDisplayName>Test</deploymentDisplayName>
		<preferOneMContext>true</preferOneMContext>
	</bootstrap>
	<providers>
		<provider name="p" kind="openai" cache="implicit">
			<baseURL>https://example.invalid/v1</baseURL>
			<apiKey>k</apiKey>
			<models>
				<model id="z-ai/glm-5.3-flash" label="GLM" tier="opus" tierDefault="true"
				       contextWindow="1310720" maxOutputTokens="131072"/>
				<model id="small-model" label="Small" tier="haiku"
				       contextWindow="128000" maxOutputTokens="8192"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`

func newTestServer(t *testing.T) http.Handler {
	t.Helper()
	cfg, err := config.Parse([]byte(testConfigXML))
	require.NoError(t, err, "test config should parse")
	s := New(cfg, http.DefaultClient, slog.New(slog.DiscardHandler))
	s.SetRates(map[string]pricing.Rates{
		"z-ai/glm-5.3-flash": {InputPerMtok: 0.075, OutputPerMtok: 0.25, CacheReadPerMtok: 0.015},
	})
	return s.Handler()
}

func do(t *testing.T, h http.Handler, method, path, key string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, nil)
	if key != "" {
		req.Header.Set("x-api-key", key)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// Claude Desktop probes for RFC 8414 metadata to decide whether the gateway is
// its own authorization server. A 401 there reads as "an authorization server
// exists and refused you" and starts an SSO flow this gateway cannot complete,
// so these paths must answer 404 without requiring a credential.
func TestWellKnownProbesAre404Unauthenticated(t *testing.T) {
	h := newTestServer(t)
	for _, path := range []string{
		"/.well-known/oauth-authorization-server",
		"/.well-known/openid-configuration",
		"/.well-known/oauth-authorization-server/v1",
	} {
		rec := do(t, h, http.MethodGet, path, "")
		assert.Equal(t, http.StatusNotFound, rec.Code, "%s must be 404, never 401", path)
	}
}

func TestHealthzNeedsNoKey(t *testing.T) {
	rec := do(t, newTestServer(t), http.MethodGet, "/healthz", "")
	assert.Equal(t, http.StatusOK, rec.Code, "healthz should be open")
}

func TestModelsRequiresKey(t *testing.T) {
	h := newTestServer(t)
	assert.Equal(t, http.StatusUnauthorized,
		do(t, h, http.MethodGet, "/v1/models", "").Code, "missing key should be 401")
	assert.Equal(t, http.StatusUnauthorized,
		do(t, h, http.MethodGet, "/v1/models", "wrong").Code, "wrong key should be 401")
	assert.Equal(t, http.StatusOK,
		do(t, h, http.MethodGet, "/v1/models", testKey).Code, "correct key should pass")
}

// The reason this program exists: a model's real window reaches Claude Desktop
// instead of being clamped to Anthropic's 200000.
func TestModelsAdvertiseRealContextWindow(t *testing.T) {
	rec := do(t, newTestServer(t), http.MethodGet, "/v1/models", testKey)
	require.Equal(t, http.StatusOK, rec.Code, "discovery should succeed")

	var resp wire.ModelsResponse
	require.NoError(t, json.NewDecoder(rec.Body).Decode(&resp), "discovery body should decode")
	require.Len(t, resp.Data, 2, "both models should be listed")

	big := resp.Data[0]
	assert.Equal(t, 1310720, big.MaxInputTokens, "real window must be advertised verbatim")
	assert.True(t, big.SupportsOneM, "a 1310720-token model must offer the 1M variant")
	assert.Equal(t, "opus", big.AnthropicFamilyTier, "tier keeps the model from being dropped")
	assert.True(t, big.IsFamilyDefault, "tierDefault should carry through")
	assert.NotEqual(t, "z-ai/glm-5.3-flash", big.ID, "a rejected upstream ID must be encoded")

	small := resp.Data[1]
	assert.Equal(t, 128000, small.MaxInputTokens, "sub-1M window should be reported as-is")
	assert.False(t, small.SupportsOneM, "a 128000-token model must not claim 1M")
}

func TestBootstrapCarriesGatewayAndPricing(t *testing.T) {
	rec := do(t, newTestServer(t), http.MethodGet, "/bootstrap", testKey)
	require.Equal(t, http.StatusOK, rec.Code, "bootstrap should succeed")

	var doc map[string]any
	require.NoError(t, json.NewDecoder(rec.Body).Decode(&doc), "bootstrap body should decode")

	assert.Equal(t, "gateway", doc["inferenceProvider"], "provider should be gateway")
	assert.Equal(t, "x-api-key", doc["inferenceGatewayAuthScheme"], "auth scheme should match the key check")
	assert.Equal(t, "http://127.0.0.1:8787", doc["inferenceGatewayBaseUrl"], "gateway URL should come from config")
	assert.Equal(t, true, doc["modelPrefer1mContext"], "preferOneMContext should map through")
	assert.Equal(t, true, doc["modelDiscoveryEnabled"], "discovery should stay on")

	models, ok := doc["inferenceModels"].([]any)
	require.True(t, ok, "inferenceModels should be a list")
	assert.Len(t, models, 2, "both models should be listed")

	rows, ok := doc["inferenceModelPricing"].([]any)
	require.True(t, ok, "inferenceModelPricing should be a list")
	require.Len(t, rows, 1, "only the model with a detected rate should be priced")
	row := rows[0].(map[string]any)
	assert.InDelta(t, 0.075, row["inputPerMtok"], 1e-12, "detected input rate should carry through")
	assert.InDelta(t, 0.25, row["outputPerMtok"], 1e-12, "detected output rate should carry through")

	_, hasAnalysis := doc["chatAdvancedFileAnalysisEnabled"]
	assert.False(t, hasAnalysis, "an unset toggle should be omitted so the app keeps its own default")
	_, hasToolSearch := doc["toolSearchEnabled"]
	assert.False(t, hasToolSearch, "an unset toggle should be omitted so the app keeps its own default")
}

func TestBootstrapCarriesChatSurfaceToggles(t *testing.T) {
	xml := strings.Replace(testConfigXML,
		"<preferOneMContext>true</preferOneMContext>",
		"<chatTabEnabled>true</chatTabEnabled>"+
			"<chatAdvancedFileAnalysisEnabled>true</chatAdvancedFileAnalysisEnabled>"+
			"<toolSearchEnabled>true</toolSearchEnabled>", 1)
	cfg, err := config.Parse([]byte(xml))
	require.NoError(t, err, "config with chat surface toggles should parse")

	s := New(cfg, http.DefaultClient, slog.New(slog.DiscardHandler))
	rec := do(t, s.Handler(), http.MethodGet, "/bootstrap", testKey)
	require.Equal(t, http.StatusOK, rec.Code, "bootstrap should succeed")

	var doc map[string]any
	require.NoError(t, json.NewDecoder(rec.Body).Decode(&doc), "bootstrap body should decode")

	assert.Equal(t, true, doc["chatTabEnabled"], "chatTabEnabled should map through")
	assert.Equal(t, true, doc["chatAdvancedFileAnalysisEnabled"], "advanced file analysis should map through under its flat key")
	assert.Equal(t, true, doc["toolSearchEnabled"], "tool search should map through")
}

func TestBootstrapCarriesImportAndExtensions(t *testing.T) {
	xml := strings.Replace(testConfigXML,
		"<preferOneMContext>true</preferOneMContext>",
		"<desktopExtensionEnabled>true</desktopExtensionEnabled>"+
			"<claudeAiImport>"+
			"<enabled>true</enabled>"+
			"<url>https://example.invalid/export</url>"+
			"<oauthIssuer>https://example.invalid</oauthIssuer>"+
			"<oauthClientId>cid</oauthClientId>"+
			"<bannerBehavior>detect</bannerBehavior>"+
			"</claudeAiImport>", 1)
	cfg, err := config.Parse([]byte(xml))
	require.NoError(t, err, "a fully specified import block should parse")

	s := New(cfg, http.DefaultClient, slog.New(slog.DiscardHandler))
	rec := do(t, s.Handler(), http.MethodGet, "/bootstrap", testKey)
	require.Equal(t, http.StatusOK, rec.Code, "bootstrap should succeed")

	var doc map[string]any
	require.NoError(t, json.NewDecoder(rec.Body).Decode(&doc), "bootstrap body should decode")

	assert.Equal(t, true, doc["isDesktopExtensionEnabled"], "desktop extensions should map through under the app's flat key")

	imp, ok := doc["claudeAiImport"].(map[string]any)
	require.True(t, ok, "claudeAiImport should be a nested object, not flattened")
	assert.Equal(t, true, imp["enabled"], "import should be enabled")
	assert.Equal(t, "https://example.invalid/export", imp["url"], "export URL should carry through")
	assert.Equal(t, "https://example.invalid", imp["oauthIssuer"], "issuer should carry through")
	assert.Equal(t, "cid", imp["oauthClientId"], "client ID should carry through")
	assert.Equal(t, "detect", imp["bannerBehavior"], "banner behavior should carry through")
}

func TestBootstrapOmitsUnsetImportBlock(t *testing.T) {
	rec := do(t, newTestServer(t), http.MethodGet, "/bootstrap", testKey)
	require.Equal(t, http.StatusOK, rec.Code, "bootstrap should succeed")

	var doc map[string]any
	require.NoError(t, json.NewDecoder(rec.Body).Decode(&doc), "bootstrap body should decode")

	_, has := doc["claudeAiImport"]
	assert.False(t, has, "an absent import block should not be emitted as an empty object")
}

// A model whose rate is out of Claude Desktop's accepted range would invalidate
// the pricing key, so it is dropped rather than published.
func TestBootstrapDropsOutOfRangeRates(t *testing.T) {
	cfg, err := config.Parse([]byte(testConfigXML))
	require.NoError(t, err, "test config should parse")
	s := New(cfg, http.DefaultClient, slog.New(slog.DiscardHandler))
	s.SetRates(map[string]pricing.Rates{"z-ai/glm-5.3-flash": {InputPerMtok: 99999}})

	rec := do(t, s.Handler(), http.MethodGet, "/bootstrap", testKey)
	require.Equal(t, http.StatusOK, rec.Code, "bootstrap should succeed")
	var doc map[string]any
	require.NoError(t, json.NewDecoder(rec.Body).Decode(&doc), "bootstrap body should decode")
	_, present := doc["inferenceModelPricing"]
	assert.False(t, present, "an out-of-range rate should leave pricing unset")
}

func TestUnknownModelIsRejected(t *testing.T) {
	h := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		io.NopCloser(stringReader(`{"model":"nope","messages":[],"max_tokens":16}`)))
	req.Header.Set("x-api-key", testKey)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusNotFound, rec.Code, "an unknown model should be a 404")
}

type sr struct {
	s string
	i int
}

func (r *sr) Read(p []byte) (int, error) {
	if r.i >= len(r.s) {
		return 0, io.EOF
	}
	n := copy(p, r.s[r.i:])
	r.i += n
	return n, nil
}

func stringReader(s string) io.Reader { return &sr{s: s} }

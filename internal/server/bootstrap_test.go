package server

import (
	"log/slog"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/pricing"
)

// bootstrapFrom parses xml, serves its overlay to a client at the given origin,
// and returns the decoded document.
func bootstrapFrom(t *testing.T, xml, host string, secure bool) map[string]any {
	t.Helper()
	cfg, err := config.Parse([]byte(xml))
	require.NoError(t, err, "test config should parse")
	s := New(cfg, http.DefaultClient, slog.New(slog.DiscardHandler))
	return decodeDoc(t, fetchFrom(t, s.Handler(), "/bootstrap", host, secure))
}

func TestBootstrapCarriesGatewayAndPricing(t *testing.T) {
	doc := decodeDoc(t, do(t, newTestServer(t), http.MethodGet, "/bootstrap", testKey))

	assert.Equal(t, "gateway", doc["inferenceProvider"], "provider should be gateway")
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

// A gateway reached over https off-box can advertise its own URL and key,
// because remote intake keeps both in that case.
func TestBootstrapAdvertisesAPublicGateway(t *testing.T) {
	doc := decodeDoc(t, fetchFrom(t, newTestServer(t), "/bootstrap", "gateway.example.com:8787", true))

	assert.Equal(t, "https://gateway.example.com:8787", doc["inferenceGatewayBaseUrl"], "a public URL should be advertised")
	assert.Equal(t, testKey, doc["inferenceGatewayApiKey"], "the key should travel with the URL")
	assert.Equal(t, "static", doc["inferenceCredentialKind"], "the credential kind should travel with the key")
	assert.Equal(t, "x-api-key", doc["inferenceGatewayAuthScheme"], "the scheme should say how to send the key")
}

// inferenceGatewayBaseUrl is origin-pinned: the app drops it, and the credential
// it carries, unless its origin is exactly the one the document was fetched
// from. Naming the request's own origin keeps the two equal for a gateway that
// answers to more than one name.
func TestBootstrapGatewayURLFollowsTheRequestOrigin(t *testing.T) {
	h := newTestServer(t)
	for _, host := range []string{"gateway.example.com:8787", "gateway.localtest.me:8787"} {
		doc := decodeDoc(t, fetchFrom(t, h, "/bootstrap", host, true))
		assert.Equal(t, "https://"+host, doc["inferenceGatewayBaseUrl"], "the advertised URL should be the origin that asked")
	}
}

// A fetched document may not carry a loopback or non-https gateway URL, nor the
// credential pinned to it: the app deletes both and then calls the config
// invalid, which is worse than never advertising them.
func TestBootstrapOmitsAGatewayRemoteIntakeWouldDrop(t *testing.T) {
	h := newTestServer(t)
	for name, req := range map[string]struct {
		host   string
		secure bool
	}{
		"loopback over https": {"localhost:8787", true},
		"loopback by address": {"127.0.0.1:8787", true},
		"public over http":    {"gateway.example.com:8787", false},
	} {
		doc := decodeDoc(t, fetchFrom(t, h, "/bootstrap", req.host, req.secure))
		for _, key := range []string{
			"inferenceGatewayBaseUrl",
			"inferenceGatewayApiKey",
			"inferenceCredentialKind",
			"inferenceGatewayAuthScheme",
		} {
			assert.NotContains(t, doc, key, "%s must not be advertised to a %s fetch", key, name)
		}
	}
}

func TestBootstrapCarriesChatSurfaceToggles(t *testing.T) {
	doc := bootstrapFrom(t, strings.Replace(testConfigXML,
		"<modelPrefer1mContext>true</modelPrefer1mContext>",
		"<chatTabEnabled>true</chatTabEnabled>"+
			"<chatAdvancedFileAnalysisEnabled>true</chatAdvancedFileAnalysisEnabled>"+
			"<toolSearchEnabled>true</toolSearchEnabled>", 1), "example.com", false)

	assert.Equal(t, true, doc["chatTabEnabled"], "chatTabEnabled should map through")
	assert.Equal(t, true, doc["chatAdvancedFileAnalysisEnabled"], "advanced file analysis should map through under its flat key")
	assert.Equal(t, true, doc["toolSearchEnabled"], "tool search should map through")
}

func TestBootstrapCarriesImportAndExtensions(t *testing.T) {
	doc := bootstrapFrom(t, strings.Replace(testConfigXML,
		"<modelPrefer1mContext>true</modelPrefer1mContext>",
		"<isDesktopExtensionEnabled>true</isDesktopExtensionEnabled>"+
			"<claudeAiImport>"+
			"<enabled>true</enabled>"+
			"<bannerBehavior>detect</bannerBehavior>"+
			"</claudeAiImport>", 1), "example.com", false)

	assert.Equal(t, true, doc["isDesktopExtensionEnabled"], "desktop extensions should reach the overlay")

	imp, ok := doc["claudeAiImport"].(map[string]any)
	require.True(t, ok, "claudeAiImport should be a nested object, not flattened")
	assert.Equal(t, true, imp["enabled"], "import should be enabled")
	assert.Equal(t, "detect", imp["bannerBehavior"], "banner behavior should carry through")
	assert.NotContains(t, imp, "url", "no endpoint override should be invented")
}

func TestBootstrapOmitsUnsetImportBlock(t *testing.T) {
	doc := decodeDoc(t, do(t, newTestServer(t), http.MethodGet, "/bootstrap", testKey))
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

	doc := decodeDoc(t, do(t, s.Handler(), http.MethodGet, "/bootstrap", testKey))
	_, present := doc["inferenceModelPricing"]
	assert.False(t, present, "an out-of-range rate should leave pricing unset")
}

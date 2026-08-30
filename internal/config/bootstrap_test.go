package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// bootstrapXML wraps a bootstrap body in an otherwise valid document.
func bootstrapXML(body string) []byte {
	return []byte(`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<bootstrap>` + body + `</bootstrap>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>test-key</apiKey>
			<models>
				<model id="claude-opus" tier="opus" contextWindow="200000"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`)
}

func TestBootstrapPassesKeysThroughUntranslated(t *testing.T) {
	c, err := Parse(bootstrapXML(
		`<isDesktopExtensionEnabled>true</isDesktopExtensionEnabled>` +
			`<chatTabEnabled>false</chatTabEnabled>` +
			`<deploymentDisplayName>Liberated Claude</deploymentDisplayName>`))
	require.NoError(t, err, "arbitrary overlay keys should parse")

	doc := c.Bootstrap.JSON()
	assert.Equal(t, true, doc["isDesktopExtensionEnabled"], "true should become a JSON boolean")
	assert.Equal(t, false, doc["chatTabEnabled"], "false should become a JSON boolean")
	assert.Equal(t, "Liberated Claude", doc["deploymentDisplayName"], "other values should stay strings")
	assert.Len(t, doc, 3, "no key should be invented")
}

func TestBootstrapKeepsNumericLookingValuesAsStrings(t *testing.T) {
	c, err := Parse(bootstrapXML(
		`<claudeAiImport>` +
			`<url>https://example.invalid/e</url>` +
			`<oauthIssuer>https://example.invalid</oauthIssuer>` +
			`<oauthClientId>12345</oauthClientId>` +
			`</claudeAiImport>`))
	require.NoError(t, err, "a complete endpoint trio should parse")

	imp, ok := c.Bootstrap.JSON()["claudeAiImport"].(map[string]any)
	require.True(t, ok, "a nested element should become a nested object")
	assert.Equal(t, "12345", imp["oauthClientId"], "a numeric-looking credential must stay a string")
}

func TestBootstrapEnabledImportNeedsNoEndpoint(t *testing.T) {
	c, err := Parse(bootstrapXML(
		`<claudeAiImport><enabled>true</enabled><bannerBehavior>detect</bannerBehavior></claudeAiImport>`))
	require.NoError(t, err, "enabling import without an endpoint override is the normal case")

	imp := c.Bootstrap.JSON()["claudeAiImport"].(map[string]any)
	assert.Equal(t, true, imp["enabled"], "enabled should carry through")
	assert.Equal(t, "detect", imp["bannerBehavior"], "banner behavior should carry through")
}

func TestBootstrapImportEndpointIsAllOrNothing(t *testing.T) {
	_, err := Parse(bootstrapXML(
		`<claudeAiImport><url>https://example.invalid/e</url></claudeAiImport>`))
	require.Error(t, err, "a partial endpoint override should fail")
	assert.Contains(t, err.Error(), "oauthIssuer", "error should name the missing issuer")
	assert.Contains(t, err.Error(), "oauthClientId", "error should name the missing client ID")
	assert.Contains(t, err.Error(), "url set without", "error should name the field that was supplied")
}

func TestBootstrapRejectsUnknownBannerBehavior(t *testing.T) {
	_, err := Parse(bootstrapXML(
		`<claudeAiImport><bannerBehavior>always</bannerBehavior></claudeAiImport>`))
	require.Error(t, err, "an unknown banner behavior should fail")
	assert.Contains(t, err.Error(), "off, detect, show", "error should list the accepted values")
}

func TestBootstrapRendersItemChildrenAsArrays(t *testing.T) {
	c, err := Parse(bootstrapXML(
		`<sshHostAllowlist><item>*</item></sshHostAllowlist>` +
			`<coworkEgressAllowedHosts><item>a.invalid</item><item>b.invalid</item></coworkEgressAllowedHosts>` +
			`<authentication><disableClaudeAiSignIn>false</disableClaudeAiSignIn></authentication>`))
	require.NoError(t, err, "list and object elements should parse")

	doc := c.Bootstrap.JSON()
	assert.Equal(t, []any{"*"}, doc["sshHostAllowlist"], "a single item child should still be an array")
	assert.Equal(t, []any{"a.invalid", "b.invalid"}, doc["coworkEgressAllowedHosts"], "items should keep their order")
	assert.Equal(t, map[string]any{"disableClaudeAiSignIn": false},
		doc["authentication"], "a lone non-item child must stay an object, not become an array")
}

func TestBootstrapAbsentIsEmpty(t *testing.T) {
	c, err := Parse(bootstrapXML(``))
	require.NoError(t, err, "an empty bootstrap should parse")
	assert.Empty(t, c.Bootstrap.JSON(), "an empty bootstrap should add no keys")

	_, ok := c.Bootstrap.Bool("modelPrefer1mContext")
	assert.False(t, ok, "an absent toggle should report as not supplied")
}

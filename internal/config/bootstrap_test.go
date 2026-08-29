package config

import (
	"strings"
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

func TestParseDesktopExtensionToggle(t *testing.T) {
	c, err := Parse(bootstrapXML(`<desktopExtensionEnabled>true</desktopExtensionEnabled>`))
	require.NoError(t, err, "desktop extension toggle should parse")
	require.NotNil(t, c.Bootstrap.DesktopExtensionEnabled, "toggle should be present")
	assert.True(t, *c.Bootstrap.DesktopExtensionEnabled, "toggle should be true")
}

func TestParseImportBannerOnlyIsAllowed(t *testing.T) {
	c, err := Parse(bootstrapXML(
		`<claudeAiImport><bannerBehavior>detect</bannerBehavior></claudeAiImport>`))
	require.NoError(t, err, "a banner setting without an endpoint should be allowed")
	assert.Equal(t, "detect", c.Bootstrap.ClaudeAiImport.BannerBehavior, "banner behavior should parse")
	assert.True(t, c.Bootstrap.ClaudeAiImport.Set(), "a banner-only block still counts as set")
}

func TestParseImportEnabledRequiresEndpoint(t *testing.T) {
	_, err := Parse(bootstrapXML(
		`<claudeAiImport><enabled>true</enabled><url>https://example.invalid/e</url></claudeAiImport>`))
	require.Error(t, err, "enabling import without every endpoint field should fail")
	assert.Contains(t, err.Error(), "oauthIssuer", "error should name the missing issuer")
	assert.Contains(t, err.Error(), "oauthClientId", "error should name the missing client ID")
	assert.False(t, strings.Contains(err.Error(), " url"), "a supplied field should not be reported missing")
}

func TestParseImportRejectsUnknownBannerBehavior(t *testing.T) {
	_, err := Parse(bootstrapXML(
		`<claudeAiImport><bannerBehavior>always</bannerBehavior></claudeAiImport>`))
	require.Error(t, err, "an unknown banner behavior should fail")
	assert.Contains(t, err.Error(), "off, detect, show", "error should list the accepted values")
}

func TestParseImportAbsentIsUnset(t *testing.T) {
	c, err := Parse(bootstrapXML(`<chatTabEnabled>true</chatTabEnabled>`))
	require.NoError(t, err, "a bootstrap without an import block should parse")
	assert.False(t, c.Bootstrap.ClaudeAiImport.Set(), "an absent block should report unset")
}

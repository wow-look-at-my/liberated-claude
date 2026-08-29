package config

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/wow-look-at-my/liberated-claude/internal/alias"
)

func TestParseMinimalValid(t *testing.T) {
	xml := `<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>test-key</apiKey>
			<models>
				<model id="claude-opus" tier="opus" contextWindow="200000"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`

	c, err := Parse([]byte(xml))
	require.NoError(t, err, "parsing minimal valid config should succeed")

	assert.Equal(t, "127.0.0.1:8787", c.Server.Listen, "server listen should be parsed")
	assert.Equal(t, 1, len(c.Providers), "should have one provider")
	assert.Equal(t, "test", c.Providers[0].Name, "provider name should be test")
	assert.Equal(t, KindAnthropic, c.Providers[0].Kind, "provider kind should be anthropic")
	assert.Equal(t, 1, len(c.Providers[0].Models), "provider should have one model")
	assert.Equal(t, "claude-opus", c.Providers[0].Models[0].ID, "model id should be claude-opus")
	assert.Equal(t, "opus", c.Providers[0].Models[0].Tier, "model tier should be opus")
	assert.Equal(t, 200000, c.Providers[0].Models[0].ContextWindow, "model context window should be 200000")
}

func TestSupportsOneM(t *testing.T) {
	tests := []struct {
		name     string
		context  int
		wantOneM bool
		comment  string
	}{
		{"below threshold", 999999, false, "context window below 1M threshold should not support 1M variant"},
		{"at threshold", 1000000, true, "context window at exactly 1M threshold should support 1M variant (desktop requirement)"},
		{"above threshold", 1310720, true, "context window above threshold should support 1M variant"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{ContextWindow: tt.context}
			assert.Equal(t, tt.wantOneM, m.SupportsOneM(), tt.comment)
		})
	}
}

func TestEffectiveCache(t *testing.T) {
	tests := []struct {
		name              string
		modelCache        CacheMode
		providerCache     CacheMode
		expectedCacheMode CacheMode
		comment           string
	}{
		{
			"model cache overrides provider",
			CacheImplicit,
			CacheExplicit,
			CacheImplicit,
			"model-level cache attribute should override provider-level cache",
		},
		{
			"provider cache when model empty",
			"",
			CacheExplicit,
			CacheExplicit,
			"provider-level cache should apply when model omits cache",
		},
		{
			"both empty returns CacheNone",
			"",
			"",
			CacheNone,
			"CacheNone should be returned when neither model nor provider specifies cache",
		},
		{
			"model cache with empty provider",
			CacheImplicit,
			"",
			CacheImplicit,
			"model cache should override empty provider cache",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &Provider{Cache: tt.providerCache}
			model := &Model{Cache: tt.modelCache, provider: provider}
			assert.Equal(t, tt.expectedCacheMode, model.EffectiveCache(), tt.comment)
		})
	}
}

func TestAliasIDAndResolveRoundTrip(t *testing.T) {
	xml := `<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="anthropic" kind="anthropic">
			<baseURL>https://api.anthropic.com</baseURL>
			<apiKey>key</apiKey>
			<models>
				<model id="claude-opus-5" tier="opus" contextWindow="200000"/>
				<model id="deepseek/deepseek-v4-flash-0731" tier="sonnet" contextWindow="200000"/>
				<model id="z-ai/glm-5.3-flash" tier="opus" contextWindow="200000"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`

	c, err := Parse([]byte(xml))
	require.NoError(t, err, "parsing config should succeed")

	// Test Anthropic-looking ID keeps its id as alias
	anthropicModel, ok := c.Resolve("claude-opus-5")
	assert.True(t, ok, "Resolve(claude-opus-5) should find the model")
	assert.Equal(t, "claude-opus-5", anthropicModel.AliasID(), "Anthropic model should keep its id as alias")

	// Test foreign model gets encoded alias different from upstream id
	deepseekModel, ok := c.Resolve("deepseek/deepseek-v4-flash-0731")
	assert.True(t, ok, "Resolve(deepseek id) should find the model")
	assert.NotEqual(t, "deepseek/deepseek-v4-flash-0731", deepseekModel.AliasID(),
		"foreign model should get encoded alias different from upstream id")

	// Test Resolve accepts both upstream id and alias
	byUpstream, ok1 := c.Resolve("deepseek/deepseek-v4-flash-0731")
	byAlias, ok2 := c.Resolve(deepseekModel.AliasID())
	assert.True(t, ok1, "Resolve(upstream id) should find the model")
	assert.True(t, ok2, "Resolve(alias) should find the model")
	assert.Equal(t, byUpstream, byAlias, "both resolve paths should return same model")

	// Test Resolve with [1m] suffix works
	withOneM, ok := c.Resolve(deepseekModel.AliasID() + "[1m]")
	assert.True(t, ok, "Resolve(alias[1m]) should find the model")
	assert.Equal(t, deepseekModel, withOneM, "Resolve with [1m] should return same model")

	withOneM2, ok := c.Resolve("deepseek/deepseek-v4-flash-0731[1m]")
	assert.True(t, ok, "Resolve(upstream[1m]) should find the model")
	assert.Equal(t, deepseekModel, withOneM2, "Resolve upstream with [1m] should return same model")

	// Test Resolve of unknown id returns false
	_, ok = c.Resolve("unknown-model-xyz")
	assert.False(t, ok, "Resolve(unknown id) should return false")
}

func TestDisplayName(t *testing.T) {
	tests := []struct {
		name         string
		label        string
		id           string
		expectedName string
		comment      string
	}{
		{
			"with label",
			"My Custom Label",
			"claude-opus",
			"My Custom Label",
			"DisplayName should return label when set",
		},
		{
			"without label",
			"",
			"claude-sonnet-5",
			"claude-sonnet-5",
			"DisplayName should fall back to id when label is empty",
		},
		{
			"empty label uses id",
			"",
			"some-model-id",
			"some-model-id",
			"empty label should result in id being used",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{Label: tt.label, ID: tt.id}
			assert.Equal(t, tt.expectedName, m.DisplayName(), tt.comment)
		})
	}
}

func TestExpandEnv(t *testing.T) {
	t.Run("variable is set", func(t *testing.T) {
		t.Setenv("TEST_VAR", "test-value")
		result, missing := expandRefs("prefix-${TEST_VAR}-suffix")
		assert.Empty(t, missing, "a set variable should not be reported missing")
		assert.Equal(t, "prefix-test-value-suffix", result, "expandRefs should substitute set variables")
	})

	t.Run("variable not set is reported", func(t *testing.T) {
		os.Unsetenv("NONEXISTENT_VAR_XYZ")
		_, missing := expandRefs("prefix-${NONEXISTENT_VAR_XYZ}-suffix")
		assert.Equal(t, []string{"NONEXISTENT_VAR_XYZ"}, missing, "the unset name should be reported")
	})

	t.Run("multiple variables", func(t *testing.T) {
		t.Setenv("VAR1", "value1")
		t.Setenv("VAR2", "value2")
		result, missing := expandRefs("${VAR1}+${VAR2}")
		assert.Empty(t, missing, "set variables should not be reported missing")
		assert.Equal(t, "value1+value2", result, "expandRefs should substitute all variables")
	})

	t.Run("parse with unset server variable", func(t *testing.T) {
		xml := `<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
		<apiKey>${UNSET_API_KEY}</apiKey>
	</server>
	<providers/>
</liberatedClaude>`
		os.Unsetenv("UNSET_API_KEY")
		_, err := Parse([]byte(xml))
		assert.Error(t, err, "an unset server variable should be fatal")
		assert.Contains(t, err.Error(), "UNSET_API_KEY", "error should mention the unset variable")
	})
}

// A provider whose key is absent must be dropped rather than advertised.
// Claude Desktop probes a single model to validate the gateway, so one broken
// provider fails setup for every provider.
func TestProviderWithUnsetKeyIsSkippedNotFatal(t *testing.T) {
	os.Unsetenv("NO_SUCH_ANTHROPIC_KEY")
	t.Setenv("PRESENT_KEY", "real-key")

	xml := `<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server><listen>127.0.0.1:8787</listen><publicURL>http://x</publicURL><apiKey></apiKey></server>
	<providers>
		<provider name="anthropic" kind="anthropic" cache="explicit">
			<baseURL>https://api.anthropic.com</baseURL>
			<apiKey>${NO_SUCH_ANTHROPIC_KEY}</apiKey>
			<models>
				<model id="claude-sonnet-5" label="Sonnet" tier="sonnet"
				       contextWindow="1000000" maxOutputTokens="8192"/>
			</models>
		</provider>
		<provider name="ollama" kind="openai" cache="implicit">
			<baseURL>https://ollama.com/v1</baseURL>
			<apiKey>${PRESENT_KEY}</apiKey>
			<models>
				<model id="glm-5.3-flash" label="GLM" tier="opus"
				       contextWindow="1048576" maxOutputTokens="8192"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`

	cfg, err := Parse([]byte(xml))
	require.NoError(t, err, "one unusable provider must not fail the whole load")

	require.Len(t, cfg.Providers, 1, "only the usable provider should survive")
	assert.Equal(t, "ollama", cfg.Providers[0].Name, "the provider with a key should be kept")
	assert.Equal(t, "real-key", cfg.Providers[0].APIKey, "the surviving key should be expanded")

	require.Len(t, cfg.Skipped, 1, "the dropped provider should be recorded")
	assert.Equal(t, "anthropic", cfg.Skipped[0].Name, "the skipped provider should be named")
	assert.Equal(t, []string{"NO_SUCH_ANTHROPIC_KEY"}, cfg.Skipped[0].Missing,
		"the unset variable should be reported so the cause is visible")

	_, found := cfg.Resolve("claude-sonnet-5")
	assert.False(t, found, "a skipped provider's model must not be resolvable")
	assert.Contains(t, cfg.SkippedSummary()[0], "NO_SUCH_ANTHROPIC_KEY",
		"the summary should name the variable to set")
}

// With no provider left there is nothing to serve, and the error has to say
// why rather than claiming the config listed no providers.
func TestAllProvidersSkippedIsFatalWithCause(t *testing.T) {
	os.Unsetenv("NO_SUCH_KEY_AT_ALL")
	xml := `<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server><listen>127.0.0.1:8787</listen><publicURL>http://x</publicURL><apiKey></apiKey></server>
	<providers>
		<provider name="only" kind="openai" cache="implicit">
			<baseURL>https://example.invalid/v1</baseURL>
			<apiKey>${NO_SUCH_KEY_AT_ALL}</apiKey>
			<models>
				<model id="m" label="M" tier="opus" contextWindow="128000" maxOutputTokens="1024"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`

	_, err := Parse([]byte(xml))
	require.Error(t, err, "a gateway with no usable provider cannot start")
	assert.Contains(t, err.Error(), "NO_SUCH_KEY_AT_ALL", "the error should name the missing variable")
	assert.Contains(t, err.Error(), "only", "the error should name the skipped provider")
}

func TestValidationErrors(t *testing.T) {
	tests := []struct {
		name          string
		xml           string
		expectedError string
		comment       string
	}{
		{
			"missing server listen",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server/>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="m" tier="opus" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"server.listen is required",
			"validation should reject missing server.listen",
		},
		{
			"zero providers",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers/>
</liberatedClaude>`,
			"at least one provider is required",
			"validation should reject zero providers",
		},
		{
			"provider missing name",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="m" tier="opus" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"name attribute is required",
			"validation should reject provider without name",
		},
		{
			"provider missing kind",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="m" tier="opus" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"kind attribute is required",
			"validation should reject provider without kind",
		},
		{
			"provider unknown kind",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="unknown">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="m" tier="opus" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"unknown kind",
			"validation should reject unknown provider kind",
		},
		{
			"provider missing baseURL",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<apiKey>key</apiKey>
			<models><model id="m" tier="opus" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"baseURL is required",
			"validation should reject provider without baseURL",
		},
		{
			"provider unknown cache mode",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic" cache="badmode">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="m" tier="opus" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"unknown cache mode",
			"validation should reject unknown provider cache mode",
		},
		{
			"provider zero models",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models/>
		</provider>
	</providers>
</liberatedClaude>`,
			"at least one model is required",
			"validation should reject provider with zero models",
		},
		{
			"model missing id",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model tier="opus" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"id attribute is required",
			"validation should reject model without id",
		},
		{
			"model invalid tier",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="test" tier="invalid" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"tier attribute must be one of",
			"validation should reject invalid tier",
		},
		{
			"model empty tier",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="test" tier="" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"tier attribute must be one of",
			"validation should reject empty tier",
		},
		{
			"model zero context window",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="test" tier="opus" contextWindow="0"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"contextWindow attribute is required and must be positive",
			"validation should reject zero context window",
		},
		{
			"model negative context window",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="test" tier="opus" contextWindow="-100"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"contextWindow attribute is required and must be positive",
			"validation should reject negative context window",
		},
		{
			"model unknown cache mode",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models><model id="test" tier="opus" cache="badmode" contextWindow="200000"/></models>
		</provider>
	</providers>
</liberatedClaude>`,
			"unknown cache mode",
			"validation should reject unknown model cache mode",
		},
		{
			"alias collision",
			`<?xml version="1.0" encoding="UTF-8"?>
<liberatedClaude>
	<server>
		<listen>127.0.0.1:8787</listen>
	</server>
	<providers>
		<provider name="test" kind="anthropic">
			<baseURL>https://api.test.com</baseURL>
			<apiKey>key</apiKey>
			<models>
				<model id="deepseek/deepseek-v4-flash-0731" tier="opus" contextWindow="200000"/>
				<model id="deepseek/deepseek-v4-flash-0731" tier="sonnet" contextWindow="200000"/>
			</models>
		</provider>
	</providers>
</liberatedClaude>`,
			"collides with",
			"validation should reject duplicate alias ids",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse([]byte(tt.xml))
			assert.Error(t, err, tt.comment)
			assert.Contains(t, err.Error(), tt.expectedError, fmt.Sprintf("error should mention %q", tt.expectedError))
		})
	}
}

func TestParseExampleXML(t *testing.T) {
	examplePath := filepath.Join("..", "..", "config.example.xml")
	rawXML, err := os.ReadFile(examplePath)
	require.NoError(t, err, "should read config.example.xml from disk")

	// Set environment variables referenced in the example file
	t.Setenv("LIBERATED_CLAUDE_KEY", "test-gateway-key")
	t.Setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
	t.Setenv("OPENROUTER_API_KEY", "test-openrouter-key")
	t.Setenv("OLLAMA_API_KEY", "test-ollama-key")

	c, err := Parse(rawXML)
	require.NoError(t, err, "config.example.xml should parse and validate cleanly")

	// Verify every model has a recognized tier
	for _, model := range c.Models() {
		assert.True(t, alias.IsTier(model.Tier), fmt.Sprintf("model %q should have recognized tier, got %q", model.ID, model.Tier))
	}

	// Find the two flash models and verify they report SupportsOneM() == true
	found := map[string]bool{
		"z-ai/glm-5.3-flash":              false,
		"deepseek/deepseek-v4-flash-0731": false,
	}

	for _, model := range c.Models() {
		if model.ID == "z-ai/glm-5.3-flash" {
			assert.True(t, model.SupportsOneM(), "z-ai/glm-5.3-flash should support 1M context")
			found["z-ai/glm-5.3-flash"] = true
		}
		if model.ID == "deepseek/deepseek-v4-flash-0731" {
			assert.True(t, model.SupportsOneM(), "deepseek/deepseek-v4-flash-0731 should support 1M context")
			found["deepseek/deepseek-v4-flash-0731"] = true
		}
	}

	assert.True(t, found["z-ai/glm-5.3-flash"], "z-ai/glm-5.3-flash should be found in parsed config")
	assert.True(t, found["deepseek/deepseek-v4-flash-0731"], "deepseek/deepseek-v4-flash-0731 should be found in parsed config")
}

// Package config loads the XML file describing providers and models.
package config

import (
	"encoding/xml"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/wow-look-at-my/liberated-claude/internal/alias"
)

// OneMThreshold is where Claude Desktop treats a model as 1M-context: max_input_tokens >= 1e6.
const OneMThreshold = 1_000_000

// CacheMode says how a provider expects prompt caching to be requested.
type CacheMode string

const (
	// CacheExplicit: provider honors cache_control blocks (forwarded untouched).
	CacheExplicit CacheMode = "explicit"
	// CacheImplicit: provider caches prefixes automatically.
	CacheImplicit CacheMode = "implicit"
	// CacheNone: caching disabled (cache_control blocks stripped).
	CacheNone CacheMode = "none"
)

// Kind is the wire protocol a provider speaks.
type Kind string

const (
	// KindAnthropic: Anthropic Messages API (cache_control intact).
	KindAnthropic Kind = "anthropic"
	// KindOpenAI: OpenAI Chat Completions API (needs translation).
	KindOpenAI Kind = "openai"
)

// Config is the whole XML document.
type Config struct {
	XMLName   xml.Name   `xml:"liberatedClaude"`
	Server    Server     `xml:"server"`
	Bootstrap Bootstrap  `xml:"bootstrap"`
	Providers []Provider `xml:"providers>provider"`

	// Skipped lists providers dropped at load for missing credentials.
	Skipped []SkippedProvider `xml:"-"`
}

// SkippedProvider is a provider dropped for unset ${VAR} references. Desktop
// probes one model to validate the gateway, so a keyless provider fails setup
// for all of them.
type SkippedProvider struct {
	Name    string
	Missing []string
}

// Server describes the local listener.
type Server struct {
	// Listen is a host:port for the HTTP listener.
	Listen string `xml:"listen"`
	// APIKey is the credential Claude Desktop must present (empty disables check).
	APIKey string `xml:"apiKey"`
	// PublicURL is written to bootstrap as inferenceGatewayBaseUrl.
	PublicURL string `xml:"publicURL"`
}

// Bootstrap carries the parts of the config overlay that are not derived from
// the provider list. Every field maps to a documented Claude Desktop setting.
type Bootstrap struct {
	DeploymentDisplayName string `xml:"deploymentDisplayName"`
	ChatTabEnabled        *bool  `xml:"chatTabEnabled"`
	AutoModeEnabled       *bool  `xml:"autoModeEnabled"`
	ToolSearchEnabled     *bool  `xml:"toolSearchEnabled"`
	// PreferOneMContext picks the 1M variant by default (maps to modelPrefer1mContext).
	PreferOneMContext *bool `xml:"preferOneMContext"`
	// DisableTelemetry sets both telemetry keys the app understands.
	DisableTelemetry *bool `xml:"disableTelemetry"`
}

// Provider is one upstream API and the models reachable through it.
type Provider struct {
	Name    string    `xml:"name,attr"`
	Kind    Kind      `xml:"kind,attr"`
	BaseURL string    `xml:"baseURL"`
	APIKey  string    `xml:"apiKey"`
	Cache   CacheMode `xml:"cache,attr"`
	// ReasoningField is the JSON key this provider uses for chain-of-thought.
	ReasoningField string   `xml:"reasoningField,attr"`
	Headers        []Header `xml:"headers>header"`
	Models         []Model  `xml:"models>model"`
}

// The two spellings of the chain-of-thought field: DeepSeek uses the first,
// OpenRouter and Ollama the second.
const (
	ReasoningContent = "reasoning_content"
	Reasoning        = "reasoning"
)

// Header is an extra HTTP header sent upstream, such as OpenRouter's
// HTTP-Referer attribution.
type Header struct {
	Name  string `xml:"name,attr"`
	Value string `xml:",chardata"`
}

// Model is one entry in Claude Desktop's model picker.
type Model struct {
	// ID is the model name exactly as the upstream provider expects it.
	ID string `xml:"id,attr"`
	// Label is the display name shown in the picker.
	Label string `xml:"label,attr"`
	// Tier: sonnet, opus, haiku, fable, or mythos (required by Desktop).
	Tier string `xml:"tier,attr"`
	// TierDefault marks this model as the one a bare tier alias resolves to.
	TierDefault bool `xml:"tierDefault,attr"`
	// ContextWindow is the input-token capacity (1M-context offered if >= OneMThreshold).
	ContextWindow int `xml:"contextWindow,attr"`
	// MaxOutputTokens caps the response (zero leaves the request's own value).
	MaxOutputTokens int `xml:"maxOutputTokens,attr"`
	// Cache overrides the provider's cache mode for this model.
	Cache CacheMode `xml:"cache,attr"`

	// provider is filled in during Load so a resolved model knows its origin.
	provider *Provider
}

// Provider returns the provider this model is served by.
func (m *Model) Provider() *Provider { return m.provider }

// EffectiveCache returns the cache mode in force for this model.
func (m *Model) EffectiveCache() CacheMode {
	if m.Cache != "" {
		return m.Cache
	}
	if m.provider != nil && m.provider.Cache != "" {
		return m.provider.Cache
	}
	return CacheNone
}

// ReasoningFieldName is the key a replayed assistant turn carries its
// chain-of-thought under. Defaults to reasoning_content, the spelling whose
// absence makes DeepSeek reject a multi-turn request outright.
func (m *Model) ReasoningFieldName() string {
	if m.provider != nil && m.provider.ReasoningField != "" {
		return m.provider.ReasoningField
	}
	return ReasoningContent
}

// SupportsOneM reports whether Desktop should offer a 1M-context variant.
func (m *Model) SupportsOneM() bool { return m.ContextWindow >= OneMThreshold }

// AliasID is the advertised model ID (encoded if upstream would be rejected).
func (m *Model) AliasID() string { return alias.Encode(m.ID) }

// DisplayName is the picker label, falling back to the upstream ID.
func (m *Model) DisplayName() string {
	if m.Label != "" {
		return m.Label
	}
	return m.ID
}

// envRef matches ${NAME} references inside config text.
var envRef = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// expandRefs substitutes ${NAME} from the environment, reporting the names
// that were not set. An unset reference expands to empty and is never sent
// upstream, because the caller drops whatever it belonged to.
func expandRefs(s string) (string, []string) {
	var missing []string
	out := envRef.ReplaceAllStringFunc(s, func(ref string) string {
		name := envRef.FindStringSubmatch(ref)[1]
		val, ok := os.LookupEnv(name)
		if !ok {
			missing = append(missing, name)
			return ""
		}
		return val
	})
	return out, missing
}

// expandFields expands every field in place and returns the unset names, in
// first-seen order and without repeats.
func expandFields(fields []*string) []string {
	var missing []string
	for _, f := range fields {
		v, names := expandRefs(*f)
		*f = v
		for _, n := range names {
			if !slices.Contains(missing, n) {
				missing = append(missing, n)
			}
		}
	}
	return missing
}

// Load reads and validates the config at path.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	return Parse(raw)
}

// Parse decodes and validates config XML.
func Parse(raw []byte) (*Config, error) {
	var c Config
	if err := xml.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if err := c.expand(); err != nil {
		return nil, err
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	for i := range c.Providers {
		p := &c.Providers[i]
		for j := range p.Models {
			p.Models[j].provider = p
		}
	}
	return &c, nil
}

// expand resolves ${VAR} references in the fields that carry secrets or URLs.
//
// A missing server variable is fatal, since there is no gateway without it. A
// missing provider variable only costs that provider, so it is dropped and
// recorded in Skipped for the caller to log.
func (c *Config) expand() error {
	server := []*string{&c.Server.APIKey, &c.Server.Listen, &c.Server.PublicURL}
	if missing := expandFields(server); len(missing) > 0 {
		return fmt.Errorf("unset environment variable(s): %s", strings.Join(missing, ", "))
	}

	var kept []Provider
	for i := range c.Providers {
		p := c.Providers[i]
		fields := []*string{&p.APIKey, &p.BaseURL}
		for j := range p.Headers {
			fields = append(fields, &p.Headers[j].Value)
		}
		if missing := expandFields(fields); len(missing) > 0 {
			c.Skipped = append(c.Skipped, SkippedProvider{Name: p.Name, Missing: missing})
			continue
		}
		kept = append(kept, p)
	}
	c.Providers = kept
	return nil
}

// validate rejects a config that would produce a broken or silently degraded
// gateway. Each check corresponds to a failure that is hard to diagnose from
// Claude Desktop, where the only symptom is a missing or non-working model.
func (c *Config) validate() error {
	if c.Server.Listen == "" {
		return fmt.Errorf("server.listen is required")
	}
	if len(c.Providers) == 0 {
		if len(c.Skipped) > 0 {
			return fmt.Errorf("every provider was skipped for missing credentials: %s",
				strings.Join(c.SkippedSummary(), "; "))
		}
		return fmt.Errorf("at least one provider is required")
	}
	seenAlias := map[string]string{}
	for i := range c.Providers {
		p := &c.Providers[i]
		if p.Name == "" {
			return fmt.Errorf("provider %d: name attribute is required", i)
		}
		switch p.Kind {
		case KindAnthropic, KindOpenAI:
		case "":
			return fmt.Errorf("provider %q: kind attribute is required (anthropic or openai)", p.Name)
		default:
			return fmt.Errorf("provider %q: unknown kind %q", p.Name, p.Kind)
		}
		if p.BaseURL == "" {
			return fmt.Errorf("provider %q: baseURL is required", p.Name)
		}
		switch p.Cache {
		case "", CacheExplicit, CacheImplicit, CacheNone:
		default:
			return fmt.Errorf("provider %q: unknown cache mode %q", p.Name, p.Cache)
		}
		switch p.ReasoningField {
		case "", ReasoningContent, Reasoning:
		default:
			return fmt.Errorf("provider %q: unknown reasoningField %q (want %s or %s)",
				p.Name, p.ReasoningField, ReasoningContent, Reasoning)
		}
		if len(p.Models) == 0 {
			return fmt.Errorf("provider %q: at least one model is required", p.Name)
		}
		for j := range p.Models {
			if err := validateModel(p, &p.Models[j], j, seenAlias); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateModel(p *Provider, m *Model, idx int, seenAlias map[string]string) error {
	if m.ID == "" {
		return fmt.Errorf("provider %q model %d: id attribute is required", p.Name, idx)
	}
	// Desktop keeps a discovered model only when lo(id) passes or a valid tier
	// is present. An encoded alias never satisfies lo() on its own merits, so
	// requiring a tier here is what keeps the model in the picker at all.
	if !alias.IsTier(m.Tier) {
		return fmt.Errorf(
			"provider %q model %q: tier attribute must be one of %s (Claude Desktop drops models without a recognized tier)",
			p.Name, m.ID, strings.Join(alias.Tiers, ", "))
	}
	if m.ContextWindow <= 0 {
		return fmt.Errorf(
			"provider %q model %q: contextWindow attribute is required and must be positive",
			p.Name, m.ID)
	}
	switch m.Cache {
	case "", CacheExplicit, CacheImplicit, CacheNone:
	default:
		return fmt.Errorf("provider %q model %q: unknown cache mode %q", p.Name, m.ID, m.Cache)
	}
	// Shared advertised IDs make routing ambiguous (second shadows first).
	a := m.AliasID()
	if prev, dup := seenAlias[a]; dup {
		return fmt.Errorf("model %q collides with %q: both advertise ID %q", m.ID, prev, a)
	}
	seenAlias[a] = m.ID
	return nil
}

// SkippedSummary describes each dropped provider as "name (VAR1, VAR2)".
func (c *Config) SkippedSummary() []string {
	out := make([]string, 0, len(c.Skipped))
	for _, s := range c.Skipped {
		out = append(out, fmt.Sprintf("%s (%s)", s.Name, strings.Join(s.Missing, ", ")))
	}
	return out
}

// Models returns every configured model, in document order.
func (c *Config) Models() []*Model {
	var out []*Model
	for i := range c.Providers {
		for j := range c.Providers[i].Models {
			out = append(out, &c.Providers[i].Models[j])
		}
	}
	return out
}

// Resolve maps a model ID received from Claude Desktop back to its
// configuration. It accepts the advertised alias, the raw upstream ID, and
// either form carrying a "[1m]" variant marker.
func (c *Config) Resolve(id string) (*Model, bool) {
	base, _ := alias.SplitOneM(strings.TrimSpace(id))
	upstream := alias.Decode(base)
	for _, m := range c.Models() {
		if m.ID == upstream || m.AliasID() == base {
			return m, true
		}
	}
	return nil, false
}

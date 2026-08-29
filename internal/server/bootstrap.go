package server

import (
	"net/http"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/pricing"
)

// priceRow is one entry of Claude Desktop's inferenceModelPricing array. All
// four rates are required per row and each must fall in [0, 10000]; a row
// outside that range is rejected by the app.
type priceRow struct {
	Name              string  `json:"name"`
	InputPerMtok      float64 `json:"inputPerMtok"`
	OutputPerMtok     float64 `json:"outputPerMtok"`
	CacheReadPerMtok  float64 `json:"cacheReadPerMtok"`
	CacheWritePerMtok float64 `json:"cacheWritePerMtok"`
}

// modelEntry is one entry of Claude Desktop's inferenceModels array.
type modelEntry struct {
	Name                string `json:"name"`
	LabelOverride       string `json:"labelOverride,omitempty"`
	AnthropicFamilyTier string `json:"anthropicFamilyTier,omitempty"`
	IsFamilyDefault     bool   `json:"isFamilyDefault,omitempty"`
	SupportsOneM        bool   `json:"supports1m,omitempty"`
	PreferOneM          bool   `json:"prefer1m,omitempty"`
}

// handleBootstrap serves the config overlay (overrides app settings, read-only).
func (s *Server) handleBootstrap(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.bootstrapConfig())
}

// bootstrapConfig builds the overlay document.
func (s *Server) bootstrapConfig() map[string]any {
	b := s.cfg.Bootstrap
	out := map[string]any{
		"inferenceProvider":       "gateway",
		"inferenceCredentialKind": "static",
		"inferenceGatewayBaseUrl": s.cfg.Server.PublicURL,
		"inferenceGatewayApiKey":  s.cfg.Server.APIKey,
		// Desktop sends key as x-api-key (checked first by authenticate()).
		"inferenceGatewayAuthScheme": "x-api-key",
		// Model list from /v1/models; inferenceModels is fallback if discovery fails.
		"modelDiscoveryEnabled": true,
		"inferenceModels":       s.modelEntries(),
	}
	if b.DeploymentDisplayName != "" {
		out["deploymentDisplayName"] = b.DeploymentDisplayName
	}
	setBool(out, "chatTabEnabled", b.ChatTabEnabled)
	setBool(out, "chatAdvancedFileAnalysisEnabled", b.ChatAdvancedFileAnalysisEnabled)
	setBool(out, "autoModeEnabled", b.AutoModeEnabled)
	setBool(out, "toolSearchEnabled", b.ToolSearchEnabled)
	setBool(out, "isDesktopExtensionEnabled", b.DesktopExtensionEnabled)
	setBool(out, "modelPrefer1mContext", b.PreferOneMContext)
	if imp := importBlock(b.ClaudeAiImport); imp != nil {
		out["claudeAiImport"] = imp
	}
	if b.DisableTelemetry != nil && *b.DisableTelemetry {
		out["disableEssentialTelemetry"] = true
		out["disableNonessentialTelemetry"] = true
	}
	if rows := s.priceRows(); len(rows) > 0 {
		out["inferenceModelPricingEnabled"] = true
		out["inferenceModelPricing"] = rows
	}
	return out
}

// importBlock renders claudeAiImport as the nested object the app expects,
// returning nil when the config left the whole block out.
func importBlock(c config.ClaudeAiImport) map[string]any {
	if !c.Set() {
		return nil
	}
	out := map[string]any{}
	setBool(out, "enabled", c.Enabled)
	setString(out, "url", c.URL)
	setString(out, "oauthIssuer", c.OAuthIssuer)
	setString(out, "oauthClientId", c.OAuthClientID)
	setString(out, "bannerBehavior", c.BannerBehavior)
	return out
}

// setString adds key only when the config supplied a non-empty value.
func setString(m map[string]any, key, v string) {
	if v != "" {
		m[key] = v
	}
}

// setBool adds key only when the config supplied a value, so an unset element
// leaves the app's own default in place rather than forcing false.
func setBool(m map[string]any, key string, v *bool) {
	if v != nil {
		m[key] = *v
	}
}

// modelEntries lists the configured models in Claude Desktop's own shape.
func (s *Server) modelEntries() []modelEntry {
	models := s.cfg.Models()
	out := make([]modelEntry, 0, len(models))
	preferOneM := s.cfg.Bootstrap.PreferOneMContext != nil && *s.cfg.Bootstrap.PreferOneMContext
	for _, m := range models {
		oneM := m.SupportsOneM()
		out = append(out, modelEntry{
			Name:                m.AliasID(),
			LabelOverride:       m.DisplayName(),
			AnthropicFamilyTier: m.Tier,
			IsFamilyDefault:     m.TierDefault,
			SupportsOneM:        oneM,
			PreferOneM:          oneM && preferOneM,
		})
	}
	return out
}

// priceRows converts detected rates into pricing rows, skipping models with no
// detected rate and rows the app would reject. A model absent from this list is
// estimated at Anthropic list price instead, which is wrong but visible; an
// out-of-range row would invalidate the whole key.
func (s *Server) priceRows() []priceRow {
	rates := s.Rates()
	if len(rates) == 0 {
		return nil
	}
	var out []priceRow
	for _, m := range s.cfg.Models() {
		r, ok := lookupRate(rates, m)
		if !ok || !r.Valid() {
			continue
		}
		out = append(out, priceRow{
			Name:              m.AliasID(),
			InputPerMtok:      r.InputPerMtok,
			OutputPerMtok:     r.OutputPerMtok,
			CacheReadPerMtok:  r.CacheReadPerMtok,
			CacheWritePerMtok: r.CacheWritePerMtok,
		})
	}
	return out
}

// lookupRate finds a model's rate by upstream ID first, then by advertised ID,
// since a detector keys its table by whatever the upstream API calls the model.
func lookupRate(rates map[string]pricing.Rates, m *config.Model) (pricing.Rates, bool) {
	if r, ok := rates[m.ID]; ok {
		return r, true
	}
	r, ok := rates[m.AliasID()]
	return r, ok
}

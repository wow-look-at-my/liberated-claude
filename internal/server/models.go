package server

import (
	"net/http"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// handleModels serves discovery (/v1/models). anthropic_family_tier keeps
// models in list; max_input_tokens must be accurate for 1M-context decision.
func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	models := s.cfg.Models()
	data := make([]wire.ModelInfo, 0, len(models))
	for _, m := range models {
		data = append(data, modelInfo(m))
	}
	resp := wire.ModelsResponse{Data: data, HasMore: false}
	if len(data) > 0 {
		first, last := data[0].ID, data[len(data)-1].ID
		resp.FirstID, resp.LastID = &first, &last
	}
	writeJSON(w, http.StatusOK, resp)
}

// modelInfo renders one configured model for discovery.
func modelInfo(m *config.Model) wire.ModelInfo {
	return wire.ModelInfo{
		Type:                "model",
		ID:                  m.AliasID(),
		DisplayName:         m.DisplayName(),
		AnthropicFamilyTier: m.Tier,
		IsFamilyDefault:     m.TierDefault,
		// Both sent: supports_1m checked first; max_input_tokens for window size.
		SupportsOneM:    m.SupportsOneM(),
		MaxInputTokens:  m.ContextWindow,
		MaxOutputTokens: m.MaxOutputTokens,
	}
}

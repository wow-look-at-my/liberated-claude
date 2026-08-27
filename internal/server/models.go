package server

import (
	"net/http"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// handleModels serves discovery. Claude Desktop fetches this at launch as
// GET {inferenceGatewayBaseUrl}/v1/models?limit=1000 and populates its picker
// from the result.
//
// Two fields decide whether a model survives and how much context it is offered
// with, and both are easy to get wrong:
//
//   - anthropic_family_tier: an entry whose ID does not look Anthropic-shaped is
//     discarded unless it carries a recognized tier. Every advertised ID here is
//     an encoded alias, so the tier is what keeps the model in the list.
//   - max_input_tokens: at or above 1000000 the app offers the model's
//     1M-context variant. Reporting the model's real capacity here is the point
//     of this program; clamping it to 200000 is what makes other routers useless
//     for long-context models.
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
		// Both fields are sent. supports_1m is consulted first and settles the
		// question on its own; max_input_tokens carries the real number through
		// for any consumer that reads the window rather than the flag.
		SupportsOneM:    m.SupportsOneM(),
		MaxInputTokens:  m.ContextWindow,
		MaxOutputTokens: m.MaxOutputTokens,
	}
}

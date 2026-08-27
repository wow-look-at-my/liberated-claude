package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// handleCountTokens answers the token-count probe Claude Desktop sends before
// a turn. Without this route the request 404s and the client falls back to
// asking the model to count, which spends a whole inference call per estimate.
func (s *Server) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", err.Error())
		return
	}
	var req wire.MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("parse request: %v", err))
		return
	}
	m, ok := s.cfg.Resolve(req.Model)
	if !ok {
		writeError(w, http.StatusNotFound, "not_found_error",
			fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	p := m.Provider()

	count, ok := s.upstreamCountTokens(r, body, m)
	if ok {
		s.log.Info("count_tokens", "model", req.Model, "provider", p.Name,
			"source", "upstream", "input_tokens", count)
		writeJSON(w, http.StatusOK, map[string]int{"input_tokens": count})
		return
	}

	count = estimateInputTokens(&req)
	s.log.Info("count_tokens", "model", req.Model, "provider", p.Name,
		"source", "estimate", "input_tokens", count)
	writeJSON(w, http.StatusOK, map[string]int{"input_tokens": count})
}

// upstreamCountTokens asks the provider to count, reporting whether it could.
//
// The first call decides for the provider: one that answers 404 or 405 has no
// such endpoint, and is not asked again, so an unsupported upstream costs a
// single probe rather than a wasted round trip on every turn.
func (s *Server) upstreamCountTokens(r *http.Request, body []byte, m *config.Model) (int, bool) {
	p := m.Provider()
	if !s.countTokensSupported(p.Name) {
		return 0, false
	}
	out, err := rewriteModel(body, m.ID)
	if err != nil {
		return 0, false
	}
	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		countTokensURL(p), bytes.NewReader(out))
	if err != nil {
		return 0, false
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("anthropic-version", "2023-06-01")
	if p.Kind == config.KindAnthropic {
		upstream.Header.Set("x-api-key", p.APIKey)
	} else {
		// Ollama's Anthropic-shaped routes authenticate as OpenAI's do.
		upstream.Header.Set("Authorization", "Bearer "+p.APIKey)
	}
	applyHeaders(upstream, p)

	resp, err := s.sendUpstream(upstream)
	if err != nil {
		s.log.Warn("count_tokens upstream call failed", "provider", p.Name, "error", err)
		return 0, false
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
	if resp.StatusCode != http.StatusOK {
		s.markCountTokensUnsupported(p.Name, resp.StatusCode, raw)
		return 0, false
	}
	var counted struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(raw, &counted); err != nil || counted.InputTokens <= 0 {
		s.markCountTokensUnsupported(p.Name, resp.StatusCode, raw)
		return 0, false
	}
	return counted.InputTokens, true
}

// countTokensURL is the count route beside the provider's messages route.
func countTokensURL(p *config.Provider) string {
	base := strings.TrimRight(p.BaseURL, "/")
	if p.Kind == config.KindAnthropic {
		return base + "/v1/messages/count_tokens"
	}
	// An OpenAI-shaped baseURL already carries its version segment.
	return base + "/messages/count_tokens"
}

// markCountTokensUnsupported records that a provider cannot count, and says
// what it answered so the decision is visible rather than inferred.
func (s *Server) markCountTokensUnsupported(name string, status int, body []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.noCountTokens == nil {
		s.noCountTokens = map[string]bool{}
	}
	if s.noCountTokens[name] {
		return
	}
	s.noCountTokens[name] = true
	s.log.Warn("provider has no count_tokens endpoint, estimating from here on",
		"provider", name, "status", status, "body", strings.TrimSpace(string(body)))
}

// countTokensSupported reports whether the provider is still worth asking.
func (s *Server) countTokensSupported(name string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return !s.noCountTokens[name]
}

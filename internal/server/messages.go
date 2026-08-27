package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/transform"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// maxRequestBytes bounds a Messages request (generous headroom against attacks).
const maxRequestBytes = 256 << 20

// handleMessages proxies a Messages API call to the provider serving the
// requested model, translating in both directions when that provider speaks
// OpenAI instead.
func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
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
	model, ok := s.cfg.Resolve(req.Model)
	if !ok {
		writeError(w, http.StatusNotFound, "not_found_error",
			fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	s.log.Info("messages", "model", req.Model, "upstream", model.ID,
		"provider", model.Provider().Name, "stream", req.Stream)

	// Held for the whole exchange, since a streamed body is still upstream work.
	release, err := s.acquire(r.Context(), model.Provider())
	if err != nil {
		writeError(w, http.StatusRequestTimeout, "request_canceled", err.Error())
		return
	}
	defer release()

	if model.Provider().Kind == config.KindAnthropic {
		s.proxyAnthropic(w, r, &req, body, model)
		return
	}
	s.proxyOpenAI(w, r, &req, model)
}

// proxyAnthropic forwards the request untouched to an Anthropic-compatible
// upstream. The original bytes are replayed rather than a re-encoding of the
// parsed struct, so cache_control blocks and any beta fields this gateway does
// not model survive exactly as Claude Desktop sent them.
func (s *Server) proxyAnthropic(
	w http.ResponseWriter, r *http.Request,
	req *wire.MessagesRequest, body []byte, m *config.Model,
) {
	out, err := rewriteModel(body, m.ID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	p := m.Provider()
	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		strings.TrimRight(p.BaseURL, "/")+"/v1/messages", bytes.NewReader(out))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("anthropic-version", "2023-06-01")
	upstream.Header.Set("x-api-key", p.APIKey)
	// Beta opt-ins ride on this header; forwarding it is what lets tool search
	// and context management reach a provider that supports them.
	if b := r.Header.Get("anthropic-beta"); b != "" {
		upstream.Header.Set("anthropic-beta", b)
	}
	applyHeaders(upstream, p)

	resp, err := s.sendUpstream(upstream)
	if err != nil {
		writeError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp, req.Stream)
}

// retryBackoff is how long to wait before each retry of a throttled call.
var retryBackoff = []time.Duration{
	250 * time.Millisecond,
	1 * time.Second,
	3 * time.Second,
}

// sendUpstream issues req, retrying while the provider reports it is throttled.
// The gate above bounds this gateway's own concurrency, but a provider account
// is shared, so a 429 can still arrive from work started elsewhere.
func (s *Server) sendUpstream(req *http.Request) (*http.Response, error) {
	for attempt := 0; ; attempt++ {
		resp, err := s.client.Do(req)
		if err != nil || !throttled(resp.StatusCode) || attempt >= len(retryBackoff) {
			return resp, err
		}
		wait := retryAfter(resp, retryBackoff[attempt])
		s.log.Warn("upstream throttled, retrying",
			"status", resp.StatusCode, "attempt", attempt+1, "wait", wait)
		resp.Body.Close()

		// A replayable body is required to send the same request twice.
		if req.GetBody == nil {
			return resp, fmt.Errorf("upstream returned %d and the request cannot be replayed", resp.StatusCode)
		}
		body, err := req.GetBody()
		if err != nil {
			return nil, fmt.Errorf("replay request body: %w", err)
		}
		req.Body = body

		select {
		case <-time.After(wait):
		case <-req.Context().Done():
			return nil, req.Context().Err()
		}
	}
}

// throttled reports whether a status means "try again shortly".
func throttled(status int) bool {
	return status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable
}

// retryAfter prefers the provider's own Retry-After over the local backoff.
func retryAfter(resp *http.Response, fallback time.Duration) time.Duration {
	secs, err := strconv.Atoi(strings.TrimSpace(resp.Header.Get("Retry-After")))
	if err != nil || secs < 0 {
		return fallback
	}
	return time.Duration(secs) * time.Second
}

// proxyOpenAI translates the request to Chat Completions, sends it, and
// translates the reply back.
func (s *Server) proxyOpenAI(
	w http.ResponseWriter, r *http.Request, req *wire.MessagesRequest, m *config.Model,
) {
	oaReq, err := transform.AnthropicToOpenAI(req, m)
	if err != nil {
		// Our own 400, so relayUpstreamError never sees it.
		s.log.Error("request translation failed", "model", m.ID, "error", err)
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	payload, err := json.Marshal(oaReq)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	p := m.Provider()
	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		strings.TrimRight(p.BaseURL, "/")+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("Authorization", "Bearer "+p.APIKey)
	applyHeaders(upstream, p)

	resp, err := s.sendUpstream(upstream)
	if err != nil {
		writeError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= http.StatusBadRequest {
		s.relayUpstreamError(w, resp)
		return
	}
	advertised := m.AliasID()
	if req.Stream {
		s.streamOpenAI(w, resp, advertised)
		return
	}
	s.completeOpenAI(w, resp, advertised)
}

// streamOpenAI converts an OpenAI SSE stream into an Anthropic one.
func (s *Server) streamOpenAI(w http.ResponseWriter, resp *http.Response, advertised string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	if err := transform.StreamOpenAIToAnthropic(w, resp.Body, advertised); err != nil {
		// Status and stream already sent; only recourse is logging and ending stream.
		s.log.Error("stream translation failed", "error", err, "model", advertised)
	}
}

// completeOpenAI converts a single OpenAI reply into an Anthropic one.
func (s *Server) completeOpenAI(w http.ResponseWriter, resp *http.Response, advertised string) {
	var oaResp wire.OAResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaResp); err != nil {
		writeError(w, http.StatusBadGateway, "api_error",
			fmt.Sprintf("decode upstream response: %v", err))
		return
	}
	out, err := transform.OpenAIToAnthropic(&oaResp, advertised)
	if err != nil {
		writeError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// relayUpstreamError forwards an upstream failure with its status and message
// intact, so a provider's own explanation reaches Claude Desktop instead of a
// generic gateway error.
func (s *Server) relayUpstreamError(w http.ResponseWriter, resp *http.Response) {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	msg := strings.TrimSpace(string(raw))
	var oaErr struct {
		Error *wire.OAError `json:"error"`
	}
	if json.Unmarshal(raw, &oaErr) == nil && oaErr.Error != nil && oaErr.Error.Message != "" {
		msg = oaErr.Error.Message
	}
	s.log.Error("upstream error", "status", resp.StatusCode, "body", msg)
	writeError(w, resp.StatusCode, "api_error", msg)
}

// applyHeaders adds the provider's configured extra headers.
func applyHeaders(r *http.Request, p *config.Provider) {
	for _, h := range p.Headers {
		r.Header.Set(h.Name, h.Value)
	}
}

// rewriteModel replaces the model field in a raw request body, leaving every
// other byte alone.
func rewriteModel(body []byte, model string) ([]byte, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, fmt.Errorf("rewrite model: %w", err)
	}
	enc, err := json.Marshal(model)
	if err != nil {
		return nil, fmt.Errorf("rewrite model: %w", err)
	}
	m["model"] = enc
	return json.Marshal(m)
}

// copyResponse relays an upstream reply verbatim, flushing as it goes when the
// body is a stream.
func copyResponse(w http.ResponseWriter, resp *http.Response, stream bool) {
	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	if !stream {
		io.Copy(w, resp.Body)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		io.Copy(w, resp.Body)
		return
	}
	buf := make([]byte, 8<<10)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			w.Write(buf[:n])
			flusher.Flush()
		}
		if err != nil {
			return
		}
	}
}

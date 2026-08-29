// Package server implements the three HTTP endpoints Claude Desktop needs:
// the bootstrap config overlay, model discovery, and the Messages API itself.
package server

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/pricing"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// Server holds the loaded config and the rates detected from upstream.
type Server struct {
	cfg    *config.Config
	client *http.Client
	log    *slog.Logger

	// gates admits a bounded number of upstream calls per provider name.
	gates map[string]chan struct{}

	// codes holds authorization codes between /oauth/authorize and /oauth/token.
	codes *codeStore

	mu    sync.RWMutex
	rates map[string]pricing.Rates
	// noCountTokens names providers that answered the count probe with a refusal.
	noCountTokens map[string]bool
}

// New builds a Server. rates is keyed by advertised model ID and may be nil,
// in which case Claude Desktop falls back to its own list-price estimate.
func New(cfg *config.Config, client *http.Client, log *slog.Logger) *Server {
	gates := map[string]chan struct{}{}
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		if p.MaxConcurrent > 0 {
			gates[p.Name] = make(chan struct{}, p.MaxConcurrent)
		}
	}
	return &Server{
		cfg:           cfg,
		client:        client,
		log:           log,
		gates:         gates,
		codes:         newCodeStore(),
		rates:         map[string]pricing.Rates{},
		noCountTokens: map[string]bool{},
	}
}

// acquire waits for a slot on the provider's gate and returns the release. It
// bounds how many calls this gateway has in flight at once, because Claude
// Desktop opens dozens at a time while warming sessions and a provider that
// limits by concurrency answers the excess with 429 rather than queueing.
//
// A provider with no configured limit gets a no-op release.
func (s *Server) acquire(ctx context.Context, p *config.Provider) (func(), error) {
	gate, limited := s.gates[p.Name]
	if !limited {
		return func() {}, nil
	}
	select {
	case gate <- struct{}{}:
		return func() { <-gate }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// SetRates replaces the detected pricing table.
func (s *Server) SetRates(r map[string]pricing.Rates) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rates = r
}

// Rates returns the current pricing table.
func (s *Server) Rates() map[string]pricing.Rates {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]pricing.Rates, len(s.rates))
	for k, v := range s.rates {
		out[k] = v
	}
	return out
}

// Handler returns the routed, authenticated handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /bootstrap", s.handleBootstrap)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	mux.HandleFunc("POST /v1/messages/count_tokens", s.handleCountTokens)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /.well-known/oauth-authorization-server", s.handleAuthServerMetadata)
	mux.HandleFunc("GET /oauth/authorize", s.handleAuthorize)
	mux.HandleFunc("POST /oauth/device_authorization", s.handleDeviceAuthorization)
	mux.HandleFunc("GET /oauth/device", s.handleDeviceVerification)
	mux.HandleFunc("POST /oauth/token", s.handleToken)
	mux.HandleFunc("/.well-known/", handleWellKnown)
	return s.logRequests(s.authenticate(mux))
}

// logRequests records every request with the status it got. Without this a
// path no route matches is answered by the mux with a bare 404 that no handler
// ever sees, so a client failure leaves no trace anywhere in this process.
func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		if rec.status < http.StatusBadRequest {
			return
		}
		s.log.Warn("request failed", "method", r.Method, "path", r.URL.Path, "status", rec.status)
	})
}

// statusRecorder remembers the status written through it.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

// Flush keeps the wrapper transparent to the streaming path.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// authenticate enforces the configured key.
//
// Claude Desktop sends the credential as either an x-api-key header or an
// Authorization bearer token depending on inferenceGatewayAuthScheme, so both
// are accepted. An empty configured key disables the check entirely, which is
// only appropriate on a loopback listener.
func (s *Server) authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		want := s.cfg.Server.APIKey
		if want == "" || unauthenticatedPath(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		if presentedKey(r) != want {
			s.log.Warn("rejected request", "path", r.URL.Path, "remote", r.RemoteAddr)
			writeJSON(w, http.StatusUnauthorized,
				wire.NewAPIError("authentication_error", "invalid API key"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

// unauthenticatedPath exempts what a client reaches before it holds a credential.
func unauthenticatedPath(p string) bool {
	return p == "/healthz" ||
		strings.HasPrefix(p, "/.well-known/") ||
		strings.HasPrefix(p, "/oauth/")
}

// handleWellKnown declines the discovery documents this gateway does not host.
func handleWellKnown(w http.ResponseWriter, _ *http.Request) {
	writeError(w, http.StatusNotFound, "not_found_error",
		"this gateway hosts no discovery document at that path")
}

// presentedKey pulls the credential out of whichever header carries it.
func presentedKey(r *http.Request) string {
	if k := r.Header.Get("x-api-key"); k != "" {
		return k
	}
	if a := r.Header.Get("Authorization"); a != "" {
		return strings.TrimSpace(strings.TrimPrefix(a, "Bearer "))
	}
	return ""
}

// writeJSON sends v with the given status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		// The status line is already sent, so the response cannot be corrected.
		slog.Error("encode response", "error", err)
	}
}

// writeError sends an Anthropic-shaped error body.
func writeError(w http.ResponseWriter, status int, kind, msg string) {
	writeJSON(w, status, wire.NewAPIError(kind, msg))
}

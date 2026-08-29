package server

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// authCodeTTL bounds how long an issued code stays redeemable.
const authCodeTTL = 5 * time.Minute

// authCode is one outstanding authorization code awaiting its token exchange.
type authCode struct {
	challenge   string
	redirectURI string
	expires     time.Time
}

// codeStore holds authorization codes between the authorize and token calls.
type codeStore struct {
	mu    sync.Mutex
	codes map[string]authCode
}

func newCodeStore() *codeStore {
	return &codeStore{codes: map[string]authCode{}}
}

// put records a code and drops any that have expired.
func (c *codeStore) put(code string, v authCode) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, existing := range c.codes {
		if time.Now().After(existing.expires) {
			delete(c.codes, k)
		}
	}
	c.codes[code] = v
}

// take returns a code and removes it, so a code is redeemable exactly once.
func (c *codeStore) take(code string) (authCode, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.codes[code]
	delete(c.codes, code)
	if !ok || time.Now().After(v.expires) {
		return authCode{}, false
	}
	return v, true
}

// handleAuthServerMetadata answers the RFC 8414 probe Claude Desktop makes when
// inferenceGatewayOidc is unset, which is the gateway-as-authorization-server
// path. Returning 404 here leaves the sign-in screen with nowhere to go.
func (s *Server) handleAuthServerMetadata(w http.ResponseWriter, _ *http.Request) {
	base := strings.TrimSuffix(s.cfg.Server.PublicURL, "/")
	writeJSON(w, http.StatusOK, map[string]any{
		"issuer":                                base,
		"authorization_endpoint":                base + "/oauth/authorize",
		"token_endpoint":                        base + "/oauth/token",
		"response_types_supported":              []string{"code"},
		"grant_types_supported":                 []string{"authorization_code"},
		"code_challenge_methods_supported":      []string{"S256"},
		"token_endpoint_auth_methods_supported": []string{"none"},
	})
}

// handleAuthorize approves the sign-in and redirects back with a code.
//
// There is no consent screen and no user directory: the gateway's credential is
// the static API key, which the bootstrap document already hands to this same
// client, so the only thing the flow establishes is possession of the PKCE
// verifier. The redirect target is restricted to loopback to keep the endpoint
// from being used as an open redirector.
func (s *Server) handleAuthorize(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	redirectURI := q.Get("redirect_uri")
	if redirectURI == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "redirect_uri is required")
		return
	}
	target, err := url.Parse(redirectURI)
	if err != nil || !isLoopbackRedirect(target) {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"redirect_uri must be a loopback address")
		return
	}
	if rt := q.Get("response_type"); rt != "code" {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"only response_type=code is supported")
		return
	}
	challenge := q.Get("code_challenge")
	if challenge == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"code_challenge is required")
		return
	}
	if m := q.Get("code_challenge_method"); m != "S256" {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"only code_challenge_method=S256 is supported")
		return
	}

	code, err := randomToken()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", "could not issue an authorization code")
		return
	}
	s.codes.put(code, authCode{
		challenge:   challenge,
		redirectURI: redirectURI,
		expires:     time.Now().Add(authCodeTTL),
	})

	back := target.Query()
	back.Set("code", code)
	if state := q.Get("state"); state != "" {
		back.Set("state", state)
	}
	target.RawQuery = back.Encode()
	s.log.Info("oauth authorize", "redirect", target.Host)
	http.Redirect(w, r, target.String(), http.StatusFound)
}

// handleToken exchanges a code plus its PKCE verifier for the bearer credential.
func (s *Server) handleToken(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "malformed form body")
		return
	}
	if g := r.PostFormValue("grant_type"); g != "authorization_code" {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"only grant_type=authorization_code is supported")
		return
	}
	stored, ok := s.codes.take(r.PostFormValue("code"))
	if !ok {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"authorization code is unknown, expired, or already redeemed")
		return
	}
	if uri := r.PostFormValue("redirect_uri"); uri != "" && uri != stored.redirectURI {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"redirect_uri does not match the authorization request")
		return
	}
	verifier := r.PostFormValue("code_verifier")
	if verifier == "" || !verifierMatches(verifier, stored.challenge) {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"code_verifier does not match the code_challenge")
		return
	}

	// The bootstrap document already published this value to the same client.
	token := s.cfg.Server.APIKey
	if token == "" {
		token = "liberated"
	}
	s.log.Info("oauth token issued")
	writeJSON(w, http.StatusOK, map[string]any{
		"access_token": token,
		"token_type":   "Bearer",
		"expires_in":   int((365 * 24 * time.Hour).Seconds()),
	})
}

// isLoopbackRedirect reports whether the sign-in callback stays on this machine.
func isLoopbackRedirect(u *url.URL) bool {
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	switch u.Hostname() {
	case "127.0.0.1", "::1", "localhost":
		return true
	}
	return false
}

// verifierMatches checks a PKCE verifier against its S256 challenge.
func verifierMatches(verifier, challenge string) bool {
	sum := sha256.Sum256([]byte(verifier))
	want := base64.RawURLEncoding.EncodeToString(sum[:])
	return subtle.ConstantTimeCompare([]byte(want), []byte(challenge)) == 1
}

// randomToken returns a URL-safe 256-bit value.
func randomToken() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

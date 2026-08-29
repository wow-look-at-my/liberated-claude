package server

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"html"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"
)

const (
	// authCodeTTL bounds how long an issued code stays redeemable.
	authCodeTTL = 5 * time.Minute
	// deviceCodeTTL bounds how long a device grant may be polled for.
	deviceCodeTTL = 10 * time.Minute
	// deviceGrantType is the RFC 8628 grant identifier.
	deviceGrantType = "urn:ietf:params:oauth:grant-type:device_code"
)

// authCode is one outstanding authorization code awaiting its token exchange.
type authCode struct {
	challenge   string
	redirectURI string
	expires     time.Time
}

// deviceGrant is one outstanding device code awaiting its first poll.
type deviceGrant struct {
	userCode string
	expires  time.Time
}

// codeStore holds codes between the call that issues one and the token call
// that redeems it.
type codeStore struct {
	mu      sync.Mutex
	codes   map[string]authCode
	devices map[string]deviceGrant
}

func newCodeStore() *codeStore {
	return &codeStore{
		codes:   map[string]authCode{},
		devices: map[string]deviceGrant{},
	}
}

// putDevice records a device grant and drops any that have expired.
func (c *codeStore) putDevice(code string, v deviceGrant) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, existing := range c.devices {
		if time.Now().After(existing.expires) {
			delete(c.devices, k)
		}
	}
	c.devices[code] = v
}

// takeDevice returns a device grant and removes it.
func (c *codeStore) takeDevice(code string) (deviceGrant, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.devices[code]
	delete(c.devices, code)
	if !ok || time.Now().After(v.expires) {
		return deviceGrant{}, false
	}
	return v, true
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
func (s *Server) handleAuthServerMetadata(w http.ResponseWriter, r *http.Request) {
	base := requestOrigin(r)
	writeJSON(w, http.StatusOK, map[string]any{
		"issuer":                        base,
		"authorization_endpoint":        base + "/oauth/authorize",
		"token_endpoint":                base + "/oauth/token",
		"device_authorization_endpoint": base + "/oauth/device_authorization",
		"response_types_supported":      []string{"code"},
		"grant_types_supported": []string{
			"authorization_code",
			deviceGrantType,
		},
		"code_challenge_methods_supported":      []string{"S256"},
		"token_endpoint_auth_methods_supported": []string{"none"},
	})
}

// handleDeviceAuthorization starts an RFC 8628 device grant.
//
// The grant is approved the moment it is issued. There is no second party to
// consent: the gateway's credential is the static API key it already published
// to this client, so the verification page confirms rather than authorizes.
func (s *Server) handleDeviceAuthorization(w http.ResponseWriter, r *http.Request) {
	deviceCode, err := randomToken()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", "could not issue a device code")
		return
	}
	userCode, err := randomUserCode()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "api_error", "could not issue a user code")
		return
	}
	s.codes.putDevice(deviceCode, deviceGrant{
		userCode: userCode,
		expires:  time.Now().Add(deviceCodeTTL),
	})

	base := requestOrigin(r)
	verify := base + "/oauth/device"
	s.log.Info("oauth device grant", "user_code", userCode)
	writeJSON(w, http.StatusOK, map[string]any{
		"device_code":               deviceCode,
		"user_code":                 userCode,
		"verification_uri":          verify,
		"verification_uri_complete": verify + "?user_code=" + url.QueryEscape(userCode),
		"interval":                  1,
		"expires_in":                int(deviceCodeTTL.Seconds()),
	})
}

// handleDeviceVerification is the page the sign-in prompt sends the user to.
func (s *Server) handleDeviceVerification(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	code := r.URL.Query().Get("user_code")
	body := "<!doctype html><meta charset=utf-8><title>Liberated Claude</title>" +
		"<body style=\"font-family:system-ui;margin:4rem auto;max-width:32rem\">" +
		"<h1>Signed in</h1><p>Code <code>" + html.EscapeString(code) +
		"</code> is approved. You can close this tab and return to Claude.</p>"
	if _, err := io.WriteString(w, body); err != nil {
		s.log.Warn("device verification page", "error", err)
	}
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
	switch g := r.PostFormValue("grant_type"); g {
	case "authorization_code":
	case deviceGrantType:
		s.completeDeviceGrant(w, r)
		return
	default:
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"unsupported grant_type "+g)
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

	s.writeToken(w)
}

// completeDeviceGrant answers a device-code poll. An unknown code is reported
// with the OAuth error body the poller expects, not an Anthropic-shaped one.
func (s *Server) completeDeviceGrant(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.codes.takeDevice(r.PostFormValue("device_code")); !ok {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error":             "expired_token",
			"error_description": "device code is unknown, expired, or already redeemed",
		})
		return
	}
	s.writeToken(w)
}

// writeToken hands back the bearer both grants end at. The bootstrap document
// already published this value to the same client.
func (s *Server) writeToken(w http.ResponseWriter) {
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

// randomUserCode returns a short code a person can read off a screen.
func randomUserCode() (string, error) {
	const alphabet = "BCDFGHJKLMNPQRSTVWXZ23456789"
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	out := make([]byte, 0, 9)
	for i, b := range buf {
		if i == 4 {
			out = append(out, '-')
		}
		out = append(out, alphabet[int(b)%len(alphabet)])
	}
	return string(out), nil
}

// requestOrigin echoes back the origin the client dialled. The app requires the
// metadata to be same-origin with inferenceGatewayBaseUrl, and localhost and
// 127.0.0.1 are different origins.
func requestOrigin(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return scheme + "://" + r.Host
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

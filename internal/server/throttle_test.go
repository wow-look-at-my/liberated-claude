package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
)

// throttleConfig builds a one-provider config pointed at upstream, with the
// given concurrency cap.
func throttleConfig(t *testing.T, upstream string, maxConcurrent int) *config.Config {
	t.Helper()
	raw := `<liberatedClaude>
		<server><listen>127.0.0.1:0</listen></server>
		<providers>
			<provider name="p" kind="openai" maxConcurrent="` +
		strings.TrimSpace(itoa(maxConcurrent)) + `">
				<baseURL>` + upstream + `</baseURL>
				<models><model id="m" tier="sonnet" contextWindow="200000"/></models>
			</provider>
		</providers>
	</liberatedClaude>`
	cfg, err := config.Parse([]byte(raw))
	require.NoError(t, err)
	return cfg
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// postMessage sends a minimal Messages request to h.
func postMessage(h http.Handler) *httptest.ResponseRecorder {
	body := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestProviderConcurrencyIsCapped(t *testing.T) {
	var inFlight, peak int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		now := atomic.AddInt64(&inFlight, 1)
		for {
			old := atomic.LoadInt64(&peak)
			if now <= old || atomic.CompareAndSwapInt64(&peak, old, now) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		atomic.AddInt64(&inFlight, -1)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer upstream.Close()

	cfg := throttleConfig(t, upstream.URL, 2)
	h := New(cfg, upstream.Client(), slog.New(slog.DiscardHandler)).Handler()

	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			require.Equal(t, http.StatusOK, postMessage(h).Code)
		}()
	}
	wg.Wait()

	require.LessOrEqual(t, atomic.LoadInt64(&peak), int64(2),
		"the gate must never let more than maxConcurrent reach the provider")
}

func TestThrottledUpstreamIsRetried(t *testing.T) {
	var calls int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Every attempt must carry the body, or the retry sends an empty request.
		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Contains(t, string(raw), `"model"`)

		if atomic.AddInt64(&calls, 1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, `{"error":"too many concurrent requests"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer upstream.Close()

	cfg := throttleConfig(t, upstream.URL, 1)
	h := New(cfg, upstream.Client(), slog.New(slog.DiscardHandler)).Handler()

	w := postMessage(h)
	require.Equal(t, http.StatusOK, w.Code)
	require.EqualValues(t, 2, atomic.LoadInt64(&calls), "the 429 is retried once, then succeeds")

	var got map[string]interface{}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got))
	require.Equal(t, "message", got["type"])
}

func TestPersistentThrottleIsRelayedNotHidden(t *testing.T) {
	var calls int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt64(&calls, 1)
		w.Header().Set("Retry-After", "0")
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, `{"error":"too many concurrent requests"}`)
	}))
	defer upstream.Close()

	cfg := throttleConfig(t, upstream.URL, 1)
	h := New(cfg, upstream.Client(), slog.New(slog.DiscardHandler)).Handler()

	w := postMessage(h)
	require.Equal(t, http.StatusTooManyRequests, w.Code,
		"an upstream that stays throttled reports 429, it is not swallowed")
	require.EqualValues(t, len(retryBackoff)+1, atomic.LoadInt64(&calls))
}

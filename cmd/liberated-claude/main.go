// Command liberated-claude serves the bootstrap URL, model discovery, and
// Messages API that Claude Desktop expects, routing to whatever providers the
// XML config names.
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/pricing"
	"github.com/wow-look-at-my/liberated-claude/internal/server"
)

// upstreamTimeout bounds a proxied call (generous for long-context/slow providers).
const upstreamTimeout = 30 * time.Minute

// discoveryTimeout bounds the startup pricing fetch.
const discoveryTimeout = 20 * time.Second

func main() {
	configPath := flag.String("config", "config.xml", "path to the XML config file")
	verbose := flag.Bool("verbose", false, "log at debug level")
	flag.Parse()

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	if err := run(*configPath, log); err != nil {
		log.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run(configPath string, log *slog.Logger) error {
	cfg, err := config.Load(configPath)
	if err != nil {
		return err
	}
	logModels(cfg, log)

	client := &http.Client{Timeout: upstreamTimeout}
	srv := server.New(cfg, client, log)
	srv.SetRates(detectRates(cfg, log))

	httpSrv := &http.Server{
		Addr:              cfg.Server.Listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 30 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errc := make(chan error, 1)
	go func() {
		tls := cfg.Server.TLSCert != "" && cfg.Server.TLSKey != ""
		log.Info("listening",
			"addr", cfg.Server.Listen,
			"tls", tls,
			"bootstrapUrl", strings.TrimRight(cfg.Server.PublicURL, "/")+"/bootstrap")
		if tls {
			errc <- httpSrv.ListenAndServeTLS(cfg.Server.TLSCert, cfg.Server.TLSKey)
			return
		}
		errc <- httpSrv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		if err != nil && err != http.ErrServerClosed {
			return err
		}
		return nil
	case <-ctx.Done():
		log.Info("shutting down")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return httpSrv.Shutdown(shutdownCtx)
	}
}

// logModels reports what will be advertised, calling out the long-context
// models so a misconfigured contextWindow is visible at startup rather than as
// a silently truncated conversation later.
func logModels(cfg *config.Config, log *slog.Logger) {
	for _, s := range cfg.Skipped {
		log.Warn("provider skipped: credentials not in the environment",
			"provider", s.Name, "unset", strings.Join(s.Missing, ", "))
	}
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		if p.MaxConcurrent > 0 {
			log.Info("provider concurrency capped", "provider", p.Name, "maxConcurrent", p.MaxConcurrent)
		}
	}
	for _, m := range cfg.Models() {
		log.Info("model",
			"upstream", m.ID,
			"advertised", m.AliasID(),
			"tier", m.Tier,
			"contextWindow", m.ContextWindow,
			"offers1m", m.SupportsOneM(),
			"cache", m.EffectiveCache())
	}
}

// detectRates asks each provider what it charges so Claude Desktop can show a
// real cost instead of estimating every model at Anthropic list price.
//
// A provider that fails to answer is logged and skipped: pricing is a display
// concern, and refusing to start over it would take the gateway down for a
// cosmetic reason.
func detectRates(cfg *config.Config, log *slog.Logger) map[string]pricing.Rates {
	ctx, cancel := context.WithTimeout(context.Background(), discoveryTimeout)
	defer cancel()

	client := &http.Client{Timeout: discoveryTimeout}
	out := map[string]pricing.Rates{}
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		rates, err := fetchProviderRates(ctx, client, p)
		if err != nil {
			log.Warn("pricing detection failed", "provider", p.Name, "error", err)
			continue
		}
		for k, v := range rates {
			out[k] = v
		}
		log.Info("pricing detected", "provider", p.Name, "models", len(rates))
	}
	return out
}

// fetchProviderRates picks the detector matching the provider's API.
func fetchProviderRates(
	ctx context.Context, client *http.Client, p *config.Provider,
) (map[string]pricing.Rates, error) {
	host := strings.ToLower(p.BaseURL)
	switch {
	case strings.Contains(host, "openrouter.ai"):
		return pricing.FetchOpenRouter(ctx, client, p.BaseURL)
	case strings.Contains(host, "ollama.com"):
		return pricing.FetchOllamaCloud(ctx, client, p.BaseURL)
	default:
		// Anthropic and most OpenAI-compatible endpoints publish no price list.
		return nil, nil
	}
}

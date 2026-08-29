// Package pricing fetches and manages model pricing data from various sources.
package pricing

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
)

// Rates holds the pricing information for a model across input, output, and cache operations.
type Rates struct {
	InputPerMtok      float64
	OutputPerMtok     float64
	CacheReadPerMtok  float64
	CacheWritePerMtok float64
}

// Valid returns true if all rate fields are in the range [0, 10000] and none is NaN or Inf.
func (r Rates) Valid() bool {
	rates := []float64{r.InputPerMtok, r.OutputPerMtok, r.CacheReadPerMtok, r.CacheWritePerMtok}
	for _, rate := range rates {
		if math.IsNaN(rate) || math.IsInf(rate, 0) {
			return false
		}
		if rate < 0 || rate > 10000 {
			return false
		}
	}
	return true
}

// FetchOpenRouter retrieves model pricing from OpenRouter.
// The pricing values in the response are strings holding USD per token;
// they are converted to USD per million tokens for the returned Rates.
// Missing cache_write pricing is treated as 0.
// Negative or unparseable price strings are treated as 0 for that field.
// Returns a map keyed by OpenRouter model id.
func FetchOpenRouter(ctx context.Context, client *http.Client, baseURL string) (map[string]Rates, error) {
	baseURL = strings.TrimSuffix(baseURL, "/")
	url := baseURL + "/models"

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("fetch models: status %d: %s", resp.StatusCode, string(body))
	}

	var envelope struct {
		Data []struct {
			ID      string `json:"id"`
			Pricing struct {
				Prompt          string `json:"prompt"`
				Completion      string `json:"completion"`
				InputCacheRead  string `json:"input_cache_read"`
				InputCacheWrite string `json:"input_cache_write"`
			} `json:"pricing"`
		} `json:"data"`
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	result := make(map[string]Rates)
	for _, model := range envelope.Data {
		rates := Rates{
			InputPerMtok:     parsePrice(model.Pricing.Prompt),
			OutputPerMtok:    parsePrice(model.Pricing.Completion),
			CacheReadPerMtok: parsePrice(model.Pricing.InputCacheRead),
			// InputCacheWrite is frequently absent; treat as 0
			CacheWritePerMtok: parsePrice(model.Pricing.InputCacheWrite),
		}
		result[model.ID] = rates
	}

	return result, nil
}

// parsePrice converts a USD-per-token string to USD-per-million-tokens.
// Negative values and unparseable strings are treated as 0.
func parsePrice(s string) float64 {
	if s == "" {
		return 0
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	if f < 0 {
		return 0
	}
	return f * 1e6
}

// FetchOllamaCloud retrieves model ids from Ollama Cloud.
// Ollama Cloud publishes no per-token pricing information,
// so all returned Rates are zero. The map is keyed by model id.
func FetchOllamaCloud(ctx context.Context, client *http.Client, baseURL string) (map[string]Rates, error) {
	baseURL = strings.TrimSuffix(baseURL, "/")
	url := baseURL + "/models"

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("fetch models: status %d: %s", resp.StatusCode, string(body))
	}

	var envelope struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	result := make(map[string]Rates)
	for _, model := range envelope.Data {
		result[model.ID] = Rates{}
	}

	return result, nil
}

// Merge combines detected pricing with override pricing, where overrides take precedence.
// Returns a new map; inputs are not mutated. The result is deterministic.
func Merge(detected map[string]Rates, overrides map[string]Rates) map[string]Rates {
	result := make(map[string]Rates)

	// Start with detected prices
	for id, rates := range detected {
		result[id] = rates
	}

	// Apply overrides
	for id, rates := range overrides {
		result[id] = rates
	}

	return result
}

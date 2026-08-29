package pricing

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRatesValid(t *testing.T) {
	tests := []struct {
		name     string
		rates    Rates
		expected bool
		message  string
	}{
		{
			"all zero",
			Rates{InputPerMtok: 0, OutputPerMtok: 0, CacheReadPerMtok: 0, CacheWritePerMtok: 0},
			true,
			"all-zero Rates should be valid",
		},
		{
			"all at boundary",
			Rates{InputPerMtok: 10000, OutputPerMtok: 10000, CacheReadPerMtok: 10000, CacheWritePerMtok: 10000},
			true,
			"all rates at 10000 should be valid",
		},
		{
			"input exceeds maximum",
			Rates{InputPerMtok: 10000.1, OutputPerMtok: 1, CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"input rate exceeding 10000 should be invalid",
		},
		{
			"output exceeds maximum",
			Rates{InputPerMtok: 1, OutputPerMtok: 10001, CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"output rate exceeding 10000 should be invalid",
		},
		{
			"cache read exceeds maximum",
			Rates{InputPerMtok: 1, OutputPerMtok: 1, CacheReadPerMtok: 10000.5, CacheWritePerMtok: 1},
			false,
			"cache read rate exceeding 10000 should be invalid",
		},
		{
			"cache write exceeds maximum",
			Rates{InputPerMtok: 1, OutputPerMtok: 1, CacheReadPerMtok: 1, CacheWritePerMtok: 10001},
			false,
			"cache write rate exceeding 10000 should be invalid",
		},
		{
			"negative input",
			Rates{InputPerMtok: -1, OutputPerMtok: 1, CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"negative input rate should be invalid",
		},
		{
			"negative output",
			Rates{InputPerMtok: 1, OutputPerMtok: -0.5, CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"negative output rate should be invalid",
		},
		{
			"negative cache read",
			Rates{InputPerMtok: 1, OutputPerMtok: 1, CacheReadPerMtok: -1e-6, CacheWritePerMtok: 1},
			false,
			"negative cache read rate should be invalid",
		},
		{
			"negative cache write",
			Rates{InputPerMtok: 1, OutputPerMtok: 1, CacheReadPerMtok: 1, CacheWritePerMtok: -1},
			false,
			"negative cache write rate should be invalid",
		},
		{
			"NaN input",
			Rates{InputPerMtok: math.NaN(), OutputPerMtok: 1, CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"NaN input rate should be invalid",
		},
		{
			"NaN output",
			Rates{InputPerMtok: 1, OutputPerMtok: math.NaN(), CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"NaN output rate should be invalid",
		},
		{
			"NaN cache read",
			Rates{InputPerMtok: 1, OutputPerMtok: 1, CacheReadPerMtok: math.NaN(), CacheWritePerMtok: 1},
			false,
			"NaN cache read rate should be invalid",
		},
		{
			"NaN cache write",
			Rates{InputPerMtok: 1, OutputPerMtok: 1, CacheReadPerMtok: 1, CacheWritePerMtok: math.NaN()},
			false,
			"NaN cache write rate should be invalid",
		},
		{
			"positive infinity input",
			Rates{InputPerMtok: math.Inf(1), OutputPerMtok: 1, CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"positive infinity input rate should be invalid",
		},
		{
			"negative infinity output",
			Rates{InputPerMtok: 1, OutputPerMtok: math.Inf(-1), CacheReadPerMtok: 1, CacheWritePerMtok: 1},
			false,
			"negative infinity output rate should be invalid",
		},
		{
			"typical valid pricing",
			Rates{InputPerMtok: 0.05, OutputPerMtok: 0.15, CacheReadPerMtok: 0.01, CacheWritePerMtok: 0.02},
			true,
			"typical pricing within range should be valid",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.rates.Valid(), tt.message)
		})
	}
}

func TestFetchOpenRouterHappyPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method, "should make GET request")
		assert.Equal(t, "/models", r.URL.Path, "should request /models endpoint")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{
			"data": [
				{
					"id": "z-ai/glm-5.3-flash",
					"pricing": {
						"prompt": "0.000000075",
						"completion": "0.00000025",
						"input_cache_read": "0.000000015",
						"input_cache_write": "0.0000001"
					}
				},
				{
					"id": "deepseek/deepseek-v4-flash-0731",
					"pricing": {
						"prompt": "0.00000001",
						"completion": "0.00000005",
						"input_cache_read": "0.000000002",
						"input_cache_write": "0.00000001"
					}
				}
			]
		}`)
	}))
	defer server.Close()

	result, err := FetchOpenRouter(context.Background(), http.DefaultClient, server.URL)
	require.NoError(t, err, "FetchOpenRouter should succeed")

	assert.Equal(t, 2, len(result), "should return two models")

	// Per-token to per-Mtok: 0.000000075 * 1e6 = 0.075 (tolerance for FP precision).
	const tol = 1e-12

	glm := result["z-ai/glm-5.3-flash"]
	assert.InDelta(t, 0.075, glm.InputPerMtok, tol, "input rate should convert correctly")
	assert.InDelta(t, 0.25, glm.OutputPerMtok, tol, "output rate should convert correctly")
	assert.InDelta(t, 0.015, glm.CacheReadPerMtok, tol, "cache read rate should convert correctly")
	assert.InDelta(t, 0.1, glm.CacheWritePerMtok, tol, "cache write rate should convert correctly")

	deepseek := result["deepseek/deepseek-v4-flash-0731"]
	assert.InDelta(t, 0.01, deepseek.InputPerMtok, tol, "input rate should convert correctly")
	assert.InDelta(t, 0.05, deepseek.OutputPerMtok, tol, "output rate should convert correctly")
	assert.InDelta(t, 0.002, deepseek.CacheReadPerMtok, tol, "cache read rate should convert correctly")
	assert.InDelta(t, 0.01, deepseek.CacheWritePerMtok, tol, "cache write rate should convert correctly")
}

func TestFetchOpenRouterMissingCacheWrite(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{
			"data": [
				{
					"id": "model-no-cache-write",
					"pricing": {
						"prompt": "0.00000001",
						"completion": "0.00000002",
						"input_cache_read": "0.000000003"
					}
				}
			]
		}`)
	}))
	defer server.Close()

	result, err := FetchOpenRouter(context.Background(), http.DefaultClient, server.URL)
	require.NoError(t, err, "FetchOpenRouter should succeed even without cache_write")

	rates := result["model-no-cache-write"]
	assert.Equal(t, 0.01, rates.InputPerMtok, "input rate should convert correctly")
	assert.Equal(t, 0.02, rates.OutputPerMtok, "output rate should convert correctly")
	assert.Equal(t, 0.003, rates.CacheReadPerMtok, "cache read rate should convert correctly")
	assert.Equal(t, float64(0), rates.CacheWritePerMtok, "missing cache write should default to 0")
}

func TestFetchOpenRouterNegativePrice(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{
			"data": [
				{
					"id": "model-with-negative",
					"pricing": {
						"prompt": "0.00000001",
						"completion": "-1",
						"input_cache_read": "0.000000002",
						"input_cache_write": "0"
					}
				}
			]
		}`)
	}))
	defer server.Close()

	result, err := FetchOpenRouter(context.Background(), http.DefaultClient, server.URL)
	require.NoError(t, err, "FetchOpenRouter should succeed with negative price")

	rates := result["model-with-negative"]
	assert.Equal(t, 0.01, rates.InputPerMtok, "input rate should convert correctly")
	assert.Equal(t, float64(0), rates.OutputPerMtok, "negative price should be treated as 0")
	assert.Equal(t, 0.002, rates.CacheReadPerMtok, "cache read rate should convert correctly")
	assert.Equal(t, float64(0), rates.CacheWritePerMtok, "zero should remain zero")
}

func TestFetchOpenRouterUnparseable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{
			"data": [
				{
					"id": "model-with-unparseable",
					"pricing": {
						"prompt": "invalid-number",
						"completion": "0.00000005",
						"input_cache_read": "also-invalid",
						"input_cache_write": "0.0000001"
					}
				}
			]
		}`)
	}))
	defer server.Close()

	result, err := FetchOpenRouter(context.Background(), http.DefaultClient, server.URL)
	require.NoError(t, err, "FetchOpenRouter should succeed with unparseable prices")

	rates := result["model-with-unparseable"]
	assert.Equal(t, float64(0), rates.InputPerMtok, "unparseable price should be treated as 0")
	assert.InDelta(t, 0.05, rates.OutputPerMtok, 1e-12, "valid price should still be parsed")
	assert.Equal(t, float64(0), rates.CacheReadPerMtok, "unparseable cache read should be treated as 0")
	assert.InDelta(t, 0.1, rates.CacheWritePerMtok, 1e-12, "valid cache write should still be parsed")
}

func TestFetchOpenRouterNon200(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, "invalid key")
	}))
	defer server.Close()

	_, err := FetchOpenRouter(context.Background(), http.DefaultClient, server.URL)
	assert.Error(t, err, "FetchOpenRouter should error on non-200 status")
	assert.Contains(t, err.Error(), "401", "error should mention the status code")
}

func TestFetchOpenRouterMalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{invalid json`)
	}))
	defer server.Close()

	_, err := FetchOpenRouter(context.Background(), http.DefaultClient, server.URL)
	assert.Error(t, err, "FetchOpenRouter should error on malformed JSON")
	assert.Contains(t, err.Error(), "parse response", "error should indicate parsing failure")
}

func TestFetchOpenRouterTrailingSlash(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the path doesn't have double slash
		assert.Equal(t, "/models", r.URL.Path, "should not create double slash in path")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"data": []}`)
	}))
	defer server.Close()

	_, err := FetchOpenRouter(context.Background(), http.DefaultClient, server.URL+"/")
	require.NoError(t, err, "FetchOpenRouter should handle trailing slash")
}

func TestFetchOllamaCloudHappyPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method, "should make GET request")
		assert.Equal(t, "/models", r.URL.Path, "should request /models endpoint")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{
			"object": "list",
			"data": [
				{"id": "glm-5.3-flash", "object": "model"},
				{"id": "deepseek-v4-flash-0731", "object": "model"}
			]
		}`)
	}))
	defer server.Close()

	result, err := FetchOllamaCloud(context.Background(), http.DefaultClient, server.URL)
	require.NoError(t, err, "FetchOllamaCloud should succeed")

	assert.Equal(t, 2, len(result), "should return two models")

	// All rates should be zero
	glm := result["glm-5.3-flash"]
	assert.Equal(t, Rates{}, glm, "model should have all-zero Rates")

	deepseek := result["deepseek-v4-flash-0731"]
	assert.Equal(t, Rates{}, deepseek, "model should have all-zero Rates")
}

func TestFetchOllamaCloudNon200(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, "not found")
	}))
	defer server.Close()

	_, err := FetchOllamaCloud(context.Background(), http.DefaultClient, server.URL)
	assert.Error(t, err, "FetchOllamaCloud should error on non-200 status")
	assert.Contains(t, err.Error(), "404", "error should mention the status code")
}

func TestFetchOllamaCloudMalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{broken json`)
	}))
	defer server.Close()

	_, err := FetchOllamaCloud(context.Background(), http.DefaultClient, server.URL)
	assert.Error(t, err, "FetchOllamaCloud should error on malformed JSON")
	assert.Contains(t, err.Error(), "parse response", "error should indicate parsing failure")
}

func TestFetchOllamaCloudTrailingSlash(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/models", r.URL.Path, "should not create double slash in path")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"data": []}`)
	}))
	defer server.Close()

	_, err := FetchOllamaCloud(context.Background(), http.DefaultClient, server.URL+"/")
	require.NoError(t, err, "FetchOllamaCloud should handle trailing slash")
}

func TestMergeOverridePrecedence(t *testing.T) {
	detected := map[string]Rates{
		"model-a": {InputPerMtok: 0.1, OutputPerMtok: 0.2, CacheReadPerMtok: 0.01, CacheWritePerMtok: 0.02},
		"model-b": {InputPerMtok: 0.05, OutputPerMtok: 0.1, CacheReadPerMtok: 0.005, CacheWritePerMtok: 0.01},
	}

	overrides := map[string]Rates{
		"model-a": {InputPerMtok: 0.15, OutputPerMtok: 0.25, CacheReadPerMtok: 0.015, CacheWritePerMtok: 0.025},
		"model-c": {InputPerMtok: 1, OutputPerMtok: 2, CacheReadPerMtok: 0.1, CacheWritePerMtok: 0.2},
	}

	result := Merge(detected, overrides)

	// model-a should use override
	assert.Equal(t, 0.15, result["model-a"].InputPerMtok, "override should win for model-a input")
	assert.Equal(t, 0.25, result["model-a"].OutputPerMtok, "override should win for model-a output")

	// model-b should use detected
	assert.Equal(t, 0.05, result["model-b"].InputPerMtok, "detected should be used for model-b input")
	assert.Equal(t, 0.1, result["model-b"].OutputPerMtok, "detected should be used for model-b output")

	// model-c should be from override
	assert.Equal(t, float64(1), result["model-c"].InputPerMtok, "override-only model should be present")

	assert.Equal(t, 3, len(result), "result should have all three models")
}

func TestMergeNonMutation(t *testing.T) {
	detected := map[string]Rates{
		"model-a": {InputPerMtok: 0.1, OutputPerMtok: 0.2, CacheReadPerMtok: 0.01, CacheWritePerMtok: 0.02},
	}

	overrides := map[string]Rates{
		"model-b": {InputPerMtok: 0.5, OutputPerMtok: 1, CacheReadPerMtok: 0.05, CacheWritePerMtok: 0.1},
	}

	Merge(detected, overrides)

	// Verify inputs were not mutated
	assert.Equal(t, 1, len(detected), "detected should not be mutated")
	assert.Equal(t, 1, len(overrides), "overrides should not be mutated")
	assert.NotContains(t, detected, "model-b", "detected should not contain model-b")
	assert.NotContains(t, overrides, "model-a", "overrides should not contain model-a")
}

func TestMergeDeterministic(t *testing.T) {
	detected := map[string]Rates{
		"model-a": {InputPerMtok: 0.1, OutputPerMtok: 0.2, CacheReadPerMtok: 0.01, CacheWritePerMtok: 0.02},
		"model-b": {InputPerMtok: 0.05, OutputPerMtok: 0.1, CacheReadPerMtok: 0.005, CacheWritePerMtok: 0.01},
		"model-c": {InputPerMtok: 0.08, OutputPerMtok: 0.15, CacheReadPerMtok: 0.008, CacheWritePerMtok: 0.015},
	}

	overrides := map[string]Rates{
		"model-d": {InputPerMtok: 1, OutputPerMtok: 2, CacheReadPerMtok: 0.1, CacheWritePerMtok: 0.2},
	}

	result1 := Merge(detected, overrides)
	result2 := Merge(detected, overrides)

	// Both results should have the same keys
	assert.Equal(t, len(result1), len(result2), "both merges should have same number of keys")

	// All values should be identical
	for key := range result1 {
		assert.Equal(t, result1[key], result2[key], fmt.Sprintf("key %q should have same value in both merges", key))
	}
}

func TestMergeEmptyInputs(t *testing.T) {
	emptyDetected := make(map[string]Rates)
	emptyOverrides := make(map[string]Rates)
	populated := map[string]Rates{
		"model-a": {InputPerMtok: 0.1, OutputPerMtok: 0.2, CacheReadPerMtok: 0.01, CacheWritePerMtok: 0.02},
	}

	result1 := Merge(emptyDetected, populated)
	assert.Equal(t, 1, len(result1), "merge of empty detected with populated overrides should return overrides")

	result2 := Merge(populated, emptyOverrides)
	assert.Equal(t, 1, len(result2), "merge of populated detected with empty overrides should return detected")

	result3 := Merge(emptyDetected, emptyOverrides)
	assert.Equal(t, 0, len(result3), "merge of two empty maps should return empty map")
}

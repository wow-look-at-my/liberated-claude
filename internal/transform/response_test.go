package transform

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// TestOpenAIToAnthropicTextOnly verifies text-only response translation.
func TestOpenAIToAnthropicTextOnly(t *testing.T) {
	resp := &wire.OAResponse{
		ID:      "chatcmpl-123",
		Created: 1234567890,
		Choices: []wire.OAChoice{
			{
				Index: 0,
				Message: &wire.OAMessage{
					Role:    "assistant",
					Content: json.RawMessage(`"hello world"`),
				},
				FinishReason: ptrString("stop"),
			},
		},
		Usage: &wire.OAUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	result, err := OpenAIToAnthropic(resp, "claude-3-sonnet")
	assert.NoError(t, err, "should translate without error")
	assert.NotNil(t, result, "result should not be nil")
	assert.Equal(t, "assistant", result.Role, "role should be assistant")
	assert.Equal(t, "message", result.Type, "type should be message")
	assert.Equal(t, "claude-3-sonnet", result.Model, "model should match advertisedModel")
	assert.Equal(t, "end_turn", result.StopReason, "stop_reason should map 'stop' to 'end_turn'")
	assert.Len(t, result.Content, 1, "should have one content block")
	assert.Equal(t, "text", result.Content[0].Type, "content should be text")
	assert.Equal(t, "hello world", result.Content[0].Text, "text should match")
	assert.Equal(t, 10, result.Usage.InputTokens, "input tokens should be preserved")
	assert.Equal(t, 5, result.Usage.OutputTokens, "output tokens should be preserved")
}

// TestOpenAIToAnthropicReasoningContent verifies reasoning_content becomes a thinking block.
func TestOpenAIToAnthropicReasoningContent(t *testing.T) {
	resp := &wire.OAResponse{
		ID: "chatcmpl-456",
		Choices: []wire.OAChoice{
			{
				Index: 0,
				Message: &wire.OAMessage{
					Role:             "assistant",
					ReasoningContent: "let me think about this",
					Content:          json.RawMessage(`"the answer is 42"`),
				},
				FinishReason: ptrString("stop"),
			},
		},
		Usage: &wire.OAUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
		},
	}

	result, err := OpenAIToAnthropic(resp, "test-model")
	assert.NoError(t, err, "should translate without error")
	assert.Equal(t, 2, len(result.Content), "should have thinking and text blocks")
	assert.Equal(t, "thinking", result.Content[0].Type, "first block should be thinking")
	assert.Equal(t, "let me think about this", result.Content[0].Thinking, "thinking content should match")
	assert.Equal(t, "text", result.Content[1].Type, "second block should be text")
	assert.Equal(t, "the answer is 42", result.Content[1].Text, "text should match")
}

// TestOpenAIToAnthropicToolCalls verifies tool calls are translated to tool_use blocks.
func TestOpenAIToAnthropicToolCalls(t *testing.T) {
	resp := &wire.OAResponse{
		ID: "chatcmpl-789",
		Choices: []wire.OAChoice{
			{
				Index: 0,
				Message: &wire.OAMessage{
					Role: "assistant",
					ToolCalls: []wire.OAToolCall{
						{
							ID:   "call_abc123",
							Type: "function",
							Function: wire.OAFunctionCall{
								Name:      "get_weather",
								Arguments: `{"city":"San Francisco","unit":"celsius"}`,
							},
						},
					},
				},
				FinishReason: ptrString("tool_calls"),
			},
		},
		Usage: &wire.OAUsage{
			PromptTokens:     15,
			CompletionTokens: 3,
		},
	}

	result, err := OpenAIToAnthropic(resp, "test-model")
	assert.NoError(t, err, "should translate without error")
	assert.Len(t, result.Content, 1, "should have one tool_use block")
	assert.Equal(t, "tool_use", result.Content[0].Type, "content should be tool_use")
	assert.Equal(t, "call_abc123", result.Content[0].ID, "tool ID should match")
	assert.Equal(t, "get_weather", result.Content[0].Name, "tool name should match")

	// Verify Input is valid JSON.
	var input map[string]interface{}
	err = json.Unmarshal(result.Content[0].Input, &input)
	assert.NoError(t, err, "input should be valid JSON")
	assert.Equal(t, "San Francisco", input["city"], "input city should match")
	assert.Equal(t, "celsius", input["unit"], "input unit should match")

	// Verify stop_reason mapping for tool_calls.
	assert.Equal(t, "tool_use", result.StopReason, "stop_reason should map 'tool_calls' to 'tool_use'")
}

// TestOpenAIToAnthropicStopReasonMappings verifies all stop reason mappings.
func TestOpenAIToAnthropicStopReasonMappings(t *testing.T) {
	tests := []struct {
		openaiReason string
		expected     string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"content_filter", "refusal"},
		{"", "end_turn"},
	}

	for _, tt := range tests {
		resp := &wire.OAResponse{
			ID: "test",
			Choices: []wire.OAChoice{
				{
					Index: 0,
					Message: &wire.OAMessage{
						Role:    "assistant",
						Content: json.RawMessage(`"test"`),
					},
					FinishReason: ptrString(tt.openaiReason),
				},
			},
		}

		result, err := OpenAIToAnthropic(resp, "test")
		assert.NoError(t, err, "should translate %s", tt.openaiReason)
		assert.Equal(t, tt.expected, result.StopReason, "should map %s to %s", tt.openaiReason, tt.expected)
	}
}

// TestOpenAIToAnthropicCacheAccounting verifies cached tokens are subtracted from input_tokens.
func TestOpenAIToAnthropicCacheAccounting(t *testing.T) {
	cached := 100
	written := 50

	resp := &wire.OAResponse{
		ID: "test",
		Choices: []wire.OAChoice{
			{
				Index: 0,
				Message: &wire.OAMessage{
					Role:    "assistant",
					Content: json.RawMessage(`"response"`),
				},
				FinishReason: ptrString("stop"),
			},
		},
		Usage: &wire.OAUsage{
			PromptTokens:     250, // includes 100 cached
			CompletionTokens: 50,
			PromptTokensDetails: &wire.OAPromptTokensDetails{
				CachedTokens:     cached,
				CacheWriteTokens: written,
			},
		},
	}

	result, err := OpenAIToAnthropic(resp, "test-model")
	assert.NoError(t, err, "should translate without error")

	// InputTokens should be PromptTokens minus cached.
	assert.Equal(t, 150, result.Usage.InputTokens, "input_tokens should exclude cached tokens (250 - 100)")

	// CacheReadInputTokens should be set to cached count.
	assert.NotNil(t, result.Usage.CacheReadInputTokens, "cache_read_input_tokens should be set")
	assert.Equal(t, 100, *result.Usage.CacheReadInputTokens, "cache_read_input_tokens should match cached count")

	// CacheCreationInputTokens should be set to written count.
	assert.NotNil(t, result.Usage.CacheCreationInputTokens, "cache_creation_input_tokens should be set")
	assert.Equal(t, 50, *result.Usage.CacheCreationInputTokens, "cache_creation_input_tokens should match written count")
}

// TestOpenAIToAnthropicNoCacheAccounting verifies zero cache values don't add nil pointers.
func TestOpenAIToAnthropicNoCacheAccounting(t *testing.T) {
	resp := &wire.OAResponse{
		ID: "test",
		Choices: []wire.OAChoice{
			{
				Index: 0,
				Message: &wire.OAMessage{
					Role:    "assistant",
					Content: json.RawMessage(`"response"`),
				},
				FinishReason: ptrString("stop"),
			},
		},
		Usage: &wire.OAUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
		},
	}

	result, err := OpenAIToAnthropic(resp, "test-model")
	assert.NoError(t, err, "should translate without error")
	assert.Equal(t, 100, result.Usage.InputTokens, "input_tokens should be 100")
	assert.Nil(t, result.Usage.CacheReadInputTokens, "cache_read_input_tokens should be nil when zero")
	assert.Nil(t, result.Usage.CacheCreationInputTokens, "cache_creation_input_tokens should be nil when zero")
}

// TestOpenAIToAnthropicErrorResponse verifies error handling.
func TestOpenAIToAnthropicErrorResponse(t *testing.T) {
	resp := &wire.OAResponse{
		Error: &wire.OAError{
			Message: "model not found",
			Type:    "invalid_request_error",
		},
	}

	result, err := OpenAIToAnthropic(resp, "test-model")
	assert.Error(t, err, "should return error")
	assert.Nil(t, result, "result should be nil on error")
	assert.Contains(t, err.Error(), "model not found", "error message should contain upstream message")
}

// TestOpenAIToAnthropicNoChoices verifies error on empty choices.
func TestOpenAIToAnthropicNoChoices(t *testing.T) {
	resp := &wire.OAResponse{
		ID:      "test",
		Choices: []wire.OAChoice{},
	}

	result, err := OpenAIToAnthropic(resp, "test-model")
	assert.Error(t, err, "should return error on no choices")
	assert.Nil(t, result, "result should be nil on error")
}

// TestOpenAIToAnthropicEmptyToolCallArguments verifies tool calls with empty arguments default to {}.
func TestOpenAIToAnthropicEmptyToolCallArguments(t *testing.T) {
	resp := &wire.OAResponse{
		ID: "test",
		Choices: []wire.OAChoice{
			{
				Index: 0,
				Message: &wire.OAMessage{
					Role: "assistant",
					ToolCalls: []wire.OAToolCall{
						{
							ID:   "call_xyz",
							Type: "function",
							Function: wire.OAFunctionCall{
								Name:      "get_current_time",
								Arguments: "", // Empty
							},
						},
					},
				},
				FinishReason: ptrString("tool_calls"),
			},
		},
	}

	result, err := OpenAIToAnthropic(resp, "test-model")
	assert.NoError(t, err, "should translate without error")
	assert.Equal(t, json.RawMessage("{}"), result.Content[0].Input, "empty arguments should default to {}")
}

// TestStreamOpenAIToAnthropicBasic verifies streaming translation with text content.
func TestStreamOpenAIToAnthropicBasic(t *testing.T) {
	// Construct a multi-chunk SSE stream.
	sseStream := strings.Join([]string{
		`data: {"id":"chatcmpl-123","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-123","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-123","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`,
		`data: [DONE]`,
	}, "\n")

	src := strings.NewReader(sseStream)
	var dst bytes.Buffer

	err := StreamOpenAIToAnthropic(&dst, src, "test-model")
	assert.NoError(t, err, "should stream without error")

	output := dst.String()

	// Verify event sequence includes critical event types.
	assert.Contains(t, output, "event: message_start", "should emit message_start")
	assert.Contains(t, output, "event: content_block_start", "should emit content_block_start")
	assert.Contains(t, output, "event: content_block_delta", "should emit content_block_delta")
	assert.Contains(t, output, "event: content_block_stop", "should emit content_block_stop")
	assert.Contains(t, output, "event: message_delta", "should emit message_delta")
	assert.Contains(t, output, "event: message_stop", "should emit message_stop")

	// Verify text content is present.
	assert.Contains(t, output, "hello", "should contain 'hello' text")
	assert.Contains(t, output, " world", "should contain ' world' text")

	// Verify stop_reason in message_delta.
	assert.Contains(t, output, `"stop_reason":"end_turn"`, "should have end_turn in message_delta")
}

// TestStreamOpenAIToAnthropicToolCalls verifies tool call streaming with fragments.
func TestStreamOpenAIToAnthropicToolCalls(t *testing.T) {
	// Simulate fragmented tool call arguments.
	sseStream := strings.Join([]string{
		`data: {"id":"chatcmpl-456","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"cit"}}]},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-456","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"arguments":"y\":\"SF\","}}]},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl-456","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"arguments":"\"unit\":\"c\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":15,"completion_tokens":3,"total_tokens":18}}`,
		`data: [DONE]`,
	}, "\n")

	src := strings.NewReader(sseStream)
	var dst bytes.Buffer

	err := StreamOpenAIToAnthropic(&dst, src, "test-model")
	assert.NoError(t, err, "should stream without error")

	output := dst.String()

	// Verify tool_use content block.
	assert.Contains(t, output, `"type":"tool_use"`, "should emit tool_use content block")
	assert.Contains(t, output, `"name":"get_weather"`, "should have correct tool name")

	// Verify input_json_delta fragments.
	assert.Contains(t, output, `"type":"input_json_delta"`, "should emit input_json_delta events")

	// Verify tool_use stop reason in message_delta.
	assert.Contains(t, output, `"stop_reason":"tool_use"`, "should map 'tool_calls' to 'tool_use'")
}

// TestStreamOpenAIToAnthropicReasoningContent verifies thinking blocks in stream.
func TestStreamOpenAIToAnthropicReasoningContent(t *testing.T) {
	sseStream := strings.Join([]string{
		`data: {"id":"test","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking about this"},"finish_reason":null}]}`,
		`data: {"id":"test","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"reasoning_content":" step by step"},"finish_reason":null}]}`,
		`data: {"id":"test","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"final answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`,
		`data: [DONE]`,
	}, "\n")

	src := strings.NewReader(sseStream)
	var dst bytes.Buffer

	err := StreamOpenAIToAnthropic(&dst, src, "test-model")
	assert.NoError(t, err, "should stream without error")

	output := dst.String()

	// Verify thinking content block.
	assert.Contains(t, output, `"type":"thinking"`, "should emit thinking content block")
	assert.Contains(t, output, `"type":"thinking_delta"`, "should emit thinking_delta events")

	// Verify reasoning content appears.
	assert.Contains(t, output, "thinking about this", "should contain reasoning content")
	assert.Contains(t, output, " step by step", "should contain continued reasoning content")

	// Verify text follows thinking.
	assert.Contains(t, output, "final answer", "should contain final text")
}

// TestStreamOpenAIToAnthropicCacheAccountingStreaming verifies cache accounting in streaming.
func TestStreamOpenAIToAnthropicCacheAccountingStreaming(t *testing.T) {
	cached := 100
	written := 50

	sseStream := strings.Join([]string{
		`data: {"id":"test","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":null}]}`,
		fmt.Sprintf(`data: {"id":"test","object":"text_completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":250,"completion_tokens":50,"total_tokens":300,"prompt_tokens_details":{"cached_tokens":%d,"cache_write_tokens":%d}}}`, cached, written),
		`data: [DONE]`,
	}, "\n")

	src := strings.NewReader(sseStream)
	var dst bytes.Buffer

	err := StreamOpenAIToAnthropic(&dst, src, "test-model")
	assert.NoError(t, err, "should stream without error")

	output := dst.String()

	// Verify cache accounting in message_delta.
	// Input tokens should be 250 - 100 = 150.
	assert.Contains(t, output, `"input_tokens":150`, "input_tokens should exclude cached tokens")
	assert.Contains(t, output, `"cache_read_input_tokens":100`, "cache_read_input_tokens should be set")
	assert.Contains(t, output, `"cache_creation_input_tokens":50`, "cache_creation_input_tokens should be set")
}

// TestStreamOpenAIToAnthropicErrorInStream verifies error handling for malformed SSE.
func TestStreamOpenAIToAnthropicErrorInStream(t *testing.T) {
	sseStream := `data: {"invalid json}`

	src := strings.NewReader(sseStream)
	var dst bytes.Buffer

	err := StreamOpenAIToAnthropic(&dst, src, "test-model")
	assert.Error(t, err, "should return error on malformed JSON")
	assert.Contains(t, err.Error(), "parse SSE event", "error should indicate parsing issue")
}

// Helper to create a string pointer.
func ptrString(s string) *string {
	return &s
}

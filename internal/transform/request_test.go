package transform

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

func TestAnthropicToOpenAI_SystemPromptBareString(t *testing.T) {
	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		System:    json.RawMessage([]byte(`"You are a helpful assistant"`)),
		Messages:  []wire.Message{},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on bare string system prompt")
	assert.Equal(t, "claude-3-5-sonnet-20241022", out.Model, "model should be upstream ID")
	assert.Len(t, out.Messages, 1, "should have system message")
	assert.Equal(t, "system", out.Messages[0].Role, "first message should be system")

	var systemText string
	err = json.Unmarshal(out.Messages[0].Content, &systemText)
	assert.NoError(t, err, "system content should be valid JSON string")
	assert.Equal(t, "You are a helpful assistant", systemText, "system text should match")
}

func TestAnthropicToOpenAI_SystemPromptArray(t *testing.T) {
	blocks := []wire.ContentBlock{
		{Type: "text", Text: "First part"},
		{Type: "text", Text: "Second part"},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		System:    json.RawMessage(blockJSON),
		Messages:  []wire.Message{},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on array system prompt")
	assert.Len(t, out.Messages, 1, "should have system message")

	var systemText string
	err = json.Unmarshal(out.Messages[0].Content, &systemText)
	assert.NoError(t, err, "system content should be valid JSON string")
	assert.Equal(t, "First part\n\nSecond part", systemText, "system text should concatenate blocks with newlines")
}

func TestAnthropicToOpenAI_UserMessageBareString(t *testing.T) {
	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage([]byte(`"Hello"`)),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on bare string user message")
	assert.Len(t, out.Messages, 1, "should have user message")
	assert.Equal(t, "user", out.Messages[0].Role, "message should be user role")

	var text string
	err = json.Unmarshal(out.Messages[0].Content, &text)
	assert.NoError(t, err, "content should be valid JSON string")
	assert.Equal(t, "Hello", text, "text should match")
}

func TestAnthropicToOpenAI_UserMessageArray(t *testing.T) {
	blocks := []wire.ContentBlock{
		{Type: "text", Text: "Part 1"},
		{Type: "text", Text: "Part 2"},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on array user message")
	assert.Len(t, out.Messages, 1, "should have user message")

	var parts []wire.OAContentPart
	err = json.Unmarshal(out.Messages[0].Content, &parts)
	assert.NoError(t, err, "content should be valid JSON array")
	assert.Len(t, parts, 2, "should have 2 parts")
	assert.Equal(t, "Part 1", parts[0].Text, "first part text should match")
	assert.Equal(t, "Part 2", parts[1].Text, "second part text should match")
}

func TestAnthropicToOpenAI_ImageBase64(t *testing.T) {
	source := map[string]string{
		"type":       "base64",
		"media_type": "image/png",
		"data":       "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
	}
	sourceJSON, _ := json.Marshal(source)

	blocks := []wire.ContentBlock{
		{
			Type:   "image",
			Source: sourceJSON,
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on base64 image")
	assert.Len(t, out.Messages, 1, "should have user message")

	var parts []wire.OAContentPart
	err = json.Unmarshal(out.Messages[0].Content, &parts)
	assert.NoError(t, err, "content should be valid JSON array")
	assert.Len(t, parts, 1, "should have 1 part")
	assert.Equal(t, "image_url", parts[0].Type, "part should be image_url type")

	var imageURL map[string]string
	json.Unmarshal(parts[0].ImageURL, &imageURL)
	assert.True(t, len(imageURL["url"]) > 0, "should have URL")
	assert.True(t, json.Valid([]byte(`"`+imageURL["url"]+`"`)), "URL should be in image_url object")
}

func TestAnthropicToOpenAI_ImageURL(t *testing.T) {
	source := map[string]string{
		"type": "url",
		"url":  "https://example.com/image.jpg",
	}
	sourceJSON, _ := json.Marshal(source)

	blocks := []wire.ContentBlock{
		{
			Type:   "image",
			Source: sourceJSON,
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on URL image")

	var parts []wire.OAContentPart
	json.Unmarshal(out.Messages[0].Content, &parts)
	assert.Equal(t, "image_url", parts[0].Type, "part should be image_url type")

	var imageURL map[string]string
	json.Unmarshal(parts[0].ImageURL, &imageURL)
	assert.Equal(t, "https://example.com/image.jpg", imageURL["url"], "URL should match")
}

func TestAnthropicToOpenAI_ToolUseAssistant(t *testing.T) {
	input := json.RawMessage([]byte(`{"query":"test"}`))
	blocks := []wire.ContentBlock{
		{
			Type: "text",
			Text: "I'll help you",
		},
		{
			Type:  "tool_use",
			ID:    "tool_123",
			Name:  "search",
			Input: input,
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "assistant",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on tool_use block")
	assert.Len(t, out.Messages, 1, "should have assistant message")

	msg := out.Messages[0]
	assert.Equal(t, "assistant", msg.Role, "message should be assistant role")
	assert.Len(t, msg.ToolCalls, 1, "should have 1 tool call")

	call := msg.ToolCalls[0]
	assert.Equal(t, "function", call.Type, "tool call type should be function")
	assert.Equal(t, "tool_123", call.ID, "tool call ID should match")
	assert.Equal(t, "search", call.Function.Name, "tool call function name should match")
	assert.Equal(t, `{"query":"test"}`, call.Function.Arguments, "tool call arguments should match")
}

func TestAnthropicToOpenAI_ToolUseEmptyInput(t *testing.T) {
	blocks := []wire.ContentBlock{
		{
			Type:  "tool_use",
			ID:    "tool_456",
			Name:  "get_time",
			Input: json.RawMessage([]byte{}),
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "assistant",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on empty tool_use input")

	call := out.Messages[0].ToolCalls[0]
	assert.Equal(t, "{}", call.Function.Arguments, "empty input should default to {}")
}

func TestAnthropicToOpenAI_ToolResult(t *testing.T) {
	blocks := []wire.ContentBlock{
		{
			Type:      "tool_result",
			ToolUseID: "tool_123",
			Content:   json.RawMessage([]byte(`"The result is 42"`)),
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on tool_result")
	assert.Len(t, out.Messages, 1, "should have 1 tool message")

	msg := out.Messages[0]
	assert.Equal(t, "tool", msg.Role, "message should be tool role")
	assert.Equal(t, "tool_123", msg.ToolCallID, "tool call ID should match")

	var content string
	json.Unmarshal(msg.Content, &content)
	assert.Equal(t, "The result is 42", content, "tool result content should match")
}

func TestAnthropicToOpenAI_ToolResultWithTextBefore(t *testing.T) {
	blocks := []wire.ContentBlock{
		{
			Type: "text",
			Text: "Let me look that up",
		},
		{
			Type:      "tool_result",
			ToolUseID: "tool_123",
			Content:   json.RawMessage([]byte(`"42"`)),
		},
		{
			Type: "text",
			Text: "The answer is ready",
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error with mixed content")
	assert.Len(t, out.Messages, 3, "should split into 3 messages: text, tool, text")

	assert.Equal(t, "user", out.Messages[0].Role, "first should be user with text")
	assert.Equal(t, "tool", out.Messages[1].Role, "second should be tool result")
	assert.Equal(t, "user", out.Messages[2].Role, "third should be user with remaining text")
}

func TestAnthropicToOpenAI_ThinkingBlockPreserved(t *testing.T) {
	blocks := []wire.ContentBlock{
		{
			Type:     "thinking",
			Thinking: "I should think about this",
		},
		{
			Type: "text",
			Text: "Here is my response",
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "assistant",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on thinking block")
	assert.Len(t, out.Messages, 1, "should have 1 message")

	var text string
	json.Unmarshal(out.Messages[0].Content, &text)
	assert.Equal(t, "Here is my response", text, "text block becomes the message content")
	assert.Equal(t, "I should think about this", out.Messages[0].ReasoningContent,
		"thinking block rides along as reasoning_content")
}

func TestAnthropicToOpenAI_MaxTokensClamping(t *testing.T) {
	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		Messages:  []wire.Message{},
		MaxTokens: 4096,
	}

	// Model has lower limit.
	m := &config.Model{
		ID:              "claude-3-5-sonnet-20241022",
		ContextWindow:   200000,
		MaxOutputTokens: 1024,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.Equal(t, 1024, *out.MaxTokens, "should clamp to model's MaxOutputTokens")
}

func TestAnthropicToOpenAI_MaxTokensNoModel(t *testing.T) {
	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		Messages:  []wire.Message{},
		MaxTokens: 2048,
	}

	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
		// MaxOutputTokens is 0, so no clamping.
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.Equal(t, 2048, *out.MaxTokens, "should use request's MaxTokens")
}

func TestAnthropicToOpenAI_StreamOptions(t *testing.T) {
	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		Messages:  []wire.Message{},
		MaxTokens: 1024,
		Stream:    true,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.True(t, out.Stream, "should preserve stream flag")
	assert.NotNil(t, out.StreamOptions, "should set StreamOptions when streaming")
	assert.True(t, out.StreamOptions.IncludeUsage, "should request usage in stream")
}

func TestAnthropicToOpenAI_CacheExplicitPreserved(t *testing.T) {
	blocks := []wire.ContentBlock{
		{
			Type: "text",
			Text: "Important context",
			CacheControl: &wire.CacheControl{
				Type: "ephemeral",
			},
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}

	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
		Cache:         config.CacheExplicit,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")

	var parts []wire.OAContentPart
	json.Unmarshal(out.Messages[0].Content, &parts)
	assert.Len(t, parts, 1, "should have 1 part")
	assert.NotNil(t, parts[0].CacheControl, "cache_control should be preserved in explicit mode")
	assert.Equal(t, "ephemeral", parts[0].CacheControl.Type, "cache_control type should match")
}

func TestAnthropicToOpenAI_CacheImplicitStripped(t *testing.T) {
	blocks := []wire.ContentBlock{
		{
			Type: "text",
			Text: "Important context",
			CacheControl: &wire.CacheControl{
				Type: "ephemeral",
			},
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}

	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
		Cache:         config.CacheImplicit,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")

	var text string
	json.Unmarshal(out.Messages[0].Content, &text)
	assert.Equal(t, "Important context", text, "cache_control should be stripped and text emitted as bare string")
}

func TestAnthropicToOpenAI_Tools(t *testing.T) {
	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		Messages:  []wire.Message{},
		MaxTokens: 1024,
		Tools: []wire.Tool{
			{
				Name:        "search",
				Description: "Search the web",
				InputSchema: json.RawMessage([]byte(`{"type":"object","properties":{"query":{"type":"string"}}}`)),
			},
		},
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.Len(t, out.Tools, 1, "should have 1 tool")

	tool := out.Tools[0]
	assert.Equal(t, "function", tool.Type, "tool type should be function")
	assert.Equal(t, "search", tool.Function.Name, "tool name should match")
	assert.Equal(t, "Search the web", tool.Function.Description, "tool description should match")
	assert.True(t, len(tool.Function.Parameters) > 0, "tool parameters should be preserved")
}

func TestAnthropicToOpenAI_ToolChoice(t *testing.T) {
	toolChoice := json.RawMessage([]byte(`{"type":"tool","name":"search"}`))
	req := &wire.MessagesRequest{
		Model:      "claude-3-5-sonnet-20241022",
		Messages:   []wire.Message{},
		MaxTokens:  1024,
		ToolChoice: toolChoice,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.Equal(t, toolChoice, out.ToolChoice, "tool choice should be passed through as raw JSON")
}

func TestAnthropicToOpenAI_Temperature(t *testing.T) {
	temp := 0.5
	req := &wire.MessagesRequest{
		Model:       "claude-3-5-sonnet-20241022",
		Messages:    []wire.Message{},
		MaxTokens:   1024,
		Temperature: &temp,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.NotNil(t, out.Temperature, "temperature should be set")
	assert.Equal(t, 0.5, *out.Temperature, "temperature value should match")
}

func TestAnthropicToOpenAI_TopP(t *testing.T) {
	topP := 0.9
	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		Messages:  []wire.Message{},
		MaxTokens: 1024,
		TopP:      &topP,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.NotNil(t, out.TopP, "top_p should be set")
	assert.Equal(t, 0.9, *out.TopP, "top_p value should match")
}

func TestAnthropicToOpenAI_StopSequences(t *testing.T) {
	req := &wire.MessagesRequest{
		Model:         "claude-3-5-sonnet-20241022",
		Messages:      []wire.Message{},
		MaxTokens:     1024,
		StopSequences: []string{"END", "STOP"},
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")
	assert.Equal(t, []string{"END", "STOP"}, out.Stop, "stop sequences should be preserved")
}

func TestAnthropicToOpenAI_InvalidSystemPrompt(t *testing.T) {
	req := &wire.MessagesRequest{
		Model:     "claude-3-5-sonnet-20241022",
		System:    json.RawMessage([]byte(`{invalid json`)),
		Messages:  []wire.Message{},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	_, err := AnthropicToOpenAI(req, m)
	assert.Error(t, err, "should error on invalid system prompt")
}

func TestAnthropicToOpenAI_InvalidMessageContent(t *testing.T) {
	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage([]byte(`{invalid`)),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	_, err := AnthropicToOpenAI(req, m)
	assert.Error(t, err, "should error on invalid message content")
}

func TestAnthropicToOpenAI_UnknownRole(t *testing.T) {
	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "unknown",
				Content: json.RawMessage([]byte(`"text"`)),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	_, err := AnthropicToOpenAI(req, m)
	assert.Error(t, err, "should error on unknown message role")
}

func TestAnthropicToOpenAI_AssistantCacheControl(t *testing.T) {
	blocks := []wire.ContentBlock{
		{
			Type: "text",
			Text: "Important context",
			CacheControl: &wire.CacheControl{
				Type: "ephemeral",
			},
		},
	}
	blockJSON, _ := json.Marshal(blocks)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "assistant",
				Content: json.RawMessage(blockJSON),
			},
		},
		MaxTokens: 1024,
	}

	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
		Cache:         config.CacheExplicit,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error")

	var parts []wire.OAContentPart
	json.Unmarshal(out.Messages[0].Content, &parts)
	assert.Len(t, parts, 1, "should have 1 part")
	assert.NotNil(t, parts[0].CacheControl, "cache_control should be preserved in explicit mode on assistant message")
	assert.Equal(t, "ephemeral", parts[0].CacheControl.Type, "cache_control type should match")
}

package transform

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

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

func TestAnthropicToOpenAI_ToolResultWithImage(t *testing.T) {
	source := map[string]string{
		"type":       "base64",
		"media_type": "image/png",
		"data":       "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
	}
	sourceJSON, _ := json.Marshal(source)

	resultContent := []wire.ContentBlock{
		{
			Type:   "image",
			Source: sourceJSON,
		},
		{
			Type: "text",
			Text: "Screenshot captured.",
		},
	}
	resultJSON, _ := json.Marshal(resultContent)

	req := &wire.MessagesRequest{
		Model: "claude-3-5-sonnet-20241022",
		Messages: []wire.Message{
			{
				Role:    "user",
				Content: json.RawMessage([]byte(`[{"type":"tool_result","tool_use_id":"tool_123","content":` + string(resultJSON) + `}]`)),
			},
		},
		MaxTokens: 1024,
	}
	m := &config.Model{
		ID:            "claude-3-5-sonnet-20241022",
		ContextWindow: 200000,
	}

	out, err := AnthropicToOpenAI(req, m)
	assert.NoError(t, err, "should not error on tool_result with image")
	assert.Len(t, out.Messages, 1, "should have tool message")
	assert.Equal(t, "tool", out.Messages[0].Role, "message should be tool role")
	assert.Equal(t, "tool_123", out.Messages[0].ToolCallID, "tool_call_id should match")

	var parts []wire.OAContentPart
	err = json.Unmarshal(out.Messages[0].Content, &parts)
	assert.NoError(t, err, "content should be valid JSON array")
	assert.Len(t, parts, 2, "should have image part and text part")
	assert.Equal(t, "image_url", parts[0].Type, "image should become image_url part")
	assert.Equal(t, "text", parts[1].Type, "text part should follow image part")
	assert.Equal(t, "Screenshot captured.", parts[1].Text, "text part should match")

	var imageURL map[string]string
	json.Unmarshal(parts[0].ImageURL, &imageURL)
	assert.Contains(t, imageURL["url"], "data:image/png;base64,", "image part should carry data URL")
}

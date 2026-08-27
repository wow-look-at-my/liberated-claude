package transform

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// AnthropicToOpenAI translates an Anthropic Messages request to OpenAI Chat Completions format.
func AnthropicToOpenAI(req *wire.MessagesRequest, m *config.Model) (*wire.OARequest, error) {
	out := &wire.OARequest{
		Model:       m.ID,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		ToolChoice:  req.ToolChoice,
	}

	// Clamp max_tokens.
	maxTokens := req.MaxTokens
	if m.MaxOutputTokens > 0 && req.MaxTokens > m.MaxOutputTokens {
		maxTokens = m.MaxOutputTokens
	}
	if maxTokens > 0 {
		out.MaxTokens = &maxTokens
	}

	// TopK has no OpenAI equivalent; drop silently (no alternative parameter exists).

	// StopSequences -> Stop.
	out.Stop = req.StopSequences

	// StreamOptions for cache accounting when streaming.
	if req.Stream {
		out.StreamOptions = &wire.OAStreamOptions{IncludeUsage: true}
	}

	// System prompt as leading message.
	if len(req.System) > 0 {
		systemText, err := reqSystemText(req.System, m.EffectiveCache())
		if err != nil {
			return nil, err
		}
		if systemText != "" {
			out.Messages = append(out.Messages, wire.OAMessage{
				Role:    "system",
				Content: json.RawMessage([]byte(fmt.Sprintf("%q", systemText))),
			})
		}
	}

	// Messages.
	for _, msg := range req.Messages {
		if msg.Role == "assistant" {
			// Assistant messages may contain tool_use blocks, which become ToolCalls.
			oamsg, err := reqAssistantMessage(msg, m.EffectiveCache())
			if err != nil {
				return nil, err
			}
			out.Messages = append(out.Messages, oamsg)
		} else if msg.Role == "user" {
			// User messages may contain tool_result blocks.
			msgs, err := reqUserMessages(msg, m.EffectiveCache())
			if err != nil {
				return nil, err
			}
			out.Messages = append(out.Messages, msgs...)
		} else {
			return nil, fmt.Errorf("unknown message role: %q", msg.Role)
		}
	}

	// Tools.
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, wire.OATool{
			Type: "function",
			Function: wire.OAFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}

	return out, nil
}

// reqSystemText extracts text from the system prompt, handling both string and array forms.
func reqSystemText(system json.RawMessage, mode config.CacheMode) (string, error) {
	if len(system) == 0 {
		return "", nil
	}

	// Try to parse as array of ContentBlock.
	var blocks []wire.ContentBlock
	if err := json.Unmarshal(system, &blocks); err == nil {
		// Successfully parsed as array; extract text blocks.
		var texts []string
		for _, block := range blocks {
			if block.Type == "text" && block.Text != "" {
				texts = append(texts, block.Text)
			}
		}
		return strings.Join(texts, "\n\n"), nil
	}

	// Not an array; try bare string.
	var s string
	if err := json.Unmarshal(system, &s); err != nil {
		return "", fmt.Errorf("system prompt is neither string nor array: %w", err)
	}
	return s, nil
}

// reqAssistantMessage converts an assistant message, extracting tool_use blocks as ToolCalls.
func reqAssistantMessage(msg wire.Message, mode config.CacheMode) (wire.OAMessage, error) {
	out := wire.OAMessage{Role: "assistant"}

	// Parse content (string or array).
	content, err := reqParseContent(msg.Content)
	if err != nil {
		return wire.OAMessage{}, err
	}

	var textParts []wire.OAContentPart
	for _, block := range content {
		if block.Type == "text" {
			part := wire.OAContentPart{Type: "text", Text: block.Text}
			if mode == config.CacheExplicit && block.CacheControl != nil {
				part.CacheControl = block.CacheControl
			}
			textParts = append(textParts, part)
		} else if block.Type == "tool_use" {
			// Extract JSON arguments.
			var inputStr string
			if len(block.Input) > 0 {
				inputStr = string(block.Input)
			} else {
				inputStr = "{}"
			}

			out.ToolCalls = append(out.ToolCalls, wire.OAToolCall{
				Type: "function",
				ID:   block.ID,
				Function: wire.OAFunctionCall{
					Name:      block.Name,
					Arguments: inputStr,
				},
			})
		}
		// Thinking blocks are dropped (not replayable upstream).
	}

	// If only text and no tool calls, emit as bare string if no cache directive.
	// Otherwise emit as array of parts to anchor cache_control.
	if len(out.ToolCalls) == 0 && len(textParts) > 0 {
		if len(textParts) == 1 && textParts[0].Type == "text" {
			// Check if the single text part has cache control.
			if mode == config.CacheExplicit && textParts[0].CacheControl != nil {
				// Must use array form to preserve cache_control.
				parts, _ := json.Marshal(textParts)
				out.Content = json.RawMessage(parts)
			} else {
				// Plain text, emit as bare string.
				s, _ := json.Marshal(textParts[0].Text)
				out.Content = json.RawMessage(s)
			}
		} else {
			// Multiple text parts, emit as array.
			parts, _ := json.Marshal(textParts)
			out.Content = json.RawMessage(parts)
		}
	} else if len(textParts) > 0 {
		// Tool calls and text; emit parts array (forced if cache_control present).
		parts, _ := json.Marshal(textParts)
		out.Content = json.RawMessage(parts)
	}

	return out, nil
}

// reqUserMessages converts a user message, extracting tool_result blocks as separate OAMessages.
// Returns multiple OAMessages if tool_result blocks are present, to preserve the ordering
// constraint that tool result messages come before any remaining text.
func reqUserMessages(msg wire.Message, mode config.CacheMode) ([]wire.OAMessage, error) {
	// Parse content (string or array).
	content, err := reqParseContent(msg.Content)
	if err != nil {
		return nil, err
	}

	var result []wire.OAMessage
	var textParts []wire.OAContentPart

	for _, block := range content {
		if block.Type == "text" {
			part := wire.OAContentPart{Type: "text", Text: block.Text}
			if mode == config.CacheExplicit && block.CacheControl != nil {
				part.CacheControl = block.CacheControl
			}
			textParts = append(textParts, part)
		} else if block.Type == "image" {
			part, err := reqImagePart(block, mode)
			if err != nil {
				return nil, err
			}
			textParts = append(textParts, part)
		} else if block.Type == "tool_result" {
			// Emit accumulated text/images as a message first if any.
			if len(textParts) > 0 {
				result = append(result, reqBuildUserMessage(textParts, mode))
				textParts = nil
			}

			// Extract text from tool_result content.
			toolResultText, err := reqExtractToolResultText(block.Content)
			if err != nil {
				return nil, err
			}

			// Emit tool result message.
			result = append(result, wire.OAMessage{
				Role:       "tool",
				ToolCallID: block.ToolUseID,
				Content:    json.RawMessage([]byte(fmt.Sprintf("%q", toolResultText))),
			})
		}
		// Thinking blocks are dropped.
	}

	// Emit any remaining text/images.
	if len(textParts) > 0 {
		result = append(result, reqBuildUserMessage(textParts, mode))
	}

	return result, nil
}

// reqImagePart builds an OAContentPart for an image block.
func reqImagePart(block wire.ContentBlock, mode config.CacheMode) (wire.OAContentPart, error) {
	part := wire.OAContentPart{Type: "image_url"}

	// Parse source to extract image URL.
	var source struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	}
	if err := json.Unmarshal(block.Source, &source); err != nil {
		return wire.OAContentPart{}, fmt.Errorf("parse image source: %w", err)
	}

	if source.Type == "base64" {
		// Build data URL: data:media_type;base64,data
		dataURL := fmt.Sprintf("data:%s;base64,%s", source.MediaType, source.Data)
		imageURL, _ := json.Marshal(map[string]string{"url": dataURL})
		part.ImageURL = imageURL
	} else if source.Type == "url" {
		// Use URL directly.
		imageURL, _ := json.Marshal(map[string]string{"url": source.URL})
		part.ImageURL = imageURL
	} else {
		return wire.OAContentPart{}, fmt.Errorf("unknown image source type: %q", source.Type)
	}

	if mode == config.CacheExplicit && block.CacheControl != nil {
		part.CacheControl = block.CacheControl
	}

	return part, nil
}

// reqExtractToolResultText extracts text from tool_result content (string or array).
func reqExtractToolResultText(content json.RawMessage) (string, error) {
	if len(content) == 0 {
		return "", nil
	}

	// Try array of ContentBlock.
	var blocks []wire.ContentBlock
	if err := json.Unmarshal(content, &blocks); err == nil {
		var texts []string
		for _, block := range blocks {
			if block.Type == "text" && block.Text != "" {
				texts = append(texts, block.Text)
			}
		}
		return strings.Join(texts, "\n\n"), nil
	}

	// Try bare string.
	var s string
	if err := json.Unmarshal(content, &s); err != nil {
		return "", fmt.Errorf("tool_result content is neither string nor array: %w", err)
	}
	return s, nil
}

// reqBuildUserMessage builds an OAMessage from text and image parts.
func reqBuildUserMessage(parts []wire.OAContentPart, mode config.CacheMode) wire.OAMessage {
	out := wire.OAMessage{Role: "user"}

	// If single text part with no cache control, emit as bare string.
	if len(parts) == 1 && parts[0].Type == "text" && parts[0].CacheControl == nil {
		s, _ := json.Marshal(parts[0].Text)
		out.Content = json.RawMessage(s)
	} else {
		// Strip cache_control if not in explicit mode.
		if mode != config.CacheExplicit {
			for i := range parts {
				parts[i].CacheControl = nil
			}
		}

		// Emit as array of parts.
		b, _ := json.Marshal(parts)
		out.Content = json.RawMessage(b)
	}

	return out
}

// reqParseContent parses a Message.Content field as either a bare string or an array of ContentBlock.
func reqParseContent(content json.RawMessage) ([]wire.ContentBlock, error) {
	if len(content) == 0 {
		return nil, nil
	}

	// Try to parse as array of ContentBlock.
	var blocks []wire.ContentBlock
	if err := json.Unmarshal(content, &blocks); err == nil {
		return blocks, nil
	}

	// Try bare string.
	var s string
	if err := json.Unmarshal(content, &s); err != nil {
		return nil, fmt.Errorf("message content is neither string nor array: %w", err)
	}

	// Return as single text ContentBlock.
	return []wire.ContentBlock{{Type: "text", Text: s}}, nil
}

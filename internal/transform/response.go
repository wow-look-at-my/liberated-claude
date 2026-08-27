package transform

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// OpenAIToAnthropic translates a non-streaming OpenAI Chat Completions response
// to an Anthropic Messages API response.
func OpenAIToAnthropic(resp *wire.OAResponse, advertisedModel string) (*wire.MessagesResponse, error) {
	if resp.Error != nil {
		return nil, fmt.Errorf("upstream error: %s", resp.Error.Message)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("response has no choices")
	}

	choice := resp.Choices[0]
	if choice.Message == nil {
		return nil, fmt.Errorf("choice 0 has no message")
	}

	msg := choice.Message
	content := respContentBlocks(msg)

	// Build the response ID: use the upstream ID if present, else synthesize one.
	id := resp.ID
	if id == "" {
		id = "msg_" + strconv.FormatInt(int64(resp.Created), 36)
	}

	// Map stop reason.
	stopReason := respStopReason(choice.FinishReason)

	// Build usage with proper cache accounting.
	usage := respUsage(resp.Usage)

	return &wire.MessagesResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Model:      advertisedModel,
		Content:    content,
		StopReason: stopReason,
		Usage:      usage,
	}, nil
}

// respContentBlocks converts an OpenAI message to Anthropic content blocks.
// Reasoning content (if present) becomes a thinking block placed before text.
func respContentBlocks(msg *wire.OAMessage) []wire.ContentBlock {
	var blocks []wire.ContentBlock

	// Emit thinking block first if reasoning_content is present.
	if msg.ReasoningContent != "" {
		blocks = append(blocks, wire.ContentBlock{
			Type:     "thinking",
			Thinking: msg.ReasoningContent,
		})
	}

	// Parse Content (bare string or list of parts).
	if msg.Content != nil && len(msg.Content) > 0 {
		var content interface{}
		if err := json.Unmarshal(msg.Content, &content); err == nil {
			// Try to parse as a list of content parts.
			if parts, ok := content.([]interface{}); ok {
				for _, p := range parts {
					if part, ok := p.(map[string]interface{}); ok {
						if typ, ok := part["type"].(string); ok && typ == "text" {
							if text, ok := part["text"].(string); ok {
								blocks = append(blocks, wire.ContentBlock{
									Type: "text",
									Text: text,
								})
							}
						}
					}
				}
			}
		}

		// If we did not find parts, treat Content as a bare string.
		if len(blocks) == 0 || (len(blocks) == 1 && blocks[0].Type == "thinking") {
			var text string
			if err := json.Unmarshal(msg.Content, &text); err == nil && text != "" {
				blocks = append(blocks, wire.ContentBlock{
					Type: "text",
					Text: text,
				})
			}
		}
	}

	// Add tool_use blocks for each tool call.
	for _, tc := range msg.ToolCalls {
		input := json.RawMessage("{}")
		if tc.Function.Arguments != "" {
			// Validate that Arguments is valid JSON; return error if not.
			var tmp interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &tmp); err != nil {
				// Store what we have anyway and let the caller see the raw string.
				input = json.RawMessage(tc.Function.Arguments)
			} else {
				input = json.RawMessage(tc.Function.Arguments)
			}
		}

		blocks = append(blocks, wire.ContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}

	return blocks
}

// respStopReason maps an OpenAI finish_reason to an Anthropic stop_reason.
func respStopReason(reason *string) string {
	if reason == nil || *reason == "" || *reason == "stop" {
		return "end_turn"
	}
	switch *reason {
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}

// respUsage converts OpenAI usage to Anthropic usage, accounting for cached tokens.
// CRITICAL: Anthropic semantics are that input_tokens EXCLUDES cached tokens,
// while many OpenAI providers report prompt_tokens INCLUDING cached ones.
// We subtract cached tokens from input_tokens to match Anthropic semantics.
func respUsage(usage *wire.OAUsage) wire.Usage {
	if usage == nil {
		return wire.Usage{}
	}

	cached := usage.CachedTokens()
	written := usage.CacheWriteTokens()

	// Subtract cached from input (Anthropic input_tokens excludes them).
	inputTokens := usage.PromptTokens - cached
	if inputTokens < 0 {
		inputTokens = 0
	}

	result := wire.Usage{
		InputTokens:  inputTokens,
		OutputTokens: usage.CompletionTokens,
	}

	if cached > 0 {
		result.CacheReadInputTokens = &cached
	}
	if written > 0 {
		result.CacheCreationInputTokens = &written
	}

	return result
}

// StreamOpenAIToAnthropic converts an OpenAI SSE stream to an Anthropic SSE stream.
// It reads `data: {...}` SSE lines from src and emits Anthropic event sequences to dst.
func StreamOpenAIToAnthropic(dst io.Writer, src io.Reader, advertisedModel string) error {
	scanner := bufio.NewScanner(src)

	// Track accumulated state across chunks.
	var messageID string
	var totalInputTokens int
	var totalOutputTokens int
	var cacheReadTokens int
	var cacheWriteTokens int
	var stopReason string
	var toolCalls map[int]*streamToolCall // toolCalls[index] = accumulated tool call

	toolCalls = make(map[int]*streamToolCall)
	stopReason = "end_turn" // default

	// Track content blocks being accumulated.
	type blockState struct {
		typ   string
		index int
	}
	var currentBlock *blockState
	// Content block indices are sequential across the message.
	nextIndex := 0

	// Emit a message_start event to signal the start of streaming.
	if err := streamEmit(dst, "message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"type":    "message",
			"id":      messageID,
			"role":    "assistant",
			"model":   advertisedModel,
			"content": []interface{}{},
			"usage": map[string]interface{}{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	}); err != nil {
		return err
	}

	for scanner.Scan() {
		line := scanner.Text()

		// Handle blank lines.
		if strings.TrimSpace(line) == "" {
			continue
		}

		// Handle [DONE] marker.
		if strings.TrimSpace(line) == "data: [DONE]" {
			break
		}

		// Parse SSE line: data: <json>
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		jsonStr := strings.TrimPrefix(line, "data: ")
		var oaResp wire.OAResponse
		if err := json.Unmarshal([]byte(jsonStr), &oaResp); err != nil {
			return fmt.Errorf("parse SSE event: %w", err)
		}

		if oaResp.Error != nil {
			return fmt.Errorf("upstream error: %s", oaResp.Error.Message)
		}

		if len(oaResp.Choices) == 0 {
			continue
		}

		if messageID == "" && oaResp.ID != "" {
			messageID = oaResp.ID
		}

		choice := oaResp.Choices[0]
		if choice.Delta == nil {
			continue
		}

		// Capture finish_reason for use after loop ends.
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			stopReason = respStopReason(choice.FinishReason)
		}

		delta := choice.Delta

		// Process thinking content.
		if delta.ReasoningContent != "" {
			if currentBlock == nil || currentBlock.typ != "thinking" {
				if currentBlock != nil {
					if err := streamEmit(dst, "content_block_stop", map[string]interface{}{}); err != nil {
						return err
					}
				}
				if err := streamEmit(dst, "content_block_start", map[string]interface{}{
					"type": "content_block_start",
					"content_block": map[string]interface{}{
						"type":     "thinking",
						"thinking": "",
					},
					"index": nextIndex,
				}); err != nil {
					return err
				}
				currentBlock = &blockState{typ: "thinking", index: nextIndex}
				nextIndex++
			}

			if err := streamEmit(dst, "content_block_delta", map[string]interface{}{
				"type": "content_block_delta",
				"delta": map[string]interface{}{
					"type":     "thinking_delta",
					"thinking": delta.ReasoningContent,
				},
				"index": currentBlock.index,
			}); err != nil {
				return err
			}
		}

		// Process text content.
		text, err := streamDeltaText(delta.Content)
		if err != nil {
			return err
		}
		if text != "" {
			if currentBlock == nil || currentBlock.typ != "text" {
				if currentBlock != nil {
					if err := streamEmit(dst, "content_block_stop", map[string]interface{}{}); err != nil {
						return err
					}
				}
				if err := streamEmit(dst, "content_block_start", map[string]interface{}{
					"type": "content_block_start",
					"content_block": map[string]interface{}{
						"type": "text",
						"text": "",
					},
					"index": nextIndex,
				}); err != nil {
					return err
				}
				currentBlock = &blockState{typ: "text", index: nextIndex}
				nextIndex++
			}

			if err := streamEmit(dst, "content_block_delta", map[string]interface{}{
				"type": "content_block_delta",
				"delta": map[string]interface{}{
					"type": "text_delta",
					"text": text,
				},
				"index": currentBlock.index,
			}); err != nil {
				return err
			}
			totalOutputTokens++
		}

		// Process tool calls.
		for _, tc := range delta.ToolCalls {
			idx := 0
			if tc.Index != nil {
				idx = *tc.Index
			}

			toolCall, exists := toolCalls[idx]
			if !exists {
				toolCall = &streamToolCall{ID: tc.ID, Name: tc.Function.Name}
				toolCalls[idx] = toolCall
			}

			if tc.Function.Arguments != "" {
				toolCall.Arguments += tc.Function.Arguments
			}

			// Emit content_block_start on first chunk for this tool call.
			if !toolCall.started {
				if currentBlock != nil {
					if err := streamEmit(dst, "content_block_stop", map[string]interface{}{}); err != nil {
						return err
					}
				}

				if err := streamEmit(dst, "content_block_start", map[string]interface{}{
					"type": "content_block_start",
					"content_block": map[string]interface{}{
						"type":  "tool_use",
						"id":    tc.ID,
						"name":  tc.Function.Name,
						"input": json.RawMessage("{}"),
					},
					"index": nextIndex,
				}); err != nil {
					return err
				}
				currentBlock = &blockState{typ: "tool_use", index: nextIndex}
				nextIndex++
				toolCall.started = true
			}

			// Emit input_json_delta for the arguments fragment.
			if tc.Function.Arguments != "" {
				if err := streamEmit(dst, "content_block_delta", map[string]interface{}{
					"type": "content_block_delta",
					"delta": map[string]interface{}{
						"type":         "input_json_delta",
						"partial_json": tc.Function.Arguments,
					},
					"index": currentBlock.index,
				}); err != nil {
					return err
				}
			}
		}

		// Process usage (final chunk when include_usage was set).
		if oaResp.Usage != nil {
			cached := oaResp.Usage.CachedTokens()
			written := oaResp.Usage.CacheWriteTokens()

			// Save totals for the final message_delta.
			totalInputTokens = oaResp.Usage.PromptTokens - cached
			if totalInputTokens < 0 {
				totalInputTokens = 0
			}
			totalOutputTokens = oaResp.Usage.CompletionTokens
			cacheReadTokens = cached
			cacheWriteTokens = written
		}

		// Flush if supported.
		if f, ok := dst.(http.Flusher); ok {
			f.Flush()
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan SSE: %w", err)
	}

	// Close the last content block.
	if currentBlock != nil {
		if err := streamEmit(dst, "content_block_stop", map[string]interface{}{}); err != nil {
			return err
		}
	}

	// Emit message_delta with final usage.
	deltaUsage := map[string]interface{}{
		"input_tokens":  totalInputTokens,
		"output_tokens": totalOutputTokens,
	}
	if cacheReadTokens > 0 {
		deltaUsage["cache_read_input_tokens"] = cacheReadTokens
	}
	if cacheWriteTokens > 0 {
		deltaUsage["cache_creation_input_tokens"] = cacheWriteTokens
	}

	if err := streamEmit(dst, "message_delta", map[string]interface{}{
		"type":  "message_delta",
		"delta": map[string]interface{}{"stop_reason": stopReason},
		"usage": deltaUsage,
	}); err != nil {
		return err
	}

	// Emit message_stop to signal end of stream.
	if err := streamEmit(dst, "message_stop", map[string]interface{}{}); err != nil {
		return err
	}

	if f, ok := dst.(http.Flusher); ok {
		f.Flush()
	}

	return nil
}

// streamToolCall accumulates a tool call across chunks.
type streamToolCall struct {
	ID        string
	Name      string
	Arguments string
	started   bool
}

// streamEmit writes an Anthropic SSE event to dst: event: <name>\ndata: <json>\n\n
// streamDeltaText returns the text carried by a streaming delta's content.
//
// The field arrives either as a bare JSON string or as a list of typed parts.
// Both decode cleanly into an interface value, so the decoded Go type decides
// which form was sent; a failed decode would mean the payload is not JSON at
// all, which is an error rather than a hint about the shape.
func streamDeltaText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var decoded interface{}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return "", fmt.Errorf("decode delta content: %w", err)
	}
	switch v := decoded.(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	case []interface{}:
		return streamPartsText(v), nil
	default:
		return "", fmt.Errorf("delta content has unexpected type %T", decoded)
	}
}

// streamPartsText concatenates the text of every text part in a content list.
func streamPartsText(parts []interface{}) string {
	var b strings.Builder
	for _, p := range parts {
		part, ok := p.(map[string]interface{})
		if !ok {
			continue
		}
		if typ, _ := part["type"].(string); typ != "text" {
			continue
		}
		if s, ok := part["text"].(string); ok {
			b.WriteString(s)
		}
	}
	return b.String()
}

func streamEmit(dst io.Writer, eventName string, data interface{}) error {
	// Marshal data to JSON.
	payload, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("marshal event %s: %w", eventName, err)
	}

	// Write SSE line: event: <name>\ndata: <json>\n\n
	_, err = fmt.Fprintf(dst, "event: %s\ndata: %s\n\n", eventName, payload)
	return err
}

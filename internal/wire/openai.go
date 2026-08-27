package wire

import "encoding/json"

// OAContentPart is one part of a multimodal OpenAI message.
// CacheControl (non-standard) allows cache_control translation by some gateways.
type OAContentPart struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	ImageURL     json.RawMessage `json:"image_url,omitempty"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
}

// OAMessage is one OpenAI chat message (Content raw: API accepts string or parts).
type OAMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  []OAToolCall    `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	// ReasoningContent is chain-of-thought from providers like DeepSeek.
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// OAToolCall is a function call requested by the model.
type OAToolCall struct {
	Index    *int           `json:"index,omitempty"`
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type,omitempty"`
	Function OAFunctionCall `json:"function"`
}

// OAFunctionCall holds function name and JSON arguments (string for streaming).
type OAFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// OATool is a tool definition in OpenAI form.
type OATool struct {
	Type     string     `json:"type"`
	Function OAFunction `json:"function"`
}

// OAFunction describes a callable function.
type OAFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// OARequest is an OpenAI Chat Completions request.
type OARequest struct {
	Model       string          `json:"model"`
	Messages    []OAMessage     `json:"messages"`
	MaxTokens   *int            `json:"max_tokens,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	Tools       []OATool        `json:"tools,omitempty"`
	ToolChoice  json.RawMessage `json:"tool_choice,omitempty"`
	// StreamOptions asks for final usage chunk while streaming (prevents cache loss).
	StreamOptions *OAStreamOptions `json:"stream_options,omitempty"`
}

// OAStreamOptions controls extra data sent with a stream.
type OAStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// OAUsage is OpenAI token accounting.
type OAUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// PromptTokensDetails carries the cache hit count on providers that
	// implement automatic prefix caching.
	PromptTokensDetails *OAPromptTokensDetails `json:"prompt_tokens_details,omitempty"`
	// CacheCreationInputTokens and CacheReadInputTokens: gateways' Anthropic accounting.
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
}

// OAPromptTokensDetails breaks down prompt tokens.
type OAPromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
	// CacheWriteTokens is non-standard (DeepSeek's prompt_cache_miss_tokens).
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

// OAChoice is one completion alternative.
type OAChoice struct {
	Index        int        `json:"index"`
	Message      *OAMessage `json:"message,omitempty"`
	Delta        *OAMessage `json:"delta,omitempty"`
	FinishReason *string    `json:"finish_reason,omitempty"`
}

// OAResponse is a Chat Completions reply, streaming or not.
type OAResponse struct {
	ID      string     `json:"id"`
	Object  string     `json:"object"`
	Created int64      `json:"created"`
	Model   string     `json:"model"`
	Choices []OAChoice `json:"choices"`
	Usage   *OAUsage   `json:"usage,omitempty"`
	Error   *OAError   `json:"error,omitempty"`
}

// OAError is an OpenAI-shaped error body.
type OAError struct {
	Message string          `json:"message"`
	Type    string          `json:"type,omitempty"`
	Code    json.RawMessage `json:"code,omitempty"`
}

// CachedTokens returns the number of prompt tokens served from cache,
// tolerating the several places providers report it.
func (u *OAUsage) CachedTokens() int {
	if u == nil {
		return 0
	}
	if u.CacheReadInputTokens != nil {
		return *u.CacheReadInputTokens
	}
	if u.PromptTokensDetails != nil {
		return u.PromptTokensDetails.CachedTokens
	}
	return 0
}

// CacheWriteTokens returns the number of prompt tokens written to cache.
func (u *OAUsage) CacheWriteTokens() int {
	if u == nil {
		return 0
	}
	if u.CacheCreationInputTokens != nil {
		return *u.CacheCreationInputTokens
	}
	if u.PromptTokensDetails != nil {
		return u.PromptTokensDetails.CacheWriteTokens
	}
	return 0
}

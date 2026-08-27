// Package wire holds the request and response shapes for the two APIs this
// gateway bridges: the Anthropic Messages API that Claude Desktop speaks, and
// the OpenAI Chat Completions API that most other providers speak.
//
// Fields are pointers or omitempty wherever absence differs from a zero value,
// so a request is forwarded with the same shape it arrived in. That matters
// most for cache_control: an absent block and an empty one mean different
// things to a provider that bills for cache writes.
package wire

import "encoding/json"

// CacheControl marks a content block as a prompt-cache breakpoint.
type CacheControl struct {
	Type string `json:"type"`
	// TTL is Anthropic's optional cache lifetime ("5m", "1h").
	TTL string `json:"ttl,omitempty"`
}

// ContentBlock is one piece of message content. The Anthropic API allows text,
// images, tool calls, tool results, and thinking blocks to share a list, so the
// unrecognized fields are preserved verbatim via Extra.
type ContentBlock struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
	Source       json.RawMessage `json:"source,omitempty"`
	// Tool use.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// Tool result.
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   *bool           `json:"is_error,omitempty"`
	// Thinking.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// Message is one turn in the conversation.
type Message struct {
	Role string `json:"role"`
	// Content is either a bare string or a list of ContentBlock. Both forms are
	// legal on the wire, so it stays raw until a transform needs it.
	Content json.RawMessage `json:"content"`
}

// Tool is a tool definition offered to the model.
type Tool struct {
	Name         string          `json:"name"`
	Description  string          `json:"description,omitempty"`
	InputSchema  json.RawMessage `json:"input_schema,omitempty"`
	CacheControl *CacheControl   `json:"cache_control,omitempty"`
	Type         string          `json:"type,omitempty"`
}

// MessagesRequest is the body Claude Desktop POSTs to /v1/messages.
type MessagesRequest struct {
	Model     string          `json:"model"`
	Messages  []Message       `json:"messages"`
	System    json.RawMessage `json:"system,omitempty"`
	MaxTokens int             `json:"max_tokens"`
	Stream    bool            `json:"stream,omitempty"`

	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	TopK          *int            `json:"top_k,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Tools         []Tool          `json:"tools,omitempty"`
	ToolChoice    json.RawMessage `json:"tool_choice,omitempty"`
	Metadata      json.RawMessage `json:"metadata,omitempty"`
	Thinking      json.RawMessage `json:"thinking,omitempty"`

	// Extra keeps beta fields this gateway does not model (context_management
	// among them) so they reach an Anthropic upstream unchanged.
	Extra map[string]json.RawMessage `json:"-"`
}

// Usage reports token accounting. The cache fields are what make caching
// visible in Claude Desktop; dropping them makes a working cache look broken.
type Usage struct {
	InputTokens              int  `json:"input_tokens"`
	OutputTokens             int  `json:"output_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
}

// MessagesResponse is a non-streaming Messages API reply.
type MessagesResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   string         `json:"stop_reason,omitempty"`
	StopSequence *string        `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage"`
}

// ModelInfo is one entry in the /v1/models listing.
//
// SupportsOneM and MaxInputTokens are the two fields Claude Desktop consults to
// decide whether to offer a 1M-context variant:
//
//	l(e.supports_1m, e.max_input_tokens)
//	  where l = (a, b) => typeof a == "boolean" ? a : typeof b == "number" && b >= 1e6
//
// AnthropicFamilyTier is not optional in practice: a model carrying neither a
// recognized tier nor an Anthropic-looking ID is dropped from the listing.
type ModelInfo struct {
	Type                string `json:"type"`
	ID                  string `json:"id"`
	DisplayName         string `json:"display_name"`
	CreatedAt           string `json:"created_at,omitempty"`
	AnthropicFamilyTier string `json:"anthropic_family_tier,omitempty"`
	IsFamilyDefault     bool   `json:"is_family_default,omitempty"`
	SupportsOneM        bool   `json:"supports_1m,omitempty"`
	MaxInputTokens      int    `json:"max_input_tokens,omitempty"`
	MaxOutputTokens     int    `json:"max_output_tokens,omitempty"`
}

// ModelsResponse is the /v1/models listing envelope.
type ModelsResponse struct {
	Data    []ModelInfo `json:"data"`
	HasMore bool        `json:"has_more"`
	FirstID *string     `json:"first_id"`
	LastID  *string     `json:"last_id"`
}

// APIError is an Anthropic-shaped error body.
type APIError struct {
	Type  string       `json:"type"`
	Error APIErrorBody `json:"error"`
}

// APIErrorBody carries the error kind and message.
type APIErrorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// NewAPIError builds an error body in the shape Claude Desktop expects.
func NewAPIError(kind, msg string) APIError {
	return APIError{Type: "error", Error: APIErrorBody{Type: kind, Message: msg}}
}

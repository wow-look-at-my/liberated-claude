package transform

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// assertSystemOnlyFirst checks the constraint the upstream server enforces:
// a system message is legal at index 0 and nowhere else.
func assertSystemOnlyFirst(t *testing.T, out *wire.OARequest) {
	t.Helper()
	for i, msg := range out.Messages {
		if i == 0 {
			continue
		}
		assert.NotEqual(t, "system", msg.Role,
			"message %d is a system message, which upstream rejects with "+
				"\"System message must be at the beginning\"", i)
	}
}

func TestAnthropicToOpenAI_SystemRoleInMessages(t *testing.T) {
	m := &config.Model{ID: "glm-5.3-flash", ContextWindow: 1048576}

	t.Run("bare string content", func(t *testing.T) {
		req := &wire.MessagesRequest{
			MaxTokens: 128,
			Messages: []wire.Message{
				{Role: "system", Content: json.RawMessage([]byte(`"You are terse."`))},
				{Role: "user", Content: json.RawMessage([]byte(`"Say OK"`))},
			},
		}
		out, err := AnthropicToOpenAI(req, m)
		require.NoError(t, err, "a system turn in messages must be accepted")
		require.Len(t, out.Messages, 2, "both turns should survive")
		assert.Equal(t, "system", out.Messages[0].Role, "the system turn keeps its role")
		assert.JSONEq(t, `"You are terse."`, string(out.Messages[0].Content),
			"system text should carry through")
		assert.Equal(t, "user", out.Messages[1].Role, "ordering should be preserved")
	})

	t.Run("content block array", func(t *testing.T) {
		req := &wire.MessagesRequest{
			MaxTokens: 128,
			Messages: []wire.Message{
				{Role: "system", Content: json.RawMessage(
					[]byte(`[{"type":"text","text":"Block form."}]`))},
				{Role: "user", Content: json.RawMessage([]byte(`"Say OK"`))},
			},
		}
		out, err := AnthropicToOpenAI(req, m)
		require.NoError(t, err, "block-form system content must be accepted")
		require.Len(t, out.Messages, 2, "both turns should survive")
		assert.JSONEq(t, `"Block form."`, string(out.Messages[0].Content),
			"block text should be flattened into the system message")
	})

	t.Run("mid-conversation system turn rides as a user turn", func(t *testing.T) {
		req := &wire.MessagesRequest{
			MaxTokens: 128,
			System:    json.RawMessage([]byte(`"You are terse."`)),
			Messages: []wire.Message{
				{Role: "user", Content: json.RawMessage([]byte(`"Say OK"`))},
				{Role: "assistant", Content: json.RawMessage([]byte(`"OK"`))},
				{Role: "system", Content: json.RawMessage([]byte(`"Now answer in French."`))},
				{Role: "user", Content: json.RawMessage([]byte(`"Say OK"`))},
			},
		}
		out, err := AnthropicToOpenAI(req, m)
		require.NoError(t, err, "a late system turn must be accepted")
		require.Len(t, out.Messages, 5, "every turn should survive")
		assert.Equal(t, "user", out.Messages[3].Role,
			"upstream rejects a system message that is not first, so it demotes to user")
		assert.JSONEq(t, `"Now answer in French."`, string(out.Messages[3].Content),
			"the instruction text should carry through unchanged")
		assertSystemOnlyFirst(t, out)
	})

	t.Run("leading system turns merge with req.System", func(t *testing.T) {
		req := &wire.MessagesRequest{
			MaxTokens: 128,
			System:    json.RawMessage([]byte(`"You are terse."`)),
			Messages: []wire.Message{
				{Role: "system", Content: json.RawMessage([]byte(`"Answer in French."`))},
				{Role: "system", Content: json.RawMessage([]byte(`"Never apologize."`))},
				{Role: "user", Content: json.RawMessage([]byte(`"Say OK"`))},
			},
		}
		out, err := AnthropicToOpenAI(req, m)
		require.NoError(t, err, "stacked leading system turns must be accepted")
		require.Len(t, out.Messages, 2, "the three system sources collapse into one message")
		assert.JSONEq(t, `"You are terse.\n\nAnswer in French.\n\nNever apologize."`,
			string(out.Messages[0].Content), "every system source should be kept, in order")
		assert.Equal(t, "user", out.Messages[1].Role, "the user turn follows")
		assertSystemOnlyFirst(t, out)
	})

	t.Run("empty system turn is dropped", func(t *testing.T) {
		req := &wire.MessagesRequest{
			MaxTokens: 128,
			Messages: []wire.Message{
				{Role: "system", Content: json.RawMessage([]byte(`""`))},
				{Role: "user", Content: json.RawMessage([]byte(`"Say OK"`))},
			},
		}
		out, err := AnthropicToOpenAI(req, m)
		require.NoError(t, err, "an empty system turn should not be an error")
		require.Len(t, out.Messages, 1, "an empty system turn adds nothing upstream")
		assert.Equal(t, "user", out.Messages[0].Role, "only the user turn remains")
	})
}

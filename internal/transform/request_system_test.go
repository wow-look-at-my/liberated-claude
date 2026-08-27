package transform

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// Claude Desktop puts system turns in the messages array, not only in the
// top-level system field. Rejecting that role failed every real chat with
// `400 unknown message role: "system"`.
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

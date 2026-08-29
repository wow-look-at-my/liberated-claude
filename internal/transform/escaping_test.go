package transform

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// TestControlBytesSurviveAsValidJSON guards the 500 that Go's %q verb caused:
// it escapes a control byte as \x1b, and JSON defines no \x escape, so the
// request body was rejected by the encoder before it ever left the process.
func TestControlBytesSurviveAsValidJSON(t *testing.T) {
	// Terminal output reaches a tool result with escape sequences intact.
	nasty := "\x1b[31mred\x1b[0m\x00\x07 tab\there"

	toolResult, err := json.Marshal([]wire.ContentBlock{{
		Type:      "tool_result",
		ToolUseID: "tu_1",
		Content:   jsonString(nasty),
	}})
	require.NoError(t, err)

	req := &wire.MessagesRequest{
		System: jsonString(nasty),
		Messages: []wire.Message{
			{Role: "user", Content: toolResult},
		},
	}
	m := &config.Model{ID: "m", ContextWindow: 200000}

	out, err := AnthropicToOpenAI(req, m)
	require.NoError(t, err)

	// The whole request must marshal, which is where the failure surfaced.
	body, err := json.Marshal(out)
	require.NoError(t, err)
	require.NotContains(t, string(body), `\x`, "JSON has no \\x escape")

	// And it must decode back to exactly the bytes that went in.
	var round wire.OARequest
	require.NoError(t, json.Unmarshal(body, &round))
	var system, result string
	require.NoError(t, json.Unmarshal(round.Messages[0].Content, &system))
	require.NoError(t, json.Unmarshal(round.Messages[1].Content, &result))
	require.Equal(t, nasty, system)
	require.Equal(t, nasty, result)
}

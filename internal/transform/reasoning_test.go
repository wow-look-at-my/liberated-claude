package transform

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/wow-look-at-my/liberated-claude/internal/config"
	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// modelWithReasoningField builds a model whose provider spells the
// chain-of-thought field the given way.
func modelWithReasoningField(field string) *config.Model {
	raw := `<liberatedClaude>
		<server><listen>127.0.0.1:0</listen></server>
		<providers>
			<provider name="p" kind="openai" reasoningField="` + field + `">
				<baseURL>https://example.invalid/v1</baseURL>
				<models><model id="m" tier="sonnet" contextWindow="200000"/></models>
			</provider>
		</providers>
	</liberatedClaude>`
	cfg, err := config.Parse([]byte(raw))
	if err != nil {
		panic(err)
	}
	return cfg.Models()[0]
}

func TestAssistantThinkingIsReplayedUpstream(t *testing.T) {
	content, err := json.Marshal([]wire.ContentBlock{
		{Type: "thinking", Thinking: "17*23 = 17*20 + 17*3"},
		{Type: "text", Text: "391"},
	})
	require.NoError(t, err)
	req := &wire.MessagesRequest{
		Messages: []wire.Message{{Role: "assistant", Content: content}},
	}

	t.Run("reasoning_content", func(t *testing.T) {
		out, err := AnthropicToOpenAI(req, modelWithReasoningField("reasoning_content"))
		require.NoError(t, err)
		require.Len(t, out.Messages, 1)
		require.Equal(t, "17*23 = 17*20 + 17*3", out.Messages[0].ReasoningContent)
		require.Empty(t, out.Messages[0].Reasoning)

		// DeepSeek reads the wire key, so the emitted JSON has to carry it.
		enc, err := json.Marshal(out.Messages[0])
		require.NoError(t, err)
		require.Contains(t, string(enc), `"reasoning_content":`)
	})

	t.Run("reasoning", func(t *testing.T) {
		out, err := AnthropicToOpenAI(req, modelWithReasoningField("reasoning"))
		require.NoError(t, err)
		require.Len(t, out.Messages, 1)
		require.Equal(t, "17*23 = 17*20 + 17*3", out.Messages[0].Reasoning)
		require.Empty(t, out.Messages[0].ReasoningContent)
	})
}

func TestAssistantThinkingDefaultsToReasoningContent(t *testing.T) {
	content, err := json.Marshal([]wire.ContentBlock{
		{Type: "thinking", Thinking: "one"},
		{Type: "thinking", Thinking: "two"},
	})
	require.NoError(t, err)
	req := &wire.MessagesRequest{
		Messages: []wire.Message{{Role: "assistant", Content: content}},
	}
	out, err := AnthropicToOpenAI(req, modelWithReasoningField(""))
	require.NoError(t, err)
	require.Len(t, out.Messages, 1)
	require.Equal(t, "one\n\ntwo", out.Messages[0].ReasoningContent)
}

func TestResponseThinkingFromEitherFieldName(t *testing.T) {
	cases := map[string]wire.OAMessage{
		"reasoning_content": {Role: "assistant", ReasoningContent: "step by step"},
		"reasoning":         {Role: "assistant", Reasoning: "step by step"},
	}
	for name, msg := range cases {
		t.Run(name, func(t *testing.T) {
			msg.Content = json.RawMessage(`"391"`)
			resp := &wire.OAResponse{
				ID:      "chatcmpl-1",
				Choices: []wire.OAChoice{{Message: &msg}},
			}
			out, err := OpenAIToAnthropic(resp, "advertised")
			require.NoError(t, err)
			require.Len(t, out.Content, 2)
			require.Equal(t, "thinking", out.Content[0].Type)
			require.Equal(t, "step by step", out.Content[0].Thinking)
			require.Equal(t, "text", out.Content[1].Type)
			require.Equal(t, "391", out.Content[1].Text)
		})
	}
}

func TestStreamThinkingFromEitherFieldName(t *testing.T) {
	cases := map[string]string{
		"reasoning_content": `{"choices":[{"delta":{"reasoning_content":"pondering"}}]}`,
		"reasoning":         `{"choices":[{"delta":{"reasoning":"pondering"}}]}`,
	}
	for name, chunk := range cases {
		t.Run(name, func(t *testing.T) {
			src := strings.NewReader("data: " + chunk + "\n\ndata: [DONE]\n\n")
			var buf strings.Builder
			require.NoError(t, StreamOpenAIToAnthropic(&buf, src, "advertised"))
			require.Contains(t, buf.String(), `"type":"thinking"`)
			require.Contains(t, buf.String(), `"thinking":"pondering"`)
		})
	}
}

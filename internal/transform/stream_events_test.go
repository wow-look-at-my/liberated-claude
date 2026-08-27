package transform

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// streamEvents parses the SSE text into the decoded payload of each event.
func streamEvents(t *testing.T, sse string) []map[string]interface{} {
	t.Helper()
	var out []map[string]interface{}
	for _, line := range strings.Split(sse, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var payload map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload))
		out = append(out, payload)
	}
	return out
}

// TestStreamEventsAreSelfDescribing guards the contract that made Claude
// Desktop hang on "writing...": a client dispatches on the payload's type, so
// an event whose body is bare {} leaves the message unfinished forever.
func TestStreamEventsAreSelfDescribing(t *testing.T) {
	chunks := []string{
		`{"id":"chatcmpl-1","choices":[{"delta":{"reasoning":"pondering"}}]}`,
		`{"id":"chatcmpl-1","choices":[{"delta":{"content":"391"}}]}`,
		`{"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":4}}`,
	}
	var src strings.Builder
	for _, c := range chunks {
		src.WriteString("data: " + c + "\n\n")
	}
	src.WriteString("data: [DONE]\n\n")

	var buf strings.Builder
	require.NoError(t, StreamOpenAIToAnthropic(&buf, strings.NewReader(src.String()), "advertised"))
	events := streamEvents(t, buf.String())

	var types []string
	for _, e := range events {
		typ, ok := e["type"].(string)
		require.True(t, ok, "every event payload carries its own type: %v", e)
		types = append(types, typ)
	}
	require.Equal(t, []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_stop",
		"content_block_start", "content_block_delta", "content_block_stop",
		"message_delta", "message_stop",
	}, types)

	// A stop event has to name the block it closes, and the indices have to
	// pair with the starts.
	require.EqualValues(t, 0, events[1]["index"])
	require.EqualValues(t, 0, events[3]["index"])
	require.EqualValues(t, 1, events[4]["index"])
	require.EqualValues(t, 1, events[6]["index"])

	msg := events[0]["message"].(map[string]interface{})
	require.NotEmpty(t, msg["id"], "an empty message id fails client validation")
	require.Equal(t, "advertised", msg["model"])
}

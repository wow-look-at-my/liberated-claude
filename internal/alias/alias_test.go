package alias

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

// Real upstream IDs that Desktop's screen rejects. Each names a model somebody
// would plausibly want to route, which is why the encoding has to exist.
var rejectedUpstream = []string{
	"deepseek-v3",
	"deepseek/deepseek-chat",
	"z-ai/glm-4.6",
	"openai/gpt-5.6",
	"x-ai/grok-4",
	"google/gemini-2.5-pro",
	"moonshotai/kimi-k2",
	"qwen/qwen3-235b",
	"meta-llama/llama-4-maverick",
	"mistralai/mistral-large",
	"nvidia/nemotron-4",
	"amazon.nova-pro-v1:0",
	"microsoft/phi-4",
	"ibm/granite-3.1",
}

func TestAcceptsRejectsForeignModels(t *testing.T) {
	for _, id := range rejectedUpstream {
		assert.False(t, Accepts(id))

	}
}

func TestAcceptsAdmitsAnthropicIDs(t *testing.T) {
	accepted := []string{
		"claude-opus-4",
		"claude-sonnet-4-5",
		"opus",
		"sonnet-4.5",
		"haiku",
		"fable",
		"mythos",
		"anthropic.claude-3-5-sonnet",
	}
	for _, id := range accepted {
		assert.True(t, Accepts(id))

	}
}

// The whole point of Encode: whatever goes in, the result must survive the
// screen. A round trip must also return the original, or requests cannot be
// routed back to the upstream model.
func TestEncodeAlwaysAcceptedAndRoundTrips(t *testing.T) {
	for _, id := range rejectedUpstream {
		enc := Encode(id)
		assert.True(t, Accepts(enc))

		got := Decode(enc)
		assert.Equal(t, id, got)

	}
}

// An ID that already passes must not be mangled: real Anthropic models should
// stay readable in the picker.
func TestEncodePassesThroughAcceptedIDs(t *testing.T) {
	for _, id := range []string{"claude-opus-4", "claude-sonnet-4-5", "opus"} {
		assert.Equal(t, id, Encode(id), "Encode must not touch %q", id)
		assert.Equal(t, id, Decode(id), "Decode must not touch %q", id)
	}
}

// A foreign token must not reappear in the encoded form. Hex uses only 0-9a-f,
// so this holds structurally; the test pins it against a future encoding change.
func TestEncodedFormContainsNoForeignToken(t *testing.T) {
	for _, id := range rejectedUpstream {
		enc := Encode(id)
		assert.False(t, ForeignTokens.MatchString(enc))

	}
}

func TestDecodeLeavesMalformedInputAlone(t *testing.T) {
	// Odd length, non-hex body, and empty body are all unresolvable; each must
	// come back untouched so the caller reports an unknown model.
	for _, id := range []string{prefix + "abc", prefix + "zz", prefix} {
		got := Decode(id)
		assert.Equal(t, id, got)

	}
}

func TestSplitOneM(t *testing.T) {
	cases := []struct {
		in       string
		wantBase string
		wantOneM bool
	}{
		{"claude-opus-4[1m]", "claude-opus-4", true},
		{"claude-opus-4[1M]", "claude-opus-4", true},
		{"claude-opus-4 [1m]", "claude-opus-4", true},
		{"claude-opus-4", "claude-opus-4", false},
		{"claude[1m]-opus", "claude[1m]-opus", false},
	}
	for _, c := range cases {
		base, oneM := SplitOneM(c.in)
		assert.Equal(t, c.wantBase, base, "base of %q", c.in)
		assert.Equal(t, c.wantOneM, oneM, "oneM of %q", c.in)
	}
}

func TestIsTier(t *testing.T) {
	for _, s := range []string{"opus", "Sonnet", " haiku ", "fable", "mythos"} {
		assert.True(t, IsTier(s))

	}
	for _, s := range []string{"", "turbo", "opus-4"} {
		assert.False(t, IsTier(s))

	}
}

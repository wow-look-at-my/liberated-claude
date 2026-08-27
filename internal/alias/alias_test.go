package alias

import "testing"

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
		if Accepts(id) {
			t.Errorf("Accepts(%q) = true, want false: Desktop would drop this ID", id)
		}
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
		if !Accepts(id) {
			t.Errorf("Accepts(%q) = false, want true", id)
		}
	}
}

// The whole point of Encode: whatever goes in, the result must survive the
// screen. A round trip must also return the original, or requests cannot be
// routed back to the upstream model.
func TestEncodeAlwaysAcceptedAndRoundTrips(t *testing.T) {
	for _, id := range rejectedUpstream {
		enc := Encode(id)
		if !Accepts(enc) {
			t.Errorf("Encode(%q) = %q, which Desktop still rejects", id, enc)
		}
		if got := Decode(enc); got != id {
			t.Errorf("Decode(Encode(%q)) = %q, want %q", id, got, id)
		}
	}
}

// An ID that already passes must not be mangled: real Anthropic models should
// stay readable in the picker.
func TestEncodePassesThroughAcceptedIDs(t *testing.T) {
	for _, id := range []string{"claude-opus-4", "claude-sonnet-4-5", "opus"} {
		if got := Encode(id); got != id {
			t.Errorf("Encode(%q) = %q, want unchanged", id, got)
		}
		if got := Decode(id); got != id {
			t.Errorf("Decode(%q) = %q, want unchanged", id, got)
		}
	}
}

// A foreign token must not reappear in the encoded form. Hex uses only 0-9a-f,
// so this holds structurally; the test pins it against a future encoding change.
func TestEncodedFormContainsNoForeignToken(t *testing.T) {
	for _, id := range rejectedUpstream {
		if enc := Encode(id); ForeignTokens.MatchString(enc) {
			t.Errorf("Encode(%q) = %q contains a foreign token", id, enc)
		}
	}
}

func TestDecodeLeavesMalformedInputAlone(t *testing.T) {
	// Odd length, non-hex body, and empty body are all unresolvable; each must
	// come back untouched so the caller reports an unknown model.
	for _, id := range []string{prefix + "abc", prefix + "zz", prefix} {
		if got := Decode(id); got != id {
			t.Errorf("Decode(%q) = %q, want unchanged", id, got)
		}
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
		if base != c.wantBase || oneM != c.wantOneM {
			t.Errorf("SplitOneM(%q) = (%q, %v), want (%q, %v)",
				c.in, base, oneM, c.wantBase, c.wantOneM)
		}
	}
}

func TestIsTier(t *testing.T) {
	for _, s := range []string{"opus", "Sonnet", " haiku ", "fable", "mythos"} {
		if !IsTier(s) {
			t.Errorf("IsTier(%q) = false, want true", s)
		}
	}
	for _, s := range []string{"", "turbo", "opus-4"} {
		if IsTier(s) {
			t.Errorf("IsTier(%q) = true, want false", s)
		}
	}
}

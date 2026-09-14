// Package alias produces model IDs that Claude Desktop will accept.
//
// Desktop screens every model ID it discovers. Accepts below reproduces that
// screen from the shipped app so the IDs this gateway advertises are known to
// pass rather than guessed at: the app lowercases the ID, rejects it outright
// when a token belonging to a non-Anthropic model matches, and otherwise admits
// it when it is a bare tier alias or contains an Anthropic-flavored substring.
//
// The upshot is that a real upstream ID containing "deepseek" or "glm" can never
// be advertised as-is, since both are rejection tokens. Encode turns such an ID
// into an accepted form, and Decode is its inverse.
package alias

import (
	"regexp"
	"strings"
)

// ForeignTokens rejects an ID outright when any of them appears in it.
// Transcribed from the app's uge regex.
var ForeignTokens = regexp.MustCompile(
	`ark-code|astron|command-r|deepseek|doubao|gemini|gemma|glm|gpt|grok|hermes|` +
		`hy3|kimi|lfm|\bling\b|llama|longcat|mimo|minimax|mistral|mixtral|moonshot|` +
		`nemotron|openai|phi-|qianfan|qwen|tc-code|\bunic\b|yi-|stepfun|step-3|seed-|` +
		`bytedance|hunyuan|granite|amazon\.nova|nova-|devstral|ministral|ernie|codex|` +
		`arcee|trinity|abab|phi\d|\bk2\.|\bm2\.|jamba|arctic|solar|mercury|zamba|` +
		`kat-coder|\bds-|dpsk`)

// Tiers are the family tiers Desktop recognizes, in the app's order.
var Tiers = []string{"sonnet", "opus", "haiku", "fable", "mythos"}

// bareTier matches an ID that is nothing but a tier alias, optionally versioned.
var bareTier = regexp.MustCompile(`^(sonnet|opus|haiku|fable|mythos)(-[\d.]+)?$`)

// anthropicHints admit an ID when none of ForeignTokens matched.
var anthropicHints = append([]string{"claude"}, append(append([]string{}, Tiers...), "anthropic")...)

// prefix marks an encoded ID: "claude-lc-" (passes anthropicHints, no foreign tokens).
const prefix = "claude-lc-"

// Accepts reports whether Claude Desktop's model-ID screen admits id.
// It mirrors the app's lo() exactly, including the lowercase fold.
func Accepts(id string) bool {
	t := strings.ToLower(id)
	if ForeignTokens.MatchString(t) {
		return false
	}
	if bareTier.MatchString(t) {
		return true
	}
	for _, h := range anthropicHints {
		if strings.Contains(t, h) {
			return true
		}
	}
	return false
}

// IsTier reports whether s names a family tier Desktop recognizes.
func IsTier(s string) bool {
	t := strings.ToLower(strings.TrimSpace(s))
	for _, tier := range Tiers {
		if t == tier {
			return true
		}
	}
	return false
}

// Encode returns an ID for upstream that Desktop accepts. Pass-through if
// accepted; otherwise hex-encode (no foreign tokens survive the encoding).
func Encode(upstream string) string {
	if Accepts(upstream) {
		return upstream
	}
	var b strings.Builder
	b.Grow(len(prefix) + 2*len(upstream))
	b.WriteString(prefix)
	const hexDigits = "0123456789abcdef"
	for i := 0; i < len(upstream); i++ {
		c := upstream[i]
		b.WriteByte(hexDigits[c>>4])
		b.WriteByte(hexDigits[c&0x0f])
	}
	return b.String()
}

// Decode reverses Encode. IDs without prefix or malformed encodings return unchanged.
func Decode(id string) string {
	body, ok := strings.CutPrefix(id, prefix)
	if !ok {
		return id
	}
	if len(body) == 0 || len(body)%2 != 0 {
		return id
	}
	out := make([]byte, 0, len(body)/2)
	for i := 0; i < len(body); i += 2 {
		hi, ok1 := unhex(body[i])
		lo, ok2 := unhex(body[i+1])
		if !ok1 || !ok2 {
			return id
		}
		out = append(out, hi<<4|lo)
	}
	return string(out)
}

func unhex(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}

// oneMSuffix matches the "[1m]" marker for 1M-context model names (from app's Vs()).
var oneMSuffix = regexp.MustCompile(`(?i)^(.+?)\[1m\]$`)

// SplitOneM strips the trailing "[1m]" marker from model IDs.
func SplitOneM(id string) (base string, oneM bool) {
	m := oneMSuffix.FindStringSubmatch(id)
	if m == nil {
		return id, false
	}
	return strings.TrimSpace(m[1]), true
}

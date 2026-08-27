// Package alias produces model IDs that Claude Desktop will accept.
//
// Desktop screens every model ID it discovers. The screen is reproduced here
// verbatim from the shipped app so the IDs this gateway advertises are known to
// pass rather than guessed at:
//
//	function lo(e) {
//	  let t = e.toLowerCase();
//	  return uge.test(t) ? !1 : co.test(t) || lge.some((e) => t.includes(e));
//	}
//
// uge is a list of tokens belonging to non-Anthropic models; a match rejects the
// ID outright. co matches a bare tier alias. lge is a list of Anthropic-flavored
// substrings, any one of which admits the ID.
//
// The upshot is that a real upstream ID such as "deepseek-v3" or "z-ai/glm-4.6"
// can never be advertised as-is: "deepseek" and "glm" are both rejection tokens.
// Encode turns such an ID into one that passes, and Decode is its inverse.
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

// prefix marks an encoded ID. It contains "claude", which satisfies
// anthropicHints, and is itself free of any foreign token.
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

// Encode returns an ID for upstream that Desktop will accept.
//
// An upstream ID that already passes the screen is returned unchanged, so
// genuine Anthropic models keep their real names and stay recognizable in the
// picker. Anything else is hex-encoded behind prefix: hex has no letters beyond
// a-f, so no foreign token can survive the encoding and reappear in the result.
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

// Decode reverses Encode, returning the upstream model ID.
//
// An ID that does not carry prefix was passed through by Encode and is returned
// unchanged. A malformed encoding is likewise returned unchanged: the caller
// looks the result up in its model table, so an unresolvable ID surfaces there
// as an unknown model rather than as a silent mangling here.
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

// oneMSuffix matches the "[1m]" marker Desktop appends to name the
// 1M-context variant of a model. Transcribed from the app's Vs():
//
//	let t = e.match(/^(.+?)\[1m\]$/i);
var oneMSuffix = regexp.MustCompile(`(?i)^(.+?)\[1m\]$`)

// SplitOneM strips a trailing "[1m]" marker, reporting whether one was present.
// Requests for a model's 1M variant may arrive with the marker attached, so
// every incoming model ID is run through this before being resolved.
func SplitOneM(id string) (base string, oneM bool) {
	m := oneMSuffix.FindStringSubmatch(id)
	if m == nil {
		return id, false
	}
	return strings.TrimSpace(m[1]), true
}

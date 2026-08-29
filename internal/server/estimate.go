package server

import (
	"sync"

	tokenizer "github.com/wow-look-at-my/go-tokenizer"

	"github.com/wow-look-at-my/liberated-claude/internal/wire"
)

// perMessageOverhead is the framing each turn costs beyond its own text.
const perMessageOverhead = 4

// bpe is the shared cl100k_base tokenizer, built once and reused.
var bpe = sync.OnceValues(tokenizer.New)

// countInputTokens measures the prompt of a Messages request, for a provider
// that will not count it for us.
//
// Everything billed as input is included: the system prompt, every turn, and
// the tool schemas, which routinely outweigh the conversation. cl100k_base is
// not the vocabulary these models use, so the result is close, not exact.
func countInputTokens(req *wire.MessagesRequest) (int, error) {
	tok, err := bpe()
	if err != nil {
		return 0, err
	}
	total, err := tok.CountTokens(string(req.System))
	if err != nil {
		return 0, err
	}
	for _, msg := range req.Messages {
		n, err := tok.CountTokens(string(msg.Content))
		if err != nil {
			return 0, err
		}
		total += n + perMessageOverhead
	}
	for _, t := range req.Tools {
		n, err := tok.CountTokens(t.Name + t.Description + string(t.InputSchema))
		if err != nil {
			return 0, err
		}
		total += n
	}
	if len(req.ToolChoice) > 0 {
		n, err := tok.CountTokens(string(req.ToolChoice))
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

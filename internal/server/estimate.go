package server

import "github.com/wow-look-at-my/liberated-claude/internal/wire"

// bytesPerToken is the divisor used when no provider will count for us.
//
// Real tokenizers land between 2 and 4 bytes per token depending on how much
// of the text is prose; 2 is the dense-JSON end of that range, so the estimate
// errs high. That direction is deliberate: a client that thinks it has more
// room than it does overflows the context, while one that thinks it has less
// merely compacts sooner.
const bytesPerToken = 2

// perMessageOverhead is the framing each turn costs beyond its own text.
const perMessageOverhead = 4

// estimateInputTokens approximates the prompt size of a Messages request.
//
// The count covers everything the provider bills as input: the system prompt,
// every turn, and the tool schemas, which routinely outweigh the conversation.
func estimateInputTokens(req *wire.MessagesRequest) int {
	total := textTokens(len(req.System))
	for _, msg := range req.Messages {
		total += textTokens(len(msg.Content)) + perMessageOverhead
	}
	for _, t := range req.Tools {
		total += textTokens(len(t.Name) + len(t.Description) + len(t.InputSchema))
	}
	if len(req.ToolChoice) > 0 {
		total += textTokens(len(req.ToolChoice))
	}
	return total
}

// textTokens converts a byte length to an estimated token count, rounding up
// so that any content at all counts as at least one token.
func textTokens(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + bytesPerToken - 1) / bytesPerToken
}

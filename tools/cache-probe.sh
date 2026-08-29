#!/usr/bin/env bash
# Send the same large cache_control prefix twice and print each usage block, to
# see whether a provider reports cache reads on its Anthropic-native endpoint.
#
# usage: cache-probe.sh <base-url> <model>
set -euo pipefail

base="${1:?base url required}"
model="${2:?model required}"
: "${OLLAMA_API_KEY:?OLLAMA_API_KEY must be set}"

big=$(printf 'You are a precise assistant. %.0s' $(seq 1 400))
req=$(jq -nc --arg s "$big" --arg m "$model" '{
	model: $m,
	max_tokens: 32,
	system: [{type: "text", text: $s, cache_control: {type: "ephemeral"}}],
	messages: [{role: "user", content: "Say OK"}]
}')

for i in 1 2; do
	printf 'call %s: ' "$i"
	curl -s -X POST "${base}/v1/messages" \
		-H "Authorization: Bearer ${OLLAMA_API_KEY}" \
		-H 'anthropic-version: 2023-06-01' \
		-H 'content-type: application/json' \
		-d "$req" | jq -c '.usage'
	sleep 2
done

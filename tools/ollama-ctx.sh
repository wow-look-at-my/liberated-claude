#!/usr/bin/env bash
# Print "<model> <context_length> <capabilities>" for each Ollama Cloud model
# named on the command line. context_length lives under a family-prefixed key in
# model_info, so the key name differs per model and has to be matched.
set -euo pipefail

for model in "$@"; do
	body=$(printf '{"model":"%s"}' "$model")
	ctx=$(curl -s --max-time 25 -X POST https://ollama.com/api/show \
		-H 'Content-Type: application/json' -d "$body" |
		jq -r '[.model_info // {} | to_entries[] | select(.key|endswith("context_length")) | .value] | first // "unknown"')
	printf '%s\t%s\n' "$model" "$ctx"
done

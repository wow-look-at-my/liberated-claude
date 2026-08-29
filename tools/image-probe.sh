#!/bin/bash
# Live probe of the gateway's image path.
#
# Builds a base64 test image, sends two /v1/messages requests:
#   1. image in a plain user message
#   2. image inside a tool_result block (how Claude Code delivers screenshots)
# and prints what each upstream model answered, so image loss can be traced to
# the gateway transform or to the upstream provider.
#
# Usage: tools/image-probe.sh [model-alias ...]
# Default: both configured defaults (glm-5.3-flash, kimi-k3 aliases).

set -euo pipefail

cd "$(dirname "$0")/.."

ALIASES=("$@")
if [ ${#ALIASES[@]} -eq 0 ]; then
	ALIASES=("${GLM_ALIAS:-claude-lc-676c6d2d352e332d666c617368}" "claude-lc-6b696d692d6b33")
fi

IMG_B64=$(node tools/test-pattern.mjs | tail -1)

ask_user_image() {
	local alias=$1
	jq -n --arg alias "$alias" --arg img "$IMG_B64" '{
		model: $alias,
		max_tokens: 100,
		stream: false,
		messages: [{
			role: "user",
			content: [
				{ type: "image", source: { type: "base64", media_type: "image/png", data: $img } },
				{ type: "text", text: "What colors are in this image? Describe the layout in one sentence." }
			]
		}]
	}' | curl -s http://127.0.0.1:8787/v1/messages \
		-H "content-type: application/json" \
		--data-binary @- | jq -c '.content // .error // .'
	echo
}

ask_tool_result_image() {
	local alias=$1
	jq -n --arg alias "$alias" --arg img "$IMG_B64" '{
		model: $alias,
		max_tokens: 100,
		stream: false,
		messages: [
			{
				role: "user",
				content: [{ type: "text", text: "Use the screenshot tool, then tell me what colors are in the image you got." }]
			},
			{
				role: "assistant",
				content: [{ type: "tool_use", id: "probe-1", name: "screenshot", input: {} }]
			},
			{
				role: "user",
				content: [{
					type: "tool_result",
					tool_use_id: "probe-1",
					content: [
						{ type: "image", source: { type: "base64", media_type: "image/png", data: $img } },
						{ type: "text", text: "Screenshot captured." }
					]
				}]
			},
			{
				role: "user",
				content: [{ type: "text", text: "Now tell me: what colors were in the image returned by the tool?" }]
			}
		]
	}' | curl -s http://127.0.0.1:8787/v1/messages \
		-H "content-type: application/json" \
		--data-binary @- | jq -c '.content // .error // .'
	echo
}

for alias in "${ALIASES[@]}"; do
	echo "=== $alias: user-message image ==="
	ask_user_image "$alias"
	echo "=== $alias: tool_result image ==="
	ask_tool_result_image "$alias"
done
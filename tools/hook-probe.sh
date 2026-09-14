#!/opt/homebrew/bin/bash
# Feed a PreToolUse Bash payload to each installed cleanup-bash-cmds hook copy
# and print what it decides, so a silent no-op can be told apart from a rewrite.
set -uo pipefail

payload='{"tool_name":"Bash","tool_input":{"command":"sleep 60; echo waited"}}'

for h in \
	/Users/mhaynie/repos/cc-marketplace/plugins/cleanup-bash-cmds/hook.sh \
	/Users/mhaynie/.claude/plugins/npm-cache/node_modules/@buildhost/cc-marketplace__cleanup-bash-cmds/hook.sh \
	/Users/mhaynie/.claude/plugins/cache/wow-cc-marketplace/cleanup-bash-cmds/1194/hook.sh; do
	printf '=== %s\n' "$h"
	out=$(printf '%s' "$payload" | bash "$h")
	printf 'exit=%d out=%s\n' "$?" "${out:-<empty>}"
done

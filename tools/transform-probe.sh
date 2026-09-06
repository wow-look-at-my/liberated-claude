#!/opt/homebrew/bin/bash
# Runs cleanup-bash-cmds' transform.jq the way hook.sh does, on one command, and
# prints the decision plus the regenerated command. Isolates "the rule does not
# match" from "the hook never ran".
set -euo pipefail

dir=$1
cmd=$2

probe=$(printf '%s' $': 2>/dev/null\n: 2>>/dev/null\n: 2>&1\n: && :\n: || :\n: | :\n: |& :\n: <<CBC_A\nCBC_A\n: <<-CBC_B\nCBC_B' | shfmt --to-json)
ops=$(printf '%s' "$probe" | jq -c '{
	gt: .Stmts[0].Redirs[0].Op,
	app: .Stmts[1].Redirs[0].Op,
	dup: .Stmts[2].Redirs[0].Op,
	and: .Stmts[3].Cmd.Op,
	or: .Stmts[4].Cmd.Op,
	pipe: .Stmts[5].Cmd.Op,
	pipeall: .Stmts[6].Cmd.Op,
	hdoc: .Stmts[7].Redirs[0].Op,
	dashhdoc: .Stmts[8].Redirs[0].Op
}')
printf 'ops: %s\n' "$ops"

ast=$(printf '%s' "$cmd" | shfmt --to-json)
result=$(printf '%s' "$ast" | jq -c --argjson ops "$ops" -f "$dir/transform.jq")
printf 'deny=%s changed=%s rules=%s\n' \
	"$(printf '%s' "$result" | jq -r .deny)" \
	"$(printf '%s' "$result" | jq -r .changed)" \
	"$(printf '%s' "$result" | jq -r .rules)"
printf 'cleaned: '
printf '%s' "$result" | jq -c .ast | shfmt --from-json

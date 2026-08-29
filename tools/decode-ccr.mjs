#!/usr/bin/env node
// Claude Code Router hides the upstream model ID inside its own model name as
// hex, so a config exported from it lists names like
// "anthropic/claude-ccr-h<hex>". This recovers the original IDs.
import { readFileSync } from "node:fs";

const file = process.argv[2];
if (!file) {
	console.error("usage: decode-ccr.mjs <config.json>");
	process.exit(2);
}

const cfg = JSON.parse(readFileSync(file, "utf8"));
for (const m of cfg.inferenceModels ?? []) {
	const hex = /claude-ccr-h([0-9a-fA-F]+)$/.exec(m.name ?? "")?.[1];
	const decoded = hex ? Buffer.from(hex, "hex").toString("utf8") : "(not encoded)";
	const flags = [m.supports1m ? "1m" : "", m.prefer1m ? "prefer1m" : ""]
		.filter(Boolean)
		.join(",");
	console.log(`${decoded}\t${m.labelOverride ?? ""}\t${flags}`);
}

#!/usr/bin/env node
// Extract narrow windows around interesting strings in a large binary/asar,
// so the surrounding code can be read without dumping the whole file.
import { readFileSync } from "node:fs";

const [, , file, ...terms] = process.argv;
if (!file || terms.length === 0) {
	console.error("usage: asar-probe.mjs <file> <term> [term...]");
	process.exit(2);
}

const WINDOW = Number(process.env.WINDOW || 600);
const MAX_HITS = Number(process.env.MAX_HITS || 6);
const buf = readFileSync(file);
const text = buf.toString("latin1");

for (const term of terms) {
	console.log(`\n########## ${term} ##########`);
	let idx = -1;
	let hits = 0;
	const seen = new Set();
	while ((idx = text.indexOf(term, idx + 1)) !== -1 && hits < MAX_HITS) {
		const start = Math.max(0, idx - WINDOW);
		const end = Math.min(text.length, idx + term.length + WINDOW);
		const slice = text.slice(start, end).replace(/[^\x20-\x7E]/g, ".");
		// Collapse near-duplicate hits from repeated bundles.
		const key = slice.slice(WINDOW - 120, WINDOW + 200);
		if (seen.has(key)) continue;
		seen.add(key);
		hits++;
		console.log(`\n--- hit ${hits} @ ${idx} ---`);
		console.log(slice);
	}
	if (hits === 0) console.log("(no hits)");
}

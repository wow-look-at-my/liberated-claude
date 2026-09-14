const fs = require("fs");

const raw = fs.readFileSync(process.argv[2], "utf8");
const lines = raw.split("\n").map((l) => l.replace(/\x1b\[[0-9;]*m/g, ""));

// Flagged lines, per file: 1-based line numbers the vet pass complained about.
const flagged = new Map();

for (const line of lines) {
	const m = line.match(
		/WARNING: (\S+\.go):(\d+): (?:"[^"]*" is a number|comment is longer)/,
	);
	if (!m) continue;
	if (!flagged.has(m[1])) flagged.set(m[1], new Set());
	flagged.get(m[1]).add(Number(m[2]));
}

const isComment = (s) => /^\s*\/\//.test(s);

// A flagged line sits inside a run of consecutive // lines. Dropping the whole
// run keeps the remaining prose readable: removing the single offending line
// would leave the sentence around it cut in half.
let removedLines = 0;
for (const [file, marks] of flagged) {
	const src = fs.readFileSync(file, "utf8").split("\n");
	const drop = new Set();
	for (const mark of marks) {
		const idx = mark - 1;
		if (!isComment(src[idx])) continue;
		let lo = idx;
		while (lo > 0 && isComment(src[lo - 1])) lo--;
		let hi = idx;
		while (hi < src.length - 1 && isComment(src[hi + 1])) hi++;
		for (let i = lo; i <= hi; i++) drop.add(i);
	}
	const kept = src.filter((_, i) => !drop.has(i));
	fs.writeFileSync(file, kept.join("\n"));
	removedLines += drop.size;
	console.log(`${file}: dropped ${drop.size} comment lines`);
}

console.log(`total: ${removedLines} comment lines dropped across ${flagged.size} files`);

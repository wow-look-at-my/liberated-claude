// pm2 process definition.
//
// interpreter "none" is required: the script is a compiled binary, and without
// it pm2 hands the file to node.
//
// watch targets the built binary, so `go-toolchain` producing a new build/
// artifact restarts the gateway on its own. watch_delay lets the linker finish
// writing before the restart fires, which otherwise catches a partial file.
//
// Secrets are read from the environment pm2 inherits at start; none are written
// here, and the config.xml this points at is gitignored.
module.exports = {
	apps: [
		{
			name: "liberated-claude",
			script: "./build/liberated-claude",
			args: "-config config.xml",
			cwd: __dirname,
			interpreter: "none",
			watch: ["build/liberated-claude"],
			watch_delay: 2000,
			ignore_watch: ["node_modules", ".git", "logs", "config.xml"],
			autorestart: true,
			max_restarts: 10,
			restart_delay: 1000,
			out_file: "logs/liberated-claude.out.log",
			error_file: "logs/liberated-claude.err.log",
			merge_logs: true,
			time: true,
		},
	],
};

import { readFile, realpath } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type ExtensionAPI, withFileMutationQueue } from "@earendil-works/pi-coding-agent";

const project = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

export default function (pi: ExtensionAPI) {
  pi.on("tool_result", async (event, ctx) => {
    if (event.isError || !["edit", "write"].includes(event.toolName)) return;
    const input = event.input.path;
    if (typeof input !== "string" || !/\.pyi?$/.test(input)) return;

    const note = (text: string) => ({
      content: [...event.content, { type: "text" as const, text }],
    });
    try {
      const root = await realpath(project);
      const file = await realpath(resolve(ctx.cwd, input));
      const local = relative(root, file);
      if (isAbsolute(local) || local === ".." || local.startsWith("../")) return;
      if (local.split(/[\\/]/).some((part) => part.startsWith("."))) return;

      return await withFileMutationQueue(file, async () => {
        const before = await readFile(file, "utf8");
        const diagnostics: string[] = [];
        for (const args of [["check", "--fix", "--no-unsafe-fixes"], ["format"]]) {
          const result = await pi.exec(join(root, ".venv/bin/ruff"),
            [...args, "--force-exclude", "--", file], { cwd: root, timeout: 10000 });
          if (result.code !== 0) {
            diagnostics.push(`${args[0]}: ${result.stdout}${result.stderr}`.slice(0, 4000));
          }
        }
        const changed = before !== await readFile(file, "utf8");
        if (diagnostics.length) {
          return note(`File saved; Ruff needs attention:\n${diagnostics.join("\n")}\nRe-read before editing.`);
        }
        if (changed) return note("Ruff applied safe fixes and formatting. Re-read before editing.");
      });
    } catch (error) {
      return note(`File saved; Ruff hook failed (run uv sync --locked if needed): ${error}`);
    }
  });
}

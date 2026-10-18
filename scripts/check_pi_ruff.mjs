// Run with Node and a global Pi install; no model or credentials are used.
import assert from "node:assert/strict";
import { execFileSync, execFile } from "node:child_process";
import { mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { promisify } from "node:util";

const root = process.cwd();
const globalModules = execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim();
const piPackage = join(globalModules, "@earendil-works/pi-coding-agent");
const require = createRequire(join(piPackage, "package.json"));
const { createJiti } = require("jiti");
const jiti = createJiti(import.meta.url, {
  alias: { "@earendil-works/pi-coding-agent": join(piPackage, "dist/index.js") },
});
const extension = await jiti.import(resolve(".pi/extensions/ruff.ts"), { default: true });
const exec = promisify(execFile);

function setup(run = async (command, args, options) => {
  try {
    const { stdout, stderr } = await exec(command, args, options);
    return { code: 0, stdout, stderr };
  } catch (error) {
    return { code: error.code, stdout: error.stdout ?? "", stderr: error.stderr ?? error.message };
  }
}) {
  let handler;
  extension({ on: (event, callback) => {
    assert.equal(event, "tool_result");
    handler = callback;
  }, exec: run });
  return (path, overrides = {}) => handler({
    toolName: "write", input: { path }, content: [], isError: false, ...overrides,
  }, { cwd: root });
}

await test("Pi fixes and formats edits and writes; second pass is unchanged", async () => {
  const dir = await mkdtemp(join(root, "tests/ruff-check-"));
  const file = join(dir, "space name.py");
  try {
    const handler = setup();
    for (const toolName of ["edit", "write"]) {
      await writeFile(file, "import sys\nimport os\nprint( sys.version,os.name )\n");
      const result = await handler(file, { toolName });
      assert.match(result.content[0].text, /Re-read/);
      assert.equal(await readFile(file, "utf8"), "import os\nimport sys\n\nprint(sys.version, os.name)\n");
      assert.equal(await handler(file), undefined);
    }
    await writeFile(file, "print(undefined_name)\n");
    assert.match((await handler(file)).content[0].text, /F821/);
    await writeFile(file, "def broken(\n");
    assert.match((await handler(file)).content[0].text, /needs attention/);
    assert.equal(await readFile(file, "utf8"), "def broken(\n");
    const missing = setup(async () => { throw new Error("missing Ruff"); });
    assert.match((await missing(file)).content[0].text, /hook failed/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

await test("Pi skips failed tools, other tools, non-Python and outside paths", async () => {
  const outside = await mkdtemp(join(tmpdir(), "ruff-outside-"));
  const link = join(root, "tests/ruff-outside-link.py");
  const handler = setup(async () => { assert.fail("Ruff must not run"); });
  try {
    const file = join(outside, "file.py");
    await writeFile(file, "x=1\n");
    await symlink(file, link);
    assert.equal(await handler(file), undefined);
    assert.equal(await handler(link), undefined);
    assert.equal(await handler("README.md"), undefined);
    assert.equal(await handler("missing.py", { isError: true }), undefined);
    assert.equal(await handler("missing.py", { toolName: "bash" }), undefined);
    assert.equal(await readFile(file, "utf8"), "x=1\n");
  } finally {
    await rm(link, { force: true });
    await rm(outside, { recursive: true, force: true });
  }
});

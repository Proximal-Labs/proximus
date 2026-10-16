# Proximus

Proximus is a host-side coding agent for long-horizon tasks, packaged as an
external agent for [Harbor](https://pypi.org/project/harbor/). It provides a
LiteLLM tool loop, time budgeting, context compaction, Git-backed submissions,
winner restore, and trajectory output.

Status: alpha (`0.1.0a1`), targeting Harbor `0.23.0`.

## Local development

Requires Python 3.12+, uv, and Git/Bash for the real submission-script tests.

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests scripts
uv build
```

Check the built wheel in a fresh environment outside this checkout:

```sh
uv run python scripts/verify_wheel.py dist/harbor_proximus-0.1.0a1-py3-none-any.whl
```

Dependency installation uses the network/cache. Tests block network access and use
fake providers; shell submission tests operate on temporary local Git repos.
`uv.lock` records the development environment. The wheel verifier also accepts
`--litellm-version 1.97.0` to exercise the declared LiteLLM lower bound.

## Use with Harbor

Install the built wheel in the same Python environment as upstream Harbor:

```sh
uv pip install --python /path/to/venv/bin/python dist/harbor_proximus-0.1.0a1-py3-none-any.whl
```

Select `harbor_proximus:Proximus` through `AgentConfig.import_path` or through
`AgentConfig.name`. Both routes are supported by upstream Harbor 0.23.0 and covered
by the offline factory/run tests. See [examples/agent.yaml](examples/agent.yaml)
for an agent configuration fragment to merge into a complete Harbor job.

Credentials follow Harbor's own agents. Proximus calls the model from the
orchestrator host. Set the provider's usual variable (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, ...) in your shell, or in Harbor's agent env (`--ae
'KEY=${VAR}'`, or `env:` with `${VAR}` templates in the job config). The agent
env wins; shell variables are read by LiteLLM. Provider base URLs
(`OPENAI_BASE_URL`, ...) work the same way, also with `api_key_env`, which names
a different variable for one model (e.g. two OpenAI-compatible endpoints).
Harbor overlays the agent env onto sandbox commands; Proximus blanks the
credential and base-URL variables it uses there, so the model cannot read
them. Other agent-env variables stay visible in the sandbox. Bedrock,
SageMaker and Vertex credentials are left to LiteLLM's own lookup from the
shell. Never put keys in agent kwargs (`--ak`, `llm_call_kwargs`): Harbor
saves kwargs verbatim in the job files. Options are Harbor
agent kwargs (`--ak key=value`); `harbor agent schema harbor_proximus:Proximus`
lists them.

There are no baked model/endpoint/reasoning presets. LiteLLM supplies public
model metadata; pass `reasoning_effort`, `llm_call_kwargs`, or
`litellm_model_info` explicitly when needed. Unknown metadata defaults to a 200K
context limit, no vision, and potentially zero cost telemetry. Configure custom
models rather than relying on these fallbacks for a benchmark run.

The model calls bash, submit, list_submissions, and view_image when supported.
The first submit records a Git commit and arms completion. A later-turn submit
confirms completion; intervening work can disarm confirmation. The latest successful
submission wins. At exit, auto-submit and restore are enabled by default, so the
workspace holds the winning commit when Harbor captures it for the verifier.
Provider failures remain failures even when work is saved.

Proximus is infrastructure-agnostic: it runs on the orchestrator host and acts in
the sandbox only through Harbor's environment interface, so it works with any
Harbor environment (Docker, Modal, ...). Inside the sandbox it needs `git` and
`bash`; submissions are commits in the workspace repository. Moving or storing
results is left to Harbor and the runner.

The work budget must leave headroom below Harbor's hard timeout for submission and
restore. For tasks that already describe a timer, set `task_provides_timer: true`:
Proximus re-anchors that timer without introducing a second model-facing clock.

## Options and customization

Options mirror Harbor's Terminus 2 where it has one: `max_turns`,
`enable_summarize`, `temperature`, `api_base`, `max_thinking_tokens`,
`model_info`, `llm_call_kwargs`, and `prompt_template_path` (Harbor's Jinja
template around `{{ instruction }}`). Proximus adds `compaction_threshold`
(fraction of the context limit, default 0.85) and `disallowed_tools`
(`list_submissions`, `view_image`; `bash` and `submit` are required). Unset,
every option keeps the default behavior. `llm_call_kwargs` wins over the other
request settings.

For deeper changes, subclass the agent and select your class with
`--agent mypkg.module:MyProximus`, as with other Harbor agents:

```python
from harbor_proximus import Proximus, tools

class MyProximus(Proximus):
    def system_prompt(self) -> str:
        return super().system_prompt() + "\n\nPrefer small commits."

    def build_tools(self) -> list[tools.Tool]:
        return [*super().build_tools(), tools.Tool("count_lines", "...", Args, self._count)]

    async def _count(self, args) -> tools.ToolResult:
        result = await self.environment.exec(f"wc -l {args.path}", timeout_sec=30)
        return tools.ToolResult(str(result.stdout))
```

The hooks are `system_prompt()`, `initial_message(instruction, workdir,
environment)`, `summary_request()` and `build_tools()`. `system_prompt()` and
`build_tools()` run inside `Proximus.__init__`, so set any attributes they need
before calling `super().__init__()`; `self.environment` exists only during a
run, so use it in tool handlers, not while building tools. After a compaction
the conversation restarts from a handoff message that repeats the (templated)
instruction, not the `initial_message()` text. Other model-facing text (submit
replies, nudges) drives the submit-confirmation protocol and is not a hook.

`max_turns` counts loop iterations, including retries after an output-length
error, an image rejection or a compaction. `max_thinking_tokens` and
`reasoning_effort` are mutually exclusive; Anthropic rejects `temperature`
other than 1 while thinking is on.

## Task configuration

A task can declare Proximus options in its `task.toml`. Harbor ignores these
tables and gives a custom agent no access to the task, so a runner reads them
with `harbor_proximus.task_agent_kwargs(task_dir, model_name, overrides,
defaults)`, passes the result as `AgentConfig.kwargs`, and sets Harbor's hard
agent timeout (`override_timeout_sec`) to `harbor_proximus.hard_timeout_sec(kwargs)`:

```toml
[agent]
timeout_sec = 72000.0

[agent.proximus]
task_provides_timer = true

[agent.proximus.models."anthropic/claude-opus-5"]
reasoning_effort = "max"
api_key_env = "ANTHROPIC_API_KEY"
```

Later layers win: the caller's defaults, the `[agent.proximus]` table, the
table for the run's model, then the caller's overrides. Nested tables such as
`llm_call_kwargs` merge per key. `time_budget_sec` defaults to
`[agent].timeout_sec`, a relative `prompt_template_path` in the task's tables is
resolved against the task directory, and `workspace_git_excludes` defaults to
the excludes of the task's `/app` artifact, so submissions never commit files
grading drops. `hard_timeout_sec()` adds room for the exit submit and restore
above the work budget. Unknown or invalid options raise `ValueError`.

## Compatibility and limits

- Requires Harbor `0.23.0` exactly and Python 3.12+. The test suite passes on
  Python 3.12, 3.13 and 3.14 with LiteLLM 1.97.0 (the declared lower bound) and
  1.105.1; other LiteLLM versions are not tested.
- At LLM construction Proximus installs a small process-wide LiteLLM
  compatibility shim for reasoning-only Responses API turns; no Harbor code is
  patched. Recheck it when upgrading LiteLLM.
- MCP servers, Harbor skills, native-session resume and handoff are not
  supported; a task or job that configures MCP servers or skills fails at setup.
- The sandbox must provide `git` and `bash`. A task image without git fails at
  setup with a clear error.
- `[agent.proximus]` in `task.toml` is read by runners that call
  `task_agent_kwargs`; a plain `harbor run` ignores it.
- Results depend on model settings. Reproducing a published score requires the
  same model configuration (reasoning effort, routing, budget) as the original run.

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

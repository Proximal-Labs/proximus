# Proximus

Proximus is a coding agent for long-horizon tasks, packaged as an external agent
for [Harbor](https://pypi.org/project/harbor/). It runs a tool loop on the
orchestrator host, calling models through the vendors' official SDKs
(`anthropic`, `openai`), and acts in the sandbox only through Harbor's environment
interface, so it works with any Harbor environment (Docker, Modal, ...).

Status: alpha (`0.1.0a1`), for Harbor `0.24.0`.

## Install and run

```sh
# Harbor declares litellm but Proximus never uses it; keep it out of the environment.
echo "litellm; sys_platform == 'never'" > overrides.txt
uv pip install --overrides overrides.txt harbor==0.24.0 \
  "harbor-proximus @ git+https://github.com/Proximal-Labs/proximus"
harbor run ... --agent harbor_proximus:Proximus --model anthropic/claude-opus-5 \
  --ak time_budget_sec=7200
```

[examples/agent.yaml](examples/agent.yaml) shows a job-config fragment.
`harbor agent schema harbor_proximus:Proximus` lists every option.

## How it works

- **Tools:** `bash`, `submit`, and `list_submissions`. `view_image` requires an
  explicit opt-in and a vision-capable model.
- **Submissions** are Git commits in the workspace. The first `submit` records a
  commit; a `submit` on a later turn confirms it and ends the run, and any other
  tool call in between cancels the confirmation. The latest submission wins.
- **At exit** Proximus auto-submits unsaved work and checks out the winning
  commit, so Harbor captures the submitted tree for the verifier.
- **Time:** `time_budget_sec` paces the run. Set Harbor's hard agent timeout above
  it (see `hard_timeout_sec` below) so the exit submit and restore can finish. With
  `task_provides_timer = true`, Proximus re-anchors the task's own timer instead of
  adding a second clock.
- **Context:** near the context limit the conversation is compacted into a handoff
  summary; the model's notes in `/tmp/proximus/PROGRESS.md` survive it.
- **Sandbox needs:** `bash`, and `git` in the default submission mode.

## Git-free completion

Set `submission_mode: "none"` for tasks whose instructions define a file deliverable:

```yaml
kwargs:
  submission_mode: "none"
  time_budget_sec: 1200
```

This mode exposes `done` instead of `submit` and `list_submissions`.
One `done()` call ends the run. It does not inspect, validate, or change the deliverable.
The prompt asks the agent to check its remaining time and budget first.
If resources and further progress remain, the agent must continue working.
This is prompt guidance, not a minimum runtime or a second confirmation call.
Proximus does not initialize Git, record a submission, or restore files, including on timeout or error.
The workspace stays as the agent left it; Harbor collects the task's configured artifacts.
Git-specific options have no effect in this mode.
`hard_timeout_sec` adds only cleanup time, not Git submission or restore time.
Trace metadata records `submission_mode` and `end_reason`; `submitted_commit` is null.

The default `submission_mode: "git"` keeps the existing behavior.
A subclass can still add tools through `build_tools()`.
Any tool that returns `ToolResult(..., ends_run=True)` ends the loop.
Plain text such as "done" does not end it.

## Bash execution and output

Tool calls run sequentially, in the order returned by the model.
This also applies to multiple calls in one response; they do not run in parallel.
Each bash call uses a fresh shell in the task's working directory.
For parallel work, launch background processes with separate logs and saved PIDs:

```sh
nohup ./run.sh > /tmp/run.log 2>&1 & echo $! > /tmp/run.pid
```

Check the logs, processes, and results with later calls.
`timeout_sec` defaults to 600 seconds and accepts at most 1,800 seconds.
The remaining work budget can reduce that wait.
Run work that may exceed its selected timeout in the background.
Timeout cleanup depends on the environment; do not assume that it killed the process.

Large output previews keep 30,000 characters from the beginning and end, plus a truncation marker.
The complete captured stdout and stderr are also saved separately in the sandbox:
`/tmp/proximus-output-<unique-id>.stdout` and `.stderr`.
The tool result gives both paths. Use later bash calls to search or read ranges.
The files preserve the original captured text, including trailing whitespace and empty streams.
Small outputs need no file transfer. Failed saves produce an explicit tool error and keep the preview.
These files last for the sandbox's lifetime unless the agent removes them.
This cannot recover output that the environment never returns, including failed text decoding.
Redirect long-running commands to files from the start to retain output across such failures.

## Image opt-in

`enable_view_image` defaults to `false`, even for vision-capable models.
Set it to `true` in agent kwargs or `[agent.proximus]` to expose `view_image`:

```yaml
kwargs:
  enable_view_image: true
```

Construction fails if the enabled tool's model lacks known vision support.
An unknown capability or a failed capability lookup is an error, not a silent tool removal.
For an unregistered vision model, provide its actual capability through `model_info.supports_vision`.
Do not claim vision support for a text-only model.
Enabling the tool while excluding it through `disallowed_tools` is also an error.
Existing image-dependent configurations must add this explicit opt-in.
Text-only runs do not require vision support.

## Models and credentials

Model names need a provider prefix, which picks the API:

| Prefix | API (SDK) |
| --- | --- |
| `anthropic/` | Anthropic Messages (`anthropic`), streamed, with 1h prompt-cache breakpoints |
| `openai/` | OpenAI Responses (`openai`), stateless with encrypted reasoning |
| any other (`deepseek/`, `openrouter/`, ...) | Plain OpenAI Chat Completions at that provider's endpoint |

Only the models in the table below are tested; any other model needs its own
settings checked before an eval. `api_base` points a route at another endpoint
(a gateway, for example); routes other than `anthropic/` and `openai/` need an
endpoint and their own key. Each model's own assistant turn (thinking
signatures, encrypted reasoning) is kept in the history and replayed verbatim.

There are no built-in model presets: pass `reasoning_effort`, `llm_call_kwargs`
or `model_info` as needed. `llm_call_kwargs` use the API's own parameter names;
names the SDK does not take go into the request body. Context limits, vision support and prices come from a built-in table,
`src/harbor_proximus/model_info.json`, which lists only the models Proximus is
evaluated with (values from the vendors' model and pricing pages):
`anthropic/claude-opus-5-5`, `anthropic/claude-haiku-5-5`, `openai/gpt-5.6-sol`
and `openai/gpt-6-astra`. `model_info` describes any other run model, and
`extra_model_info` other names. The context limit comes only from
`max_input_tokens`. If it is missing or is not a positive integer, construction
fails: set `model_info={"max_input_tokens": 200000}` to the model's actual limit.
There is no context fallback. Unknown models default to no vision and $0 cost.

Credentials follow Harbor's own agents: the provider's usual variable
(`ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`, ...) from Harbor's agent env
(`--ae 'KEY=${VAR}'`), else from your shell. `api_key_env` names a different
variable for one model. Proximus blanks the variables it uses inside the
sandbox, so the model cannot read them. An OpenAI-compatible route never falls
back to `OPENAI_API_KEY`: without its own key, construction fails. Never put
keys in `--ak` or `llm_call_kwargs`: Harbor saves agent options verbatim in job
files.

## Options and customization

Options follow Harbor's Terminus 2 names where they exist: `max_turns`,
`enable_summarize`, `temperature`, `api_base`, `max_thinking_tokens`,
`model_info`, `llm_call_kwargs` and `prompt_template_path` (a Jinja template
around `{{ instruction }}`). Proximus adds `compaction_threshold` (default 0.85),
`enable_view_image` (default false), and `disallowed_tools`
(`list_submissions`, `view_image`). Unset options keep the
default behavior, and `llm_call_kwargs` wins over the other request settings.
`max_thinking_tokens` and `reasoning_effort` are mutually exclusive, and
`max_turns` counts every loop iteration, including recovery retries.

For deeper changes, subclass the agent and select it with
`--agent mypkg.module:MyProximus`:

```python
from harbor_proximus import Proximus, tools

class MyProximus(Proximus):
    def system_prompt(self) -> str:
        return super().system_prompt() + "\n\nPrefer small commits."

    def build_tools(self) -> dict[str, tools.Tool]:
        count = tools.Tool("count_lines", "...", Args, self._count)
        return {**super().build_tools(), count.name: count}

    async def _count(self, args) -> tools.ToolResult:
        result = await self.environment.exec(f"wc -l {args.path}", timeout_sec=30)
        return tools.ToolResult(str(result.stdout))
```

The hooks are `system_prompt()`, `initial_message()`, `summary_request()` and
`build_tools()`. The first and last run inside `__init__`, so set attributes they
need before `super().__init__()`; `self.environment` exists only during a run.
After a compaction the conversation restarts from a handoff that repeats the
instruction, not `initial_message()`.

## Task configuration

Tasks can declare options in `task.toml`. Harbor ignores these tables, so a runner
reads them with `harbor_proximus.task_agent_kwargs(task_dir, model, overrides,
defaults)` and sets the hard timeout to `harbor_proximus.hard_timeout_sec(kwargs)`:

```toml
[agent.proximus]
task_provides_timer = true

[agent.proximus.models."anthropic/claude-opus-5"]
reasoning_effort = "max"
```

Later layers win: defaults, `[agent.proximus]`, the run's model table, overrides.
`time_budget_sec` defaults to `[agent].timeout_sec`, relative template paths are
task-relative, and `workspace_git_excludes` defaults to the task's `/app` artifact
excludes. Invalid options raise `ValueError` before any sandbox starts.

## Run on FrontierSWE v2

[px-eval](https://github.com/Proximal-Labs/px-eval) is a thin runner for
FrontierSWE v2 that supports Proximus. It reads each task's `[agent.proximus]` table,
sets the hard timeout from `hard_timeout_sec`, and grades the runs later in clean
verifier sandboxes:

```sh
uv run python examples/frontier_swe_v2/rollout.py /path/to/frontier-swe-v2/tasks \
  --environment environment.yaml --agent proximus --model anthropic/claude-opus-5 \
  --ak time_budget_sec=7200 --ae 'ANTHROPIC_API_KEY=${MY_KEY}'
```

See its [FrontierSWE v2 guide](https://github.com/Proximal-Labs/px-eval/blob/main/examples/frontier_swe_v2/README.md#run-with-proximus)
for setup, every option and grading.

## Limits

- Requires Harbor `0.24.0` and Python 3.12+; tested on Python 3.12.
- No MCP servers, Harbor skills, session resume or handoff; tasks configuring MCP
  servers or skills fail at setup.
- Reproducing a published score needs the original model configuration.

## Development

```sh
uv sync --locked
uv run pytest -q          # offline: fake providers, temporary Git repos
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv build && uv run python scripts/verify_wheel.py dist/harbor_proximus-0.1.0a1-py3-none-any.whl
uv run python scripts/smoke_models.py anthropic/claude-opus-5-5 \
  openai/gpt-5.6-sol --reasoning-effort high   # live: needs keys, costs cents
```

Through the Respan gateway (one key for every provider), point each API at it:
`--api-key-env RESPAN_API_KEY --anthropic-base https://api.respan.ai/api/anthropic/
--openai-base https://api.respan.ai/api`. In a run, the same is
`api_key_env=RESPAN_API_KEY` plus `api_base`.

`verify_wheel.py` reruns the suite against the installed wheel outside the
checkout and checks that litellm stays uninstalled. `smoke_models.py` runs a
tool call and its replay against each live API.

### Formatting and lint
We use ruff.

```sh
uv run --locked ruff check --fix src tests scripts
uv run --locked ruff format src tests scripts
```

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

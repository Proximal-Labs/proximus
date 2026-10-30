# Proximus

Proximus is a coding agent for long-horizon tasks, packaged as an external agent
for [Harbor](https://pypi.org/project/harbor/). It runs a LiteLLM tool loop on the
orchestrator host and acts in the sandbox only through Harbor's environment
interface, so it works with any Harbor environment (Docker, Modal, ...).

Status: alpha (`0.1.0a1`), for Harbor `0.23.0`.

## Install and run

```sh
uv pip install "harbor-proximus @ git+https://github.com/Proximal-Labs/proximus"   # next to Harbor
harbor run ... --agent harbor_proximus:Proximus --model anthropic/claude-opus-5 \
  --ak time_budget_sec=7200
```

[examples/agent.yaml](examples/agent.yaml) shows a job-config fragment.
`harbor agent schema harbor_proximus:Proximus` lists every option.

## How it works

- **Tools:** `bash`, `submit`, `list_submissions`, and `view_image` for vision models.
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
- **Sandbox needs:** `git` and `bash`.

## Models and credentials

Any LiteLLM model works (`anthropic/…`, `openai/…`, `openrouter/…`, `gemini/…`,
or `openai/<model>` with `api_base` for OpenAI-compatible endpoints). There are
no built-in model presets: pass `reasoning_effort`, `llm_call_kwargs` or
`model_info` as needed. The context limit comes only from LiteLLM's
`max_input_tokens`. If it is missing or is not a positive integer, construction
fails: set `model_info={"max_input_tokens": 200000}` to the model's actual limit.
There is no context fallback. Unknown models default to no vision and $0 cost.

Credentials follow Harbor's own agents: the provider's usual variable
(`ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`, ...) from Harbor's agent env
(`--ae 'KEY=${VAR}'`), else from your shell. `api_key_env` names a different
variable for one model. Proximus blanks the variables it uses inside the
sandbox, so the model cannot read them; Bedrock, SageMaker and Vertex keep
LiteLLM's own lookup. Never put keys in `--ak` or `llm_call_kwargs`: Harbor saves
agent options verbatim in job files.

### Gemini through the Interactions API

`model_api="gemini_interactions"` sends a `gemini/…` model to Google's
[Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
with the `google-genai` SDK instead of LiteLLM's `generateContent` route. Install
the extra: `harbor-proximus[gemini]`.

- **Stored, as Google recommends.** Interactions are stored (`store=true`) and
  each turn continues the previous one with `previous_interaction_id`, sending
  only the new steps. Google keeps stored interactions for a limited time (see
  its data retention terms). After compaction or image eviction rewrites the
  history, the next request sends the whole history and starts a new chain.
- **Stateless on request.** With `llm_call_kwargs={"store": false}` every
  request sends the whole history. The model's own steps (thoughts with
  signatures, function calls, output) are kept on each assistant message and
  replayed verbatim, as stateless requests require.
- **Settings.** `reasoning_effort` becomes `thinking_level`, with thought
  summaries on. In `llm_call_kwargs`, `max_tokens`, `temperature`, `top_p`,
  `seed`, `stop` and a `generation_config` dict go into `generation_config`;
  `timeout`, `api_version`, `extra_headers` and `extra_body` are SDK call
  options; any other key is sent as a top-level request field.
- **Same handling.** Replies, usage (thought tokens count as completion tokens,
  as on `generateContent`), cost from the model's LiteLLM prices, retries and
  context-overflow recovery work as on the default path. The SDK's own retries
  are off, so each attempt is one HTTP request.
- **Logged by LiteLLM.** LiteLLM logging callbacks fire for every attempt,
  success or failure, with the standard logging payload of a chat completion:
  `api_base` is the Interactions endpoint and `messages` the chat history.
- **Failed interactions.** An interaction with status `failed`, or with an error
  on a `model_output` step, raises a retryable `InternalServerError` carrying
  the raw status and errors. The Interaction schema has no refusal or safety
  field, so a blocked reply cannot be told from a server fault except by those
  details.
  Each reply's choice records the raw status as `interaction_status`.
- **Key.** `GEMINI_API_KEY` (or `api_key_env`), as for `gemini/` on LiteLLM.

## Options and customization

Options follow Harbor's Terminus 2 names where they exist: `max_turns`,
`enable_summarize`, `temperature`, `api_base`, `max_thinking_tokens`,
`model_info`, `llm_call_kwargs` and `prompt_template_path` (a Jinja template
around `{{ instruction }}`). Proximus adds `compaction_threshold` (default 0.85)
and `disallowed_tools` (`list_submissions`, `view_image`). Unset options keep the
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

- Requires Harbor `0.23.0` and Python 3.12+; tested on Python 3.12–3.14 with
  LiteLLM 1.97.0 and 1.105.1.
- Installs a small process-wide LiteLLM shim for reasoning-only Responses turns
  at LLM construction; Harbor itself is not patched.
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
```

`verify_wheel.py` reruns the suite against the installed wheel outside the
checkout; `--litellm-version 1.97.0` checks the lower bound.

### Formatting and lint
We use ruff.

```sh
uv run --locked ruff check --fix src tests scripts
uv run --locked ruff format src tests scripts
```

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

from pathlib import Path
from typing import Any, Self

from harbor.agents.options import AgentOptions
from harbor.utils.templating import render_prompt_template
from pydantic import ConfigDict, Field, field_validator, model_validator

# Margin the loop reserves against the time budget for one last operation.
BUDGET_MARGIN_SEC = 120


class ProximusConfig(AgentOptions):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    time_budget_sec: float | None = None
    """Wall-clock budget; enables the in-sandbox deadline file and the
    time-left line in submit responses. ``override_timeout_sec`` must sit
    comfortably ABOVE this so the loop ends gracefully before the
    orchestrator's kill."""

    submission_timeout_sec: int = 600
    """Maximum time for an in-sandbox submission Git operation. Large model
    artifacts can take several minutes to stage and commit; the old 60-second
    ceiling could kill ``git add`` mid-write and leave no winning submission."""

    auto_submit_on_exit: bool = True
    """At run end, pin the live workspace as a final submission via the same
    path as an argless ``submit()`` — recorded in ``submissions.jsonl``
    (flagged ``auto``), graded like any other submission. Skipped when the
    tree already matches the newest submission, so a clean double-submit
    ending never grows a duplicate. This is what
    makes a timed-out run gradeable at its last state."""

    restore_on_exit: bool = True
    """At run end, check out the winning submission so the verifier, and
    Harbor's capture of the workspace, see the submitted tree. Set
    False only when grading runs offline from the pinned refs: the
    workspace is then left at kill-time state and every scoring consumer
    must go through the pins."""

    finalize_only_on_submit: bool = False
    """When true, the exit restore (and auto-submit) runs only after a
    confirmed model submit. Budget, stall, provider, and environment-loss
    endings stay ungradeable."""

    task_provides_timer: bool = False
    """The task instruction owns the timer story (e.g. FrontierSWE's
    ``sandbox-timer``). When True, Proximus writes no deadline file and the
    system prompt defers to the task's timer instead of introducing a
    second clock — but the task timer still gets RE-ANCHORED to the budget
    (untouched it counts the sandbox's hard timeout, and the instruction's
    clock must tell the truth). ``time_budget_sec`` still paces the loop."""

    reasoning_effort: str | None = None
    """Optional reasoning effort. ``drop_params`` does not sanitize invalid
    values: an unsupported value (e.g. ``max`` on gpt-5.5) is still a 400."""

    api_key_env: str | None = None
    """Name of the environment variable holding this model's API key. litellm
    resolves keys by PROVIDER PREFIX (everything on the ``openai/`` route reads
    ``OPENAI_API_KEY``), so two OpenAI-compatible endpoints can otherwise
    compete for the same variable. Naming the variable here (never the key
    itself: agent configs are saved with the run's results) lets every model
    carry its own credential.
    Read from Harbor's agent env first, then the host. Without it, the
    provider's usual key variable (e.g. ``ANTHROPIC_API_KEY``) is taken from the
    agent env, else LiteLLM reads it from the host; base URLs (e.g.
    ``OPENAI_BASE_URL``) follow the same rule either way. Variables taken from
    the agent env are blanked in the sandbox for the run."""

    workspace_git_excludes: list[str] = Field(default_factory=list)
    """Gitwildmatch patterns written to the workspace repo's
    ``.git/info/exclude`` at git bootstrap (and already-tracked matches
    untracked), BEFORE the baseline commit. Sourced from the task's ``/app``
    artifacts exclude list: grading quarantines those paths anyway, so
    letting them into commits only bloats every submission and the model's
    own ``git status``. ``.git/info/exclude``
    rather than ``.gitignore`` so the agent-visible tree is untouched."""

    litellm_model_info: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Registry entries merged into litellm's global model map at LLM init
    (``litellm.register_model``), for models litellm does not ship. The
    registry gates three behaviors: the context limit (unknown model falls
    back to 200K), the ``view_image`` tool (unknown means no vision, tool
    removed), and per-turn cost telemetry (unknown prices as $0). Values
    follow litellm's model-map schema (``max_input_tokens``,
    ``input_cost_per_token``, ``supports_vision``, ...). openai-route models
    need BOTH keys: the prefixed handle (``openai/test-model`` — read by
    ``get_model_info``/``supports_vision`` at init) and the bare name
    (``test-model`` — echoed in responses, prices the cost calculator)."""

    llm_call_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra kwargs forwarded to ``litellm.acompletion`` (temperature,
    max_tokens, api_base, timeout, ...). They win over every other request
    setting, including ``temperature``, ``api_base`` and
    ``max_thinking_tokens``."""

    # Options below follow Harbor's Terminus 2 names. Unset, each one keeps
    # the default behavior.

    max_turns: int | None = Field(default=None, ge=1)
    """Maximum agent turns. Omit for no limit. Reaching it ends the run like
    an expired time budget, with the usual exit auto-submit and restore."""

    enable_summarize: bool = True
    """Compact the conversation into a handoff summary near the context
    limit. When false, a context overflow ends the run with an error."""

    compaction_threshold: float = Field(default=0.85, gt=0, le=1)
    """Fraction of the context limit at which the conversation is compacted."""

    disallowed_tools: list[str] = Field(default_factory=list)
    """Tools removed from the model's tool set, e.g. ``list_submissions`` or
    ``view_image``. ``bash`` and ``submit`` cannot be removed."""

    prompt_template_path: str | None = None
    """Jinja2 template wrapping the task instruction; must include
    ``{{ instruction }}``. Same as Harbor's installed agents."""

    temperature: float | None = None
    """Sampling temperature. Omit to use the backend default."""

    api_base: str | None = None
    """LLM API base URL."""

    max_thinking_tokens: int | None = Field(default=None, ge=1024)
    """Anthropic extended-thinking budget (min 1024). Omit for the model default."""

    model_info: dict[str, Any] | None = None
    """LiteLLM model info for the run's model (context limit, prices,
    vision), for models LiteLLM does not ship. Shorthand for a
    ``litellm_model_info`` entry under both the prefixed and bare model name."""

    @field_validator("prompt_template_path")
    @classmethod
    def _template_renders(cls, value: str | None) -> str | None:
        if value is not None:
            if not Path(value).is_file():
                raise ValueError(f"prompt_template_path not found: {value}")
            render_prompt_template(
                Path(value), "instruction"
            )  # needs {{ instruction }}
        return value

    @model_validator(mode="after")
    def _one_thinking_control(self) -> Self:
        if self.max_thinking_tokens is not None and self.reasoning_effort is not None:
            raise ValueError(
                "set either max_thinking_tokens or reasoning_effort, not both: "
                "LiteLLM maps both onto the provider's thinking setting"
            )
        return self

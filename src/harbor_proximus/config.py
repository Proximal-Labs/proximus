from pathlib import Path
from typing import Any, Literal, Self

from harbor.agents.options import AgentOptions
from harbor.utils.templating import render_prompt_template
from pydantic import ConfigDict, Field, field_validator, model_validator

# Seconds kept free at the end of the time budget: bash commands time out this
# long before the budget runs out, and a model request always gets at least
# this long, so the run can still finish cleanly.
BUDGET_MARGIN_SEC = 120
# thinking_level values the Gemini Interactions API accepts.
GEMINI_THINKING_LEVELS = frozenset({"minimal", "low", "medium", "high"})


class ProximusConfig(AgentOptions):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    time_budget_sec: float | None = None
    """Total time budget in seconds. The run stops when it is used up, whatever
    any task timer says. Set Harbor's hard agent timeout higher (see
    ``hard_timeout_sec``) so the exit submit and restore can finish."""

    submission_timeout_sec: int = 600
    """Seconds allowed for one submission's git work in the sandbox. Large
    workspaces can take minutes to commit."""

    auto_submit_on_exit: bool = True
    """When the run ends, save any unsubmitted work as a final submission, so a
    run that times out is still graded on its last state. Skipped if nothing
    changed since the last submission."""

    restore_on_exit: bool = True
    """When the run ends, check out the winning submission so the verifier sees
    it. Turn off only if grading reads the submitted commits directly; the
    workspace is then left as it was when the run stopped."""

    finalize_only_on_submit: bool = False
    """Do the exit submit and restore only if the model confirmed a submission.
    Runs that end any other way (time out, stall, error) are left as they are."""

    task_provides_timer: bool = False
    """Set when the task instruction already tells the model how to check the
    time (for example FrontierSWE's ``sandbox-timer``). Only changes what the
    model is told: Proximus skips its own deadline file and points the model to
    the task's timer. The run still ends at ``time_budget_sec``, and the task's
    timer is reset to show the same remaining time either way."""

    reasoning_effort: str | None = None
    """Reasoning effort passed to the model. A value the model does not
    support is rejected by the provider."""

    api_key_env: str | None = None
    """Name of the environment variable that holds this model's API key (never
    the key itself). Use it when two models share a provider route but need
    different keys. Read from Harbor's agent env first, then the host. Without
    it, the provider's usual variable is used."""

    workspace_git_excludes: list[str] = Field(default_factory=list)
    """Paths that submissions never commit, in ``.gitignore`` syntax (for
    example caches and checkpoints that grading ignores anyway). Written to
    ``.git/info/exclude``, so the visible workspace is unchanged."""

    litellm_model_info: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Extra LiteLLM model entries, keyed by model name, for models LiteLLM
    does not know. They set the context limit, vision support (needed for
    ``view_image``) and prices. A positive integer ``max_input_tokens`` is
    required; there is no context fallback. Unknown models get no vision and
    zero cost. For an ``openai/`` route, add both ``openai/<model>`` and
    ``<model>``. ``model_info`` is the shorthand for the run's model."""

    llm_call_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra arguments for every model request (for example ``max_tokens``,
    ``timeout`` or ``extra_body``). They override all other request settings."""

    max_turns: int | None = Field(default=None, ge=1)
    """Maximum number of loop iterations, including retries. Reaching it ends
    the run like a time-out, with the usual exit submit and restore."""

    enable_summarize: bool = True
    """Summarize the conversation when it nears the context limit. When off, a
    full context window ends the run with an error."""

    compaction_threshold: float = Field(default=0.85, gt=0, le=1)
    """Fraction of the context limit at which the conversation is summarized."""

    disallowed_tools: list[str] = Field(default_factory=list)
    """Tools to remove: ``list_submissions`` or ``view_image``. ``bash`` and
    ``submit`` are always available."""

    prompt_template_path: str | None = None
    """Jinja2 template around the task instruction. It must contain
    ``{{ instruction }}``."""

    temperature: float | None = None
    """Sampling temperature. Required for ``gemini_interactions``; otherwise
    omit to use the provider's default."""

    api_base: str | None = None
    """Base URL of the model API."""

    model_api: Literal["chat", "gemini_interactions"] = "chat"
    """How requests reach the model. ``chat`` (the default) goes through
    LiteLLM. ``gemini_interactions`` uses LiteLLM's Interactions route for
    ``gemini/`` models, with no extra dependency. Every call uses ``store=False``
    and replays the full history. ``reasoning_effort`` is required and becomes
    ``thinking_level``; ``max_tokens`` and other sampling settings in
    ``llm_call_kwargs`` go into ``generation_config``."""

    max_thinking_tokens: int | None = Field(default=None, ge=1024)
    """Thinking budget for Anthropic models (at least 1024). Cannot be combined
    with ``reasoning_effort``."""

    model_info: dict[str, Any] | None = None
    """LiteLLM model info for the run's model (context limit, vision, prices).
    Set ``max_input_tokens`` to a positive integer if the registry has no valid
    input limit. This is the only field used for the context limit."""

    @field_validator("prompt_template_path")
    @classmethod
    def _template_renders(cls, value: str | None) -> str | None:
        if value is not None:
            if not Path(value).is_file():
                raise ValueError(f"prompt_template_path not found: {value}")
            # Raises unless the template uses {{ instruction }}.
            render_prompt_template(Path(value), "instruction")
        return value

    @model_validator(mode="after")
    def _interactions_thinking_level_only(self) -> Self:
        if (
            self.model_api == "gemini_interactions"
            and self.max_thinking_tokens is not None
        ):
            raise ValueError(
                "model_api='gemini_interactions' uses reasoning_effort "
                "(thinking_level), not max_thinking_tokens"
            )
        if (
            self.model_api == "gemini_interactions"
            and self.reasoning_effort is not None
            and self.reasoning_effort not in GEMINI_THINKING_LEVELS
        ):
            raise ValueError(
                "model_api='gemini_interactions' needs reasoning_effort in "
                f"{sorted(GEMINI_THINKING_LEVELS)}, got {self.reasoning_effort!r}"
            )
        return self

    @model_validator(mode="after")
    def _one_thinking_control(self) -> Self:
        if self.max_thinking_tokens is not None and self.reasoning_effort is not None:
            raise ValueError(
                "set either max_thinking_tokens or reasoning_effort, not both: "
                "LiteLLM maps both onto the provider's thinking setting"
            )
        return self

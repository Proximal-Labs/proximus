from pathlib import Path
from typing import Any, Literal, Self

from harbor.agents.options import AgentOptions
from harbor.utils.templating import render_prompt_template
from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

# Seconds kept free at the end of the time budget: bash commands time out this
# long before the budget runs out, and a model request always gets at least
# this long, so the run can still finish cleanly.
BUDGET_MARGIN_SEC = 120

PROGRESS_FILE = "/tmp/proximus/PROGRESS.md"
BASH_OUTPUT_FILE_PREFIX = "/tmp/proximus-output-"


class ProximusConfig(AgentOptions):
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    time_budget_sec: float | None = None
    """Total time budget in seconds. The run stops when it is used up, whatever
    any task timer says. Set Harbor's hard agent timeout higher (see
    ``hard_timeout_sec``) so the exit submit and restore can finish."""

    submission_mode: Literal["git", "none"] = "git"
    """With ``git``, record and restore workspace submissions. With ``none``,
    expose ``done`` instead of submission tools and leave workspace files
    unchanged on exit. No repository is initialized; Git-specific options
    have no effect. The task defines its own deliverables."""

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

    extra_model_info: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("extra_model_info", "litellm_model_info"),
    )
    """Model entries, keyed by model name, for models the built-in table does
    not know (for example the model a server-side fallback answered with).
    They set the context limit, vision support (needed for ``view_image``) and
    prices, with the built-in table's keys. Unknown models get no vision and
    zero cost. ``model_info`` is the shorthand for the run's model. The old
    name ``litellm_model_info`` is still accepted."""

    llm_call_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra arguments for every model request, in the API's own names (for
    example ``max_tokens``, ``timeout``, ``stream`` or ``extra_body``). They
    override all other request settings; names the SDK does not know are sent
    in the request body."""

    max_turns: int | None = Field(default=None, ge=1)
    """Maximum number of loop iterations, including retries. Reaching it ends
    the run like a time-out, with the usual exit submit and restore."""

    enable_summarize: bool = True
    """Summarize the conversation when it nears the context limit. When off, a
    full context window ends the run with an error."""

    compaction_threshold: float = Field(default=0.85, gt=0, le=1)
    """Fraction of the context limit at which the conversation is summarized."""

    enable_view_image: bool = False
    """Opt in to ``view_image``. Construction fails unless the model has known
    vision support. For an unregistered vision model, set its actual capability
    with ``model_info={\"supports_vision\": True, ...}``. Cannot be combined with
    ``view_image`` in ``disallowed_tools``."""

    disallowed_tools: list[str] = Field(default_factory=list)
    """Tools to remove: ``list_submissions`` or ``view_image``. ``bash`` and
    the mode's completion tool (``submit`` or ``done``) cannot be removed."""

    prompt_template_path: str | None = None
    """Jinja2 template around the task instruction. It must contain
    ``{{ instruction }}``."""

    progress_file: str = PROGRESS_FILE
    """The model's durable notes file in the sandbox. The system prompt and the
    compaction message name it, and each compaction re-reads it."""

    bash_output_file_prefix: str = BASH_OUTPUT_FILE_PREFIX
    """Start of the sandbox paths where ``bash`` saves the full output of a
    command that is too long to show."""

    temperature: float | None = None
    """Sampling temperature. Omit to use the provider's default."""

    api_base: str | None = None
    """Base URL of the model API."""

    max_thinking_tokens: int | None = Field(default=None, ge=1024)
    """Thinking budget for Anthropic models (at least 1024). Cannot be combined
    with ``reasoning_effort``."""

    model_info: dict[str, Any] | None = None
    """Model info for the run's model (context limit, vision, prices).
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
    def _consistent_image_options(self) -> Self:
        if self.enable_view_image and "view_image" in self.disallowed_tools:
            raise ValueError(
                "enable_view_image=True conflicts with disallowed_tools containing "
                "view_image; disable the option or remove the exclusion"
            )
        return self

    @model_validator(mode="after")
    def _one_thinking_control(self) -> Self:
        if self.max_thinking_tokens is not None and self.reasoning_effort is not None:
            raise ValueError(
                "set either max_thinking_tokens or reasoning_effort, not both: "
                "both map onto the provider's thinking setting"
            )
        return self

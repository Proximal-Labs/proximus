import contextlib
import itertools
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError
from harbor.models import trajectories
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.utils.templating import render_prompt_template
from harbor.utils.trajectory_utils import format_trajectory_json

import harbor_proximus.llm as llm
import harbor_proximus.prompts as prompts
import harbor_proximus.submissions as submissions
import harbor_proximus.timer as timer
import harbor_proximus.tools as tools
import harbor_proximus.utils as utils
from harbor_proximus.config import ProximusConfig

EMPTY_TURN_STALL_LIMIT = 3
# Fraction of the context window at which proactive compaction runs
# Most-recent messages preserved verbatim across a compaction
KEEP_RECENT_MESSAGES = 6
# Byte cap on the PROGRESS.md read folded into each compaction handoff
PROGRESS_REINJECT_MAX_BYTES = 25_000


@dataclass
class RunState:
    messages: list[llm.Message] = field(default_factory=list)
    session_id: str = ""
    start_monotonic: float = field(default_factory=time.monotonic)
    n_turns: int = 0
    # Turn of the submit awaiting confirmation; see _submit.
    submit_armed_turn: int | None = None
    ended_by_submit: bool = False
    env_lost: bool = False
    consecutive_empty_turns: int = 0
    stalled: bool = False
    last_context_tokens: int | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    fallback_attempted_turns: int = 0
    fallback_turns: int = 0
    fallback_models: dict[str, int] = field(default_factory=dict)
    n_images: int = 0  # viewed images saved under <logs_dir>/images/


class Proximus(BaseAgent):
    """Long-horizon, native-function-calling agent harness."""

    capabilities = AgentCapabilities(atif=True)
    options_model = ProximusConfig

    _environment: BaseEnvironment
    _executor: tools.ToolExecutor
    _submissions: submissions.Submissions
    _instruction: str
    _tools: dict[str, tools.Tool]
    _tool_schemas: list[dict[str, Any]]

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        skills_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        load_trajectory: str | Path | None = None,
        environment_logs_dir: PurePosixPath | None = None,
        override_setup_timeout_sec: float | None = None,
        **kwargs,
    ):
        if model_name is None:
            raise ValueError("Proximus requires model_name to be set")
        super().__init__(
            logs_dir,
            model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=extra_env,
            load_trajectory=load_trajectory,
            environment_logs_dir=environment_logs_dir,
            override_setup_timeout_sec=override_setup_timeout_sec,
            **kwargs,
        )
        assert isinstance(self.options, ProximusConfig)
        self.config = self.options
        self._llm = llm.LLM(
            model_name,
            self.config,
            self._remaining_budget_sec,
            self.logger,
            provider_health_path=self.logs_dir / "provider_health.json",
            extra_env=self.extra_env,
        )
        self._context_limit = self._llm.resolve_context_limit()
        self._system_prompt = self.system_prompt()
        self._tools = self._filter_tools(self.build_tools())
        self._tool_schemas = [tool.schema() for tool in self._tools.values()]
        self._run = RunState()
        self._steps: list[trajectories.Step] = []

    @staticmethod
    @override
    def name() -> str:
        return "proximus"

    @override
    def version(self) -> str | None:
        from importlib.metadata import version

        return version("harbor-proximus")

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        if self.mcp_servers or self.skills_dir:
            raise ValueError(
                "Proximus supports neither MCP servers nor skills; "
                "remove them from the task/agent config please."
            )

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Harbor overlays the agent env onto every sandbox command. The model
        # credentials Proximus read from it are for the host-side LLM client
        # only: blank them in the sandbox (a nested overlay wins).
        hidden = {name: "" for name in sorted(self._llm.consumed_env_keys)}
        scope = getattr(environment, "scoped_exec_env", None)
        with scope(hidden) if hidden and scope else contextlib.nullcontext():
            await self._run_task(instruction, environment, context)

    async def _run_task(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Setup for every run."""
        self._run = RunState()
        self._steps = []
        if self.config.prompt_template_path:
            instruction = render_prompt_template(
                Path(self.config.prompt_template_path), instruction
            )
        self._instruction = instruction
        self._environment = environment
        workdir = await self._resolve_workdir(environment)

        self._run.session_id = self.session_id or str(uuid.uuid4())
        self._llm.session_id = self._run.session_id

        await self._ensure_progress_dir(environment)
        self._executor = tools.ToolExecutor(
            environment,
            default_workdir=workdir,
            remaining_sec=self._remaining_budget_sec,
        )
        await self._init_submissions(environment, workdir)
        await self._arm_timer(environment)
        self._seed_conversation(instruction, workdir, environment)

        try:
            await self._agent_loop()
        finally:
            try:
                should_finalize = (
                    not self.config.finalize_only_on_submit or self._run.ended_by_submit
                )
                if self._run.env_lost:
                    self.logger.info("End-of-run restore skipped: the sandbox is gone.")
                elif not should_finalize:
                    self.logger.info(
                        "End-of-run restore skipped: the model did not "
                        "end with a confirmed submit."
                    )
                else:
                    if (
                        self.config.auto_submit_on_exit
                        and not self._run.ended_by_submit
                    ):
                        await self._auto_final_submission()
                    if not self._run.env_lost and self.config.restore_on_exit:
                        await self._submissions.restore_winner()
            finally:
                self._populate_context(context)
                self._dump_artifacts()

    async def _agent_loop(self) -> None:
        for turn in itertools.count(1):
            self._run.n_turns = turn

            budget = self.config.time_budget_sec
            if budget and self._elapsed() >= budget:
                self.logger.debug("Time budget exhausted; ending run.")
                return
            if self.config.max_turns and turn > self.config.max_turns:
                self.logger.debug("Turn limit reached; ending run.")
                self._run.n_turns = turn - 1
                return

            tokens = self._run.last_context_tokens
            if (
                self.config.enable_summarize
                and tokens
                and tokens >= self.config.compaction_threshold * self._context_limit
            ):
                await self._compact()

            try:
                parsed = await self._llm.query(
                    self._run.messages, tools=self._tool_schemas
                )
            except ContextLengthExceededError:
                if not self.config.enable_summarize:
                    raise
                self.logger.debug("Context window exceeded; compacting.")
                await self._compact()
                continue  # retry the turn that overflowed
            except OutputLengthExceededError:
                self.logger.debug("Output length exceeded; asking for smaller output.")
                self._run.messages.append(
                    {"role": "user", "content": prompts.OUTPUT_LENGTH_MESSAGE}
                )
                continue
            except llm.ImageLimitExceededError:
                # A hard image-count cap means retaining 49/50 old images only
                # postpones the next failure. Rewriting even one image already
                # invalidates the cached prefix, so clear image payloads and
                # compact once: the summary preserves the model's textual
                # findings while the trajectory keeps the viewed image files.
                evicted = self._evict_all_images()
                if not evicted:
                    raise  # a claimed image limit with no images is not repairable
                self.logger.debug(
                    "Provider image-count limit exceeded; removed %d images "
                    "and compacting context.",
                    evicted,
                )
                if self.config.enable_summarize:
                    await self._compact()
                continue  # retry the rejected turn with compacted history
            except llm.InvalidImageError as e:
                # The provider rejected an attached image before the model
                # could see this turn. Remove the newest image (the one just
                # returned by view_image), then relay the exact error so the
                # model can resize/pad/repair the source and view it again.
                if not self._evict_newest_image(self._INVALID_IMAGE_EVICTED_NOTE):
                    raise  # no attached image means the request is not repairable here
                self.logger.debug(
                    "Provider rejected the newest attached image; removed it "
                    "and informed the model."
                )
                self._run.messages.append(
                    {
                        "role": "user",
                        "content": prompts.INVALID_IMAGE_MESSAGE.format(error=e),
                    }
                )
                continue  # retry without the rejected payload
            except llm.RequestTooLargeError as e:
                # A gateway body cap (not the token window): the oversized
                # payload is the WHOLE history, so no message can reach the
                # model until the history shrinks. Evict the NEWEST attached
                # image (images are the only mass that can outgrow the cap
                # under a model's context limit) and relay the provider
                # error — the model can crop/compress the file and view it
                # again. Newest-first costs the least prompt cache: the
                # prefix cache is busted only from the edited position, and
                # everything before the last image stays cached. Still too
                # large -> the next round evicts the next-newest.
                if not self._evict_newest_image():
                    raise  # nothing evictable made the request oversized
                self.logger.debug(
                    "Request exceeded the provider body cap; evicted the "
                    "newest attached image and informed the model."
                )
                note = prompts.REQUEST_TOO_LARGE_MESSAGE.format(error=e)
                # Consecutive evictions in one recovery burst update the
                # existing note instead of stacking new user messages.
                last = self._run.messages[-1]
                if last.get("role") == "user" and str(last.get("content")).startswith(
                    "Request failed: "
                ):
                    last["content"] = note
                else:
                    self._run.messages.append({"role": "user", "content": note})
                continue  # retry the turn with the slimmed history
            except Exception as e:
                budget = self.config.time_budget_sec
                if budget and self._elapsed() >= budget and llm._is_retryable(e):
                    # The retry loop was stopped by the BUDGET, not by the
                    # failure class (a provider down longer than the run had
                    # left). That is a failed trial: returning cleanly would
                    # mark it ``done`` and auto-grade a partial pin.
                    self.logger.debug(
                        "Budget expired during provider retries; failing run."
                    )
                    raise llm.ProviderRetryBudgetError(
                        "Budget expired during provider retries"
                    ) from e
                raise

            self._accumulate_usage(parsed)
            self._run.messages.extend(self._llm.assistant_messages(parsed))
            stalled = self._model_stalled(parsed)

            if not parsed.tool_calls:
                self._add_agent_step(parsed, [], [])
                self._dump_artifacts()
                if stalled:
                    self.logger.debug(
                        f"No output for {EMPTY_TURN_STALL_LIMIT} consecutive "
                        "turns; ending run."
                    )
                    self._run.stalled = True
                    return

                if self._run.submit_armed_turn is not None:
                    closer = prompts.CONTINUE_OR_CONFIRM_MESSAGE
                elif self._run.consecutive_empty_turns > 0:
                    closer = prompts.EMPTY_RESPONSE_MESSAGE
                else:
                    closer = prompts.CONTINUE_MESSAGE
                self._run.messages.append({"role": "user", "content": closer})
                continue

            end_run = await self._execute_tool_calls(parsed)
            self._dump_artifacts()
            if end_run:
                return

    def _model_stalled(self, parsed: llm.ParsedResponse) -> bool:
        """Track zero-output turns; True once EMPTY_TURN_STALL_LIMIT land in
        a row."""
        self._run.consecutive_empty_turns = (
            self._run.consecutive_empty_turns + 1 if parsed.is_empty else 0
        )
        return self._run.consecutive_empty_turns >= EMPTY_TURN_STALL_LIMIT

    async def _execute_tool_calls(self, parsed: llm.ParsedResponse) -> bool:
        """Run the turn's tool calls; True when a confirmed final submission
        (or a dead sandbox) ends the run"""
        end_run = False
        env_gone: tools.EnvironmentGoneError | None = None
        observation_results: list[trajectories.ObservationResult] = []
        atif_tool_calls: list[trajectories.ToolCall] = []
        pending_images: list[tuple[str, str]] = []
        saved_images: list[str] = []

        for call in parsed.tool_calls:
            disarmed = False
            if call.name != "submit" and self._run.submit_armed_turn is not None:
                # The confirm protocol requires submit to be the model's next
                # action; anything else cancels the pending confirmation. A
                # silent lapse leaves models puzzled ("I submitted twice — why
                # hasn't it ended?"), so the cancellation is stated on the very
                # call that caused it.
                self._run.submit_armed_turn = None
                disarmed = True
            tool = self._tools.get(call.name)
            if env_gone is not None:
                result = tools.ToolResult(
                    "Not executed: the sandbox became unreachable earlier in "
                    "this turn.",
                    is_error=True,
                )
            elif end_run:
                # Calls after a confirming submit must not execute, but each
                # still needs a result to keep the conversation valid.
                result = tools.ToolResult(
                    "Not executed: an earlier submit in this turn ended the run.",
                    is_error=True,
                )
            elif tool is None:
                result = tools.ToolResult(
                    f"tools.Tool '{call.name}' is not available in this session. "
                    f"Available tools: {', '.join(sorted(self._tools))}.",
                    is_error=True,
                )
            else:
                try:
                    result = await tool.call(call.raw_arguments)
                except tools.EnvironmentGoneError as e:
                    # Ends the run, but only after the batch drains so every
                    # tool_call still gets a result.
                    env_gone = e
                    result = tools.ToolResult(
                        f"tools.Tool call failed: {e}", is_error=True
                    )

            if disarmed:
                result.content += (
                    "\n\n(Note: this call cancelled your pending submission "
                    "confirmation — the recorded submission itself is kept. To "
                    "end the run, call submit and then submit again with no "
                    "other tool calls in between.)"
                )

            if call.name == "submit" and result.ends_run:
                self._run.ended_by_submit = True
            end_run = end_run or result.ends_run
            if result.image is not None:
                # Deferred: an image message must not split the contiguous
                # tool results.
                pending_images.append(result.image)
                path = self._save_image(*result.image)
                if path:
                    saved_images.append(path)

            # Appended per call so a mid-batch cancellation keeps the
            # completed results in the dumped state.
            self._run.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": result.content,
                }
            )
            observation_results.append(
                trajectories.ObservationResult(
                    source_call_id=call.call_id, content=result.content
                )
            )
            atif_tool_calls.append(
                trajectories.ToolCall(
                    tool_call_id=call.call_id,
                    function_name=call.name or "unknown",
                    arguments=tools.args_for_trajectory(call.raw_arguments),
                )
            )

        # end_run is set only after the drain: setting it in the except would
        # mislabel the remaining stubs as "ended by an earlier submit".
        if env_gone is not None:
            self.logger.error(
                "Environment is gone: %s. Ending the run rather than burning "
                "budget against a dead sandbox.",
                env_gone,
            )
            self._run.env_lost = True
            end_run = True

        if pending_images and not end_run:
            # Providers accept images only in user messages. Never evicted
            # mid-run: rewriting history busts the prompt cache.
            self._run.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            },
                        }
                        for media_type, image_b64 in pending_images
                    ],
                }
            )

        self._add_agent_step(
            parsed, atif_tool_calls, observation_results, images=saved_images
        )
        if env_gone is not None:
            # The trajectory and metadata are complete. Raise after recording
            # them so Harbor marks the trial as failed.
            raise env_gone
        return end_run

    _IMAGE_EXT = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
    }

    _IMAGE_EVICTED_NOTE = "[image removed: request too large]"
    _INVALID_IMAGE_EVICTED_NOTE = "[image removed: provider rejected it]"

    def _evict_newest_image(self, note: str | None = None) -> bool:
        """Replace the LAST attached image in history with a placeholder.

        Recovery for :class:`llm.RequestTooLargeError`: newest-first keeps
        the most prompt cache (the prefix before the edit stays valid) and
        targets the image the model can most easily re-view compressed.
        Returns False when no image remains to evict."""
        for message in reversed(self._run.messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for i in range(len(content) - 1, -1, -1):
                part = content[i]
                if isinstance(part, dict) and part.get("type") == "image_url":
                    content[i] = {
                        "type": "text",
                        "text": note or self._IMAGE_EVICTED_NOTE,
                    }
                    return True
        return False

    def _evict_all_images(self) -> int:
        """Remove all attached image payloads before image-limit compaction.

        The trajectory's saved files remain available, and the model can call
        ``view_image`` again after the compacted handoff when it needs one.
        """
        count = 0
        for message in self._run.messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for i, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    content[i] = {"type": "text", "text": self._IMAGE_EVICTED_NOTE}
                    count += 1
        return count

    def _save_image(self, media_type: str, image_b64: str) -> str | None:
        """Persist a viewed image under ``<logs_dir>/images/`` and return its
        RELATIVE path for the trajectory step — the only copy otherwise is the
        base64 blob inside messages.json. Best-effort: a failed write costs the
        archive one image, never the run."""
        import base64

        ext = self._IMAGE_EXT.get(media_type, "bin")
        name = f"turn{self._run.n_turns:04d}-{self._run.n_images:02d}.{ext}"
        self._run.n_images += 1
        try:
            path = self.logs_dir / "images" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(image_b64))
        except (OSError, ValueError) as e:
            self.logger.debug(f"Could not save viewed image {name}: {e}")
            return None
        return f"images/{name}"

    # ----- tools -----
    # Handlers delegate through self: the tool surface exists from
    # construction, while the executor and submissions bind in run().

    # ----- customization hooks -----
    # Override these in a subclass and select it with --agent module:Class,
    # the way Harbor agents are customized.

    def system_prompt(self) -> str:
        """The system prompt. Called once, at construction."""
        return prompts.system_prompt(self.config.task_provides_timer)

    def initial_message(
        self, instruction: str, workdir: str, environment: BaseEnvironment
    ) -> str:
        """The first user message, wrapping the (templated) task instruction."""
        return prompts.initial_user_message(
            instruction,
            workdir,
            self.config.time_budget_sec,
            environment.network_policy.network_mode,
        )

    def summary_request(self) -> str:
        """The request for a handoff summary when the conversation is compacted."""
        return prompts.SUMMARY_REQUEST

    def build_tools(self) -> list[tools.Tool]:
        """The model's tools. Called once, at construction; ``disallowed_tools``
        then removes tools by name. A subclass may add tools whose handlers
        run commands through ``self.environment`` during the run."""
        surface = [
            tools.Tool("bash", tools.BASH_DESCRIPTION, tools.BashArgs, self._bash),
            tools.Tool(
                "view_image",
                tools.VIEW_IMAGE_DESCRIPTION,
                tools.ViewImageArgs,
                self._view_image,
            ),
            tools.Tool(
                "list_submissions",
                tools.LIST_SUBMISSIONS_DESCRIPTION,
                tools.NoArgs,
                self._list_submissions,
            ),
            tools.Tool(
                "submit", tools.SUBMIT_DESCRIPTION, tools.SubmitArgs, self._submit
            ),
        ]
        if not self._llm.supports_vision():
            surface = [tool for tool in surface if tool.name != "view_image"]
        return surface

    @property
    def environment(self) -> BaseEnvironment:
        """The task sandbox of the current run."""
        return self._environment

    def _filter_tools(self, surface: list[tools.Tool]) -> dict[str, tools.Tool]:
        by_name = {tool.name: tool for tool in surface}
        if len(by_name) != len(surface):
            raise ValueError("build_tools() returned duplicate tool names")
        removed = set(self.config.disallowed_tools)
        if removed & {"bash", "submit"}:
            raise ValueError("disallowed_tools cannot remove bash or submit")
        # view_image is dropped for non-vision models, so naming it is fine.
        unknown = removed - set(by_name) - {"view_image"}
        if unknown:
            raise ValueError(f"disallowed_tools names unknown tools: {sorted(unknown)}")
        return {name: tool for name, tool in by_name.items() if name not in removed}

    async def _bash(self, args: tools.BashArgs) -> tools.ToolResult:
        return await self._executor.bash(args)

    async def _view_image(self, args: tools.ViewImageArgs) -> tools.ToolResult:
        return await self._executor.view_image(args)

    async def _list_submissions(self, _: tools.NoArgs) -> tools.ToolResult:
        return tools.ToolResult(prompts.submissions_list(self._submissions.summary()))

    async def _auto_final_submission(self) -> None:
        """Pin the end-of-run workspace as a final submission — the exact
        argless ``submit()`` path — so a run that ends without submitting
        (budget timeout, stall) still leaves its last state on the
        submissions rail: recorded in ``submissions.jsonl`` (flagged
        ``auto``), graded like any other
        submission. Skipped when the live tree already matches the newest
        submission — a clean double-submit ending must not grow a duplicate.
        Best-effort: nothing here may block the restore."""
        try:
            winner = self._submissions.winner_hash()
            if winner is not None and await self._submissions.matches_winner(
                "", winner
            ):
                self.logger.debug("End state already pinned; no auto-submission.")
                return
            submission = await self._submissions.submit("", auto=True)
            self.logger.info(
                "Auto-recorded end-of-run state as submission "
                f"#{self._submissions.success_count()} ({submission.hash[:12]})."
            )
        except tools.EnvironmentGoneError as e:
            self.logger.warning(f"Auto-submission skipped: sandbox is gone ({e}).")
            self._run.env_lost = True
        except submissions.SubmissionError as e:
            # Already recorded as an ok:false row by submissions.submit.
            self.logger.warning(f"Auto-submission failed: {e}")
        except Exception as e:  # noqa: BLE001 — never let it block the restore
            self.logger.warning(f"Auto-submission errored: {e}")

    async def _submit(self, args: tools.SubmitArgs) -> tools.ToolResult:
        """One submit records and arms; a second on a LATER turn confirms
        and ends the run (two submits in one turn must not self-confirm).
        A confirm naming the already-recorded commit REUSES that record:
        re-recording would pin the just-emptied staging dir as the newest
        winner, which is what the exit restore hands the verifier. A confirm
        naming a different commit is new work and records normally."""
        run = self._run
        confirmed = (
            run.submit_armed_turn is not None and run.submit_armed_turn < run.n_turns
        )
        run.submit_armed_turn = run.n_turns
        if confirmed:
            winner = self._submissions.winner_hash()
            if winner is not None and await self._submissions.matches_winner(
                args.commit, winner
            ):
                count = self._submissions.success_count()
                self.logger.info(f"Confirm reuses submission #{count}; run ends.")
                return tools.ToolResult(prompts.SUBMIT_CONFIRMED, ends_run=True)
        try:
            submission = await self._submissions.submit(args.commit)
            note = prompts.submission_note(
                self._submissions.success_count(),
                submission.hash,
                submission.description,
            )
        except submissions.SubmissionError as e:
            self.logger.warning(f"Submission recording failed: {e}")
            note = prompts.submission_failed(str(e), self._submissions.success_count())
        if confirmed:
            return tools.ToolResult(f"{note} {prompts.SUBMIT_CONFIRMED}", ends_run=True)
        guidance = prompts.submit_guidance(self._elapsed(), self.config.time_budget_sec)
        return tools.ToolResult(f"{note} {guidance}")

    # ----- context management -----
    # The summary request must look byte-identical to a normal turn (same
    # tools, no tool_choice) or the provider's cached prefix is invalidated.
    # History rewrites happen only where that prefix is rebuilt anyway.

    async def _compact(self) -> None:
        """Replace older history with a handoff message, keeping the recent
        tail verbatim (minus any tool results the cut stranded)."""
        summary = await self._handoff_summary()
        progress = await self._read_progress()
        handoff = prompts.handoff_message(
            self._instruction,
            summary,
            progress,
            self._elapsed(),
            self.config.time_budget_sec,
        )

        messages = self._run.messages
        tail = messages[max(2, len(messages) - KEEP_RECENT_MESSAGES) :]
        messages[:] = [
            messages[0],
            {"role": "user", "content": handoff},
            *llm.drop_orphan_tool_results(tail),
        ]
        self._run.last_context_tokens = None

        self.logger.debug("Compaction.")
        self._add_step(
            "system",
            "Context compacted; conversation history replaced with a handoff summary.",
            extra={"context_management": {"type": "compaction"}},
        )
        self._add_step("user", handoff)
        self._dump_artifacts()

    async def _handoff_summary(self) -> str | None:
        """Ask the model for a handoff summary. The prompt says not to call
        tools; if the model calls one anyway, ask again. On overflow, halve
        the history and retry (the last halving still gets its attempt)."""
        request: llm.Message = {"role": "user", "content": self.summary_request()}
        for attempt in range(3):
            try:
                parsed = await self._llm.query(
                    self._run.messages + [request], tools=self._tool_schemas
                )
                self._accumulate_usage(parsed)
                if parsed.content.strip():
                    return parsed.content
                if not parsed.tool_calls:
                    return None
            except ContextLengthExceededError:
                if attempt == 2:
                    return None
                self._drop_middle()
            except Exception as e:
                self.logger.warning(f"Handoff summary generation failed: {e}")
                return None
        return None

    async def _read_progress(self) -> str | None:
        """Tail of the model's append-only durable notes: an overgrown file
        needs its latest state, not its opening plan."""
        try:
            result = await self._environment.exec(
                f"tail -c {PROGRESS_REINJECT_MAX_BYTES} {prompts.PROGRESS_FILE} 2>/dev/null",
                timeout_sec=30,
            )
        except Exception as e:
            self.logger.debug(f"PROGRESS.md re-injection read failed: {e}")
            return None
        return str(result.stdout or "").strip() or None

    def _drop_middle(self) -> None:
        """Halve the history between the head (system prompt + task message)
        and the recent tail. Runs only on overflow, where the cached prefix
        is already lost."""
        messages = self._run.messages
        if len(messages) <= 4:
            return
        tail = messages[(len(messages) + 2) // 2 :]
        self._run.messages[:] = [
            messages[0],
            messages[1],
            {
                "role": "user",
                "content": "[Earlier context omitted to fit the context window.]",
            },
            *llm.drop_orphan_tool_results(tail),
        ]

    # ----- run setup helpers -----
    # Sandbox-side setup lives in run(), not setup(): each agent phase runs
    # under its own user (dir ownership must match the user the model's bash
    # runs as) and failures here are best-effort, never trial-fatal.

    async def _resolve_workdir(self, environment: BaseEnvironment) -> str:
        """Resolve the task working directory. A run without one can never
        submit, so failing here — at setup, clearly attributed — beats
        burning the budget on a run with no voluntary exit."""
        workdir = environment.task_env_config.workdir
        if workdir:
            return workdir
        result = await environment.exec("pwd", timeout_sec=10)
        workdir = str(result.stdout or "").strip()
        if not workdir:
            raise RuntimeError(
                "could not resolve the task working directory: task.toml "
                "declares no [environment].workdir and `pwd` in the sandbox "
                "returned nothing"
            )
        return workdir

    async def _ensure_progress_dir(self, environment: BaseEnvironment) -> None:
        """Pre-create the durable-notes directory so the model's first
        append to PROGRESS.md cannot fail on a missing parent."""
        try:
            await environment.exec(
                f"mkdir -p {prompts.PROGRESS_FILE.parent}", timeout_sec=30
            )
        except Exception as e:  # noqa: BLE001 - backends raise varied types
            self.logger.debug(
                f"Could not pre-create {prompts.PROGRESS_FILE.parent}: {e}"
            )

    async def _init_submissions(
        self, environment: BaseEnvironment, workdir: str
    ) -> None:
        """Bind the submission store and make sure the workspace is a git
        repository (initializing one when the task image ships none)."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._submissions = submissions.Submissions(
            self.logs_dir,
            environment,
            workdir=workdir,
            elapsed_sec=self._elapsed,
            logger=self.logger,
            git_excludes=self.config.workspace_git_excludes,
            submission_timeout_sec=self.config.submission_timeout_sec,
        )
        await self._submissions.ensure_git()

    async def _arm_timer(self, environment: BaseEnvironment) -> None:
        """Write the sandbox deadline file and re-anchor any task-side
        timers. With ``task_provides_timer`` the task's clock is the ONLY
        model-facing one — no deadline file, no prompt bullet, so the model
        hears a single timer story — but the re-anchor still runs: the task
        image's countdown measures the sandbox's hard timeout from container
        start, and only the reseed makes it mean "seconds until this run
        ends". A run without a time budget arms nothing."""
        remaining = self._remaining_budget_sec()
        if remaining is None:
            return
        if not self.config.task_provides_timer:
            await timer.write_deadline(environment, int(remaining), self.logger)
        # Seed with REMAINING, not the full budget: on any arm after time has
        # already passed, the full budget would over-promise.
        await timer.reanchor(environment, int(remaining), self.logger)

    def _seed_conversation(
        self,
        instruction: str,
        workdir: str,
        environment: BaseEnvironment,
    ) -> None:
        """Open the conversation: system prompt plus the initial task
        message, mirrored into the trajectory."""
        initial = self.initial_message(instruction, workdir, environment)
        self._run.messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": initial},
        ]
        self._add_step("system", self._system_prompt)
        self._add_step("user", initial)

    # ----- bookkeeping -----

    def _elapsed(self) -> float:
        return time.monotonic() - self._run.start_monotonic

    def _remaining_budget_sec(self) -> float | None:
        if not self.config.time_budget_sec:
            return None
        return max(0.0, self.config.time_budget_sec - self._elapsed())

    def _accumulate_usage(self, parsed: llm.ParsedResponse) -> None:
        run = self._run
        run.total_prompt_tokens += parsed.prompt_tokens
        run.total_completion_tokens += parsed.completion_tokens
        run.total_cached_tokens += parsed.cached_tokens
        run.total_cost_usd += parsed.cost_usd
        if parsed.fallback_attempted:
            run.fallback_attempted_turns += 1
        if parsed.fallback_used:
            run.fallback_turns += 1
            model = parsed.response_model or "unknown"
            run.fallback_models[model] = run.fallback_models.get(model, 0) + 1
        if parsed.prompt_tokens:
            self._run.last_context_tokens = (
                parsed.prompt_tokens + parsed.completion_tokens
            )

    def _populate_context(self, context: AgentContext) -> None:
        run = self._run
        context.n_input_tokens = run.total_prompt_tokens
        context.n_output_tokens = run.total_completion_tokens
        context.n_cache_tokens = run.total_cached_tokens
        context.cost_usd = run.total_cost_usd if run.total_cost_usd > 0 else None
        has_store = hasattr(self, "_submissions")
        context.metadata = {
            "n_turns": run.n_turns,
            "elapsed_sec": round(self._elapsed(), 1),
            "n_submissions": self._submissions.valid_hash_count() if has_store else 0,
            "submissions": self._submissions.summary() if has_store else [],
            # The commit grading checks out; None = score the workspace as-is.
            "submitted_commit": self._submissions.winner_hash() if has_store else None,
            "ended_by_submit": run.ended_by_submit,
            "env_lost": run.env_lost,  # cut short by a dead sandbox
            "stalled": run.stalled,  # ended by the empty-turn breaker
            "fallback_attempted_turns": run.fallback_attempted_turns,
            "fallback_turns": run.fallback_turns,
            "fallback_models": dict(run.fallback_models),
        }

    def _add_step(
        self, source: Literal["system", "user", "agent"], message: str, **kwargs: Any
    ) -> None:
        self._steps.append(
            trajectories.Step(
                step_id=len(self._steps) + 1,
                timestamp=datetime.now(UTC).isoformat(),
                source=source,
                message=message,
                **kwargs,
            )
        )

    def _add_agent_step(
        self,
        parsed: llm.ParsedResponse,
        tool_calls: list[trajectories.ToolCall],
        observation_results: list[trajectories.ObservationResult],
        images: list[str] | None = None,
    ) -> None:
        metrics = None
        if parsed.prompt_tokens or parsed.completion_tokens or parsed.reasoning_tokens:
            metrics = trajectories.Metrics(
                prompt_tokens=parsed.prompt_tokens or None,
                completion_tokens=parsed.completion_tokens or None,
                cached_tokens=parsed.cached_tokens or None,
                cost_usd=parsed.cost_usd or None,
                extra=(
                    {"thinking_tokens": parsed.reasoning_tokens}
                    if parsed.reasoning_tokens
                    else None
                ),
            )
        extra: dict[str, Any] = {
            "thinking": {
                "summary": parsed.reasoning_content,
                "tokens": parsed.reasoning_tokens,
                "display": (parsed.request_thinking or {}).get("display"),
            },
            "model_routing": {
                "requested_model": self.model_name,
                "response_model": parsed.response_model,
                "fallback_attempted": parsed.fallback_attempted,
                "fallback_used": parsed.fallback_used,
                "iterations": parsed.fallback_iterations,
                "stop_reason": parsed.stop_reason,
                "stop_details": parsed.stop_details,
                "request_thinking": parsed.request_thinking,
                "request_output_config": parsed.request_output_config,
                "request_fallbacks": parsed.request_fallbacks,
            },
        }
        if images:
            extra["images"] = images
        self._add_step(
            "agent",
            parsed.content or "",
            model_name=parsed.response_model or self.model_name,
            reasoning_effort=self.config.reasoning_effort,
            reasoning_content=parsed.reasoning_content or None,
            tool_calls=tool_calls or None,
            observation=(
                trajectories.Observation(results=observation_results)
                if observation_results
                else None
            ),
            metrics=metrics,
            llm_call_count=1,
            extra=extra,
        )

    def _dump_artifacts(self) -> None:
        run = self._run
        if self._steps:
            trajectory = trajectories.Trajectory(
                session_id=run.session_id or None,
                agent=trajectories.Agent(
                    name=self.name(),
                    version=self.version() or "unknown",
                    model_name=self.model_name,
                    tool_definitions=self._tool_schemas,
                ),
                steps=list(self._steps),
                final_metrics=trajectories.FinalMetrics(
                    total_prompt_tokens=run.total_prompt_tokens or None,
                    total_completion_tokens=run.total_completion_tokens or None,
                    total_cached_tokens=run.total_cached_tokens or None,
                    total_cost_usd=(
                        run.total_cost_usd if run.total_cost_usd > 0 else None
                    ),
                    total_steps=len(self._steps),
                ),
            )
            utils.write_artifact(
                self.logs_dir,
                "trajectory.json",
                format_trajectory_json(trajectory.to_json_dict()),
                self.logger,
            )
        utils.write_artifact(
            self.logs_dir,
            "messages.json",
            json.dumps(run.messages, indent=2, default=str),
            self.logger,
        )

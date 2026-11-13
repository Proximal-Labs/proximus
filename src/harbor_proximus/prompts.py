"""Every model-facing string Proximus injects"""

from pathlib import Path
from typing import Any

from harbor.models.task.config import NetworkMode

from harbor_proximus import utils
from harbor_proximus.timer import DEADLINE_PATH

SUMMARY_REQUEST = """Pause the task. Produce a comprehensive handoff summary of your work so far for another agent who will continue without your conversation history. Cover:
1. Major actions completed and their outcomes.
2. Key findings: important file paths, commands, configurations, error messages, measurements.
3. Files created or modified so far, and the role of each.
4. Current status: what is verified working, what is in progress.
5. Precise next steps, with rationale.
6. Pitfalls to avoid.
Be specific and factual. Do not call any tools."""

HANDOFF_HEADER = (
    "[Context compacted] Your earlier conversation history was summarized to "
    "free context space."
)
PROGRESS_FILE = Path("/tmp/proximus/PROGRESS.md")
HANDOFF_FOOTER = (
    f"Before acting: read {PROGRESS_FILE} and inspect the current "
    "state of the filesystem (e.g. `git log`, recently modified files). Do "
    "not redo completed work. Continue the task."
)

SUBMIT_TIME_LEFT = (
    "{remaining} of your {budget} time budget remains ({percent_left}% left)."
)

SUBMIT_CONTINUE_OR_CONFIRM = (
    "You can continue working and submit again later to record an improved "
    "result — a recorded submission is never lost to later work. If you are "
    "certain nothing more can be gained, call submit again as your ONLY "
    "action in your next message to confirm this as your final submission "
    "and end the run; any other tool call cancels the pending confirmation."
)

SUBMIT_CONFIRMED = "Final submission confirmed. Ending the session."

SUBMIT_FALLBACK = (
    "The final workspace state at the end of the run will be used as your "
    "result instead."
)

# Deliberately says NOTHING about multiple submissions or the two-submit ending:
# revealing those up front telegraphs that submissions can be revised, which we
# want the model to discover only when it submits (the continue-or-confirm nudge
# and the disarm note carry the full contract from that moment on).
SUBMIT_GUIDANCE = """\
- When your work is ready, record it with `submit`. A commit is optional: omit it and the harness commits the current workspace state for you, uncommitted changes included; or pass a commit that already exists in the workspace repository.
- Grading restores your latest submitted commit and then scores the workspace filesystem.
- Do not stop early. Keep improving while meaningful time remains."""

# Exactly one timer bullet renders: the deadline file _arm_timer writes,
# or the task's own clock when config.task_provides_timer is set.
DEADLINE_GUIDANCE = (
    f"- With a budget set, `{DEADLINE_PATH}` holds the Unix epoch second "
    f"the run ends: `echo $(( $(cat {DEADLINE_PATH}) - $(date +%s) ))` "
    "prints the whole seconds left at any time."
)

TASK_TIMER_GUIDANCE = (
    "- The task environment provides its own timer; check the remaining "
    "time the way the task instruction describes."
)

OUTPUT_LENGTH_MESSAGE = (
    "Your previous response exceeded the maximum output length and was "
    "discarded. Re-issue it in smaller pieces: shorter messages, smaller "
    "edits, fewer tool calls per turn."
)

# The provider's own error, verbatim, plus the one fact the model cannot
# infer: the harness evicted the newest attached image to make room.
REQUEST_TOO_LARGE_MESSAGE = (
    "Request failed: {error}\n"
    "The most recently attached image was removed from the conversation."
)

INVALID_IMAGE_MESSAGE = (
    "Request failed: {error}\n"
    "The provider rejected the most recently attached image, so its payload "
    "was removed from the conversation. Fix the source image according to "
    "the provider's stated requirements, then call `view_image` again."
)

# Content of an assistant turn with neither text nor tool calls, which
# several providers reject.
NO_OUTPUT_PLACEHOLDER = "(no output this turn)"

# Closes every text-only assistant turn (Anthropic rejects a conversation
# ending in an assistant message). Deliberately neutral: prose turns are
# legitimate work and must not be nagged.
CONTINUE_MESSAGE = "Continue."

# While a submission awaits confirmation a bare "Continue." reads as "do
# more work", so this one state names the confirm path.
CONTINUE_OR_CONFIRM_MESSAGE = (
    "Continue. If instead nothing more can be gained, call `submit` again "
    "now to confirm your recorded submission and end the run."
)

# Closes a turn that produced nothing at all. Unlike prose turns this one
# IS nagged: a bare "Continue." never breaks the pattern, and after
# EMPTY_TURN_STALL_LIMIT of these the run dies with no submission.
EMPTY_RESPONSE_MESSAGE = (
    "Your previous response was empty. Reply with a tool call, or a short "
    "note on what you are doing next. If you consider the task complete, "
    "call `submit`."
)


def system_prompt(task_provides_timer: bool = False) -> str:
    """Render the system template with submit, timer, and durable-notes
    guidance."""
    template = utils.read_template("system.md")
    timer_line = TASK_TIMER_GUIDANCE if task_provides_timer else DEADLINE_GUIDANCE
    return (
        template.replace("{{SUBMIT_GUIDANCE}}", SUBMIT_GUIDANCE)
        .replace("{{TIMER_GUIDANCE}}", timer_line)
        .replace("{{ARTIFACT_GUIDANCE}}", "")
        .replace("{{PROGRESS_FILE}}", str(PROGRESS_FILE))
    )


def initial_user_message(
    instruction: str,
    workdir: str,
    budget_sec: float | None,
    network_mode: NetworkMode,
) -> str:
    message = (
        f"Task instruction:\n\n{instruction}"
        f"\n\nThe working directory for this task is {workdir} "
        "(bash commands run there by default). Do your work there "
        "unless the task says otherwise."
    )
    if budget_sec:
        message += (
            f"\n\nYou have a total time budget of "
            f"{fmt_duration(budget_sec)} for this task."
        )
    note = _network_note(network_mode)
    if note:
        message += f"\n\n{note}"
    return message


def _network_note(mode: NetworkMode) -> str | None:
    """Restricted network access described up front, so the model does not
    waste turns on installs that cannot succeed; public access gets None."""
    if mode == NetworkMode.NO_NETWORK:
        return (
            "This sandbox has no internet access. Do not attempt to "
            "install packages or download anything; work with what is "
            "already present."
        )
    if mode == NetworkMode.ALLOWLIST:
        return (
            "Internet access is restricted to a small allowlist; assume "
            "package installs and downloads will fail."
        )
    return None


def handoff_message(
    instruction: str,
    summary: str | None,
    progress: str | None,
    elapsed: float,
    budget: float | None,
) -> str:
    """The compaction boundary message: header, original instruction, model
    summary, durable notes, remaining budget, footer."""
    parts = [
        HANDOFF_HEADER,
        f"Original task instruction:\n\n{instruction}",
        "Handoff summary of the work completed so far:\n\n"
        + (
            summary
            or f"(summary unavailable — rely on {PROGRESS_FILE} "
            "and the state of the filesystem)"
        ),
    ]
    if progress:
        parts.append(
            f"Your durable notes, re-read from {PROGRESS_FILE} at the "
            f"compaction boundary:\n\n{progress}"
        )
    if budget:
        remaining = max(0.0, budget - elapsed)
        parts.append(
            f"Time budget: about {fmt_duration(remaining)} of "
            f"{fmt_duration(budget)} remain "
            f"({round(100 * remaining / budget)}% left)."
        )
    parts.append(HANDOFF_FOOTER)
    return "\n\n".join(parts)


def submit_guidance(elapsed: float, budget: float | None) -> str:
    """Continue-or-confirm guidance for a non-final submit, with the
    time-left sentence when a budget exists."""
    if not budget:
        return SUBMIT_CONTINUE_OR_CONFIRM
    remaining = max(0.0, budget - elapsed)
    time_left = SUBMIT_TIME_LEFT.format(
        remaining=fmt_duration(remaining),
        budget=fmt_duration(budget),
        percent_left=round(100 * remaining / budget),
    )
    return f"{time_left} {SUBMIT_CONTINUE_OR_CONFIRM}"


def submission_note(n: int, commit: str, subject: str) -> str:
    subject_clause = f' ("{subject}")' if subject else ""
    return f"Submission #{n} recorded from commit {commit[:12]}{subject_clause}."


def submission_failed(detail: str, prior_submissions: int = 0) -> str:
    """With an earlier recorded submission the winner is that commit, not
    the final workspace state — the fallback sentence must stay truthful."""
    if prior_submissions:
        return (
            f"Submission could not be recorded ({detail}). Your previous "
            f"submission (#{prior_submissions}) remains your recorded result."
        )
    return f"Submission could not be recorded ({detail}). {SUBMIT_FALLBACK}"


def submissions_list(rows: list[dict[str, Any]]) -> str:
    """One line per submit call, newest last, with the currently scored
    commit marked."""
    if not rows:
        return (
            "No submissions recorded yet. If you never submit, the workspace "
            "state at the end of the run is your result."
        )
    ok_indices = [r["index"] for r in rows if r.get("ok")]
    winner = ok_indices[-1] if ok_indices else None
    lines = []
    for row in rows:
        line = f"#{row['index']}  {(row.get('hash') or '')[:12] or '????'}"
        line += f"  at {fmt_duration(row.get('elapsed_sec') or 0)} into the run"
        if row.get("description"):
            line += f'  "{row["description"]}"'
        if not row.get("ok"):
            line += "  (failed; not recorded)"
        if row["index"] == winner:
            line += "  <- current result"
        lines.append(line)
    lines.append(
        "The most recent successful submission is kept as your result."
        if winner is not None
        else "No successful submissions yet; the final workspace state "
        "would be scored instead."
    )
    return "\n".join(lines)


def fmt_duration(secs: float) -> str:
    """Render a non-negative duration using at most two abbreviated units
    (e.g. ``2h 30m``, ``45m 10s``, ``50s``)."""
    rounded = int(max(0.0, secs) + 0.5)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        units = ((hours, "h"), (minutes, "m"))
    elif minutes:
        units = ((minutes, "m"), (seconds, "s"))
    else:
        units = ((seconds, "s"),)
    units = tuple(u for u in units if u[0]) or ((0, "s"),)
    return " ".join(f"{value}{suffix}" for value, suffix in units)

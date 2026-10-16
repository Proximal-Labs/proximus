import logging

from harbor.environments.base import BaseEnvironment

from harbor_proximus import utils

_TIMER_MISSING = "missing"
DEADLINE_PATH = "/tmp/proximus/deadline"


async def write_deadline(
    environment: BaseEnvironment, remaining_sec: int, logger: logging.Logger
) -> None:
    """Write the run's end time (a Unix epoch second) to ``DEADLINE_PATH``,
    anchored to the sandbox's own clock — the model compares against
    in-sandbox ``date``."""
    if remaining_sec <= 0:
        logger.debug("No budget left at setup; skipping the deadline file.")
        return
    command = (
        "mkdir -p /tmp/proximus && "
        f"echo $(( $(date +%s) + {int(remaining_sec)} )) > {DEADLINE_PATH}"
    )
    try:
        result = await environment.exec(command, timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(f"exit {result.return_code}: {result.stderr}")
    except Exception as e:
        logger.warning(
            f"Deadline file not written; {DEADLINE_PATH} will be absent "
            f"and the model has only the budget from the initial message: {e}"
        )


async def reanchor(
    environment: BaseEnvironment, remaining_sec: int, logger: logging.Logger
) -> None:
    """Re-anchor the task image's own timer (sandbox-timer's boot anchor, or
    a restarted legacy ``/app/timer.sh``) seeded with ``remaining_sec`` so
    its countdown means "seconds until this run ends" rather than the hard
    timeout from container start."""
    script = utils.read_template("reanchor_timer.sh")
    command = f"remaining={remaining_sec}; missing={_TIMER_MISSING};\n{script}"
    try:
        result = await environment.exec(command, timeout_sec=60)
    except Exception as e:
        logger.debug(f"Task timer re-anchor failed: {e}")
        return
    lines = str(result.stdout or "").split()
    remaining = lines[-1] if lines else ""
    if remaining == _TIMER_MISSING:
        logger.debug("Task ships no timer script; nothing to re-anchor.")
        return
    if not remaining.isdigit():
        logger.warning(
            f"Task timer re-anchor produced no countdown "
            f"(rc={result.return_code}, stdout={remaining!r})"
        )
        return
    logger.info(
        f"Task timer re-anchored: seeded_remaining={remaining_sec} "
        f"timer_reports={remaining}"
    )

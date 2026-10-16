"""Shared helpers for the proximus package."""

import logging
from pathlib import Path

_PACKAGE_DIR = Path(__file__).parent


def read_template(name: str) -> str:
    """A prompt template from ``templates/``."""
    return (_PACKAGE_DIR / "templates" / name).read_text()


def read_script(name: str) -> str:
    """A shell script from ``scripts/``, run in the sandbox as is."""
    return (_PACKAGE_DIR / "scripts" / name).read_text()


def write_artifact(
    logs_dir: Path,
    name: str,
    content: str,
    logger: logging.Logger,
    *,
    level: int = logging.DEBUG,
) -> None:
    """Best-effort: a failed log write must never kill the run."""
    try:
        (logs_dir / name).write_text(content)
    except OSError as e:
        logger.log(level, f"Failed to write {name}: {e}")

"""Shared helpers for the proximus package."""

import logging
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text()


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

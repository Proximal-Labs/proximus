import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment

from harbor_proximus import tools, utils

GIT_TIMEOUT_SEC = 60
DEFAULT_SUBMISSION_TIMEOUT_SEC = 600
RESTORE_TIMEOUT_SEC = 300
_FULL_COMMIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


@dataclass
class Submission:
    """A recorded submission."""

    hash: str
    description: str
    duration_sec: float
    # True when the harness itself made the commit (a commitless submit on
    # a dirty tree); mirrored to the record row as ``harness_commit``.
    harness_committed: bool = False


class SubmissionError(Exception):
    """A submit call that could not be recorded."""


class Submissions:
    """Prepares the workspace repository, pins submitted commits, and
    records one ``submissions.jsonl`` row per submit call.

    All harness git runs as root. Empty/HEAD submit snapshots the live
    workspace so a model that cannot invoke git still has a pin; when
    that snapshot makes a commit, the row is flagged ``harness_commit``."""

    def __init__(
        self,
        logs_dir: Path,
        environment: BaseEnvironment,
        *,
        workdir: str,
        elapsed_sec: Callable[[], float],
        logger: logging.Logger,
        git_excludes: list[str] | None = None,
        submission_timeout_sec: int = DEFAULT_SUBMISSION_TIMEOUT_SEC,
    ):
        self._record_path = logs_dir / "submissions.jsonl"
        self._environment = environment
        self._workdir = workdir
        self._elapsed_sec = elapsed_sec
        self._logger = logger
        self._git_excludes = list(git_excludes or [])
        self._submission_timeout_sec = int(submission_timeout_sec)
        self._submissions: list[dict[str, Any]] = []

    def _git_exec(self, script: str, env: dict[str, str], timeout_sec: int):
        """Run a harness git script as root, not as the agent user.

        Root git refuses the agent-owned workspace repo ("dubious ownership")
        unless safe.directory covers it. Injected as GIT_CONFIG_* env — every
        git subcommand in every script inherits it, on any task image, without
        mutating the image's git config. Without it, every submit fails on
        an image that lacks the setting, and the trial is left ungradable."""
        return self._environment.exec(
            script,
            env={
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "safe.directory",
                "GIT_CONFIG_VALUE_0": self._workdir,
                **env,
            },
            timeout_sec=timeout_sec,
            user="root",
        )

    async def ensure_git(self) -> None:
        """Make the workspace a git repository."""
        result = await self._git_exec(
            utils.read_template("ensure_git.sh"),
            {
                "wd": self._workdir,
                # Newline-joined: gitwildmatch patterns cannot contain
                # newlines, so the join is unambiguous for the script.
                "exclude_patterns": "\n".join(self._git_excludes),
            },
            GIT_TIMEOUT_SEC,
        )
        if result.return_code != 0:
            detail = tools.format_exec_output(result.stdout, result.stderr)
            raise RuntimeError(
                f"workspace cannot host submissions: "
                f"{tools.truncate_middle(detail, 300)}"
            )

    async def submit(self, ref: str, *, auto: bool = False) -> Submission:
        """Pin ``ref`` as the next submission and record the row. Raises
        ``SubmissionError`` after recording the failure; a dead sandbox
        propagates as ``EnvironmentGoneError`` before anything is recorded.
        ``auto`` marks the harness's end-of-run submission so analysis can
        tell it from ones the model chose to make."""
        started = time.monotonic()
        try:
            hash_, description, harness_committed = await self._resolve_and_pin(
                self.success_count() + 1, ref
            )
        except SubmissionError as e:
            failure: dict[str, Any] = {
                "event": "submission",
                "index": len(self._submissions) + 1,
                "ok": False,
                "hash": "",
                "record_sec": self._since(started),
                "elapsed_sec": round(self._elapsed_sec(), 1),
                "error": str(e),
            }
            if auto:
                failure["auto"] = True
            self._append(failure)
            raise
        submission = Submission(
            hash_, description, self._since(started), harness_committed
        )
        entry: dict[str, Any] = {
            "event": "submission",
            "index": len(self._submissions) + 1,
            "ok": True,
            "hash": submission.hash,
            "record_sec": submission.duration_sec,
            "elapsed_sec": round(self._elapsed_sec(), 1),
        }
        if auto:
            entry["auto"] = True
        if submission.harness_committed:
            # The harness committed the live workspace for this pin (the
            # model gave no commit); flagged like ``auto`` so analysis can
            # tell harness-made commits from ones the model created.
            entry["harness_commit"] = True
        if submission.description:
            entry["description"] = submission.description
        self._append(entry)
        return submission

    async def matches_winner(self, ref: str, winner: str) -> bool:
        """True when a confirming submit names the current ``winner``: a
        bare/HEAD ref must match the live tree (the agent often cannot
        commit, so HEAD may be stale); an explicit ref must resolve to it.
        A dead sandbox propagates; any other failure reads as "does not
        match" so confirm records a fresh snapshot."""
        try:
            result = await self._git_exec(
                utils.read_template("matches_winner.sh"),
                {"wd": self._workdir, "winner": winner, "commit": ref},
                self._submission_timeout_sec,
            )
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:
            tools.raise_if_environment_gone(e)
            return False
        return result.return_code == 0

    async def _resolve_and_pin(self, index: int, ref: str) -> tuple[str, str, bool]:
        """Run ``record_commit.sh``; returns (hash, subject,
        harness_committed). Raises ``SubmissionError`` with model-readable
        guidance."""
        try:
            result = await self._git_exec(
                utils.read_template("record_commit.sh"),
                {"wd": self._workdir, "commit": ref, "idx": str(int(index))},
                self._submission_timeout_sec,
            )
        except TimeoutError as e:
            # Modal's SandboxTimeoutError subclasses TimeoutError but means
            # the sandbox died — classify before recording a failure.
            tools.raise_if_environment_gone(e)
            raise SubmissionError(
                f"recording timed out after {self._submission_timeout_sec:g}s"
            ) from None
        except tools.EnvironmentGoneError:
            raise
        except Exception as e:
            tools.raise_if_environment_gone(e)
            raise SubmissionError(f"recording failed: {e}") from e
        if result.return_code != 0:
            detail = tools.format_exec_output(result.stdout, result.stderr)
            raise SubmissionError(tools.truncate_middle(detail, 500))
        return self._parse_record_output(str(result.stdout or ""))

    @staticmethod
    def _parse_record_output(stdout: str) -> tuple[str, str, bool]:
        """Lenient on purpose: a zero exit already means the ref was pinned,
        so a malformed line degrades to empty fields, never to a failure."""
        hash_ = ""
        description = ""
        harness_committed = False
        for line in stdout.strip().splitlines():
            if line.startswith("subject="):
                description = line.removeprefix("subject=").strip()
            elif line.startswith("hash="):
                hash_ = line.removeprefix("hash=").strip()
            elif line.startswith("committed="):
                harness_committed = line.removeprefix("committed=").strip() == "1"
        return hash_, description, harness_committed

    @staticmethod
    def _since(started: float) -> float:
        return round(time.monotonic() - started, 3)

    def _append(self, entry: dict[str, Any]) -> None:
        """In memory always, to the JSONL best-effort."""
        self._submissions.append(entry)
        try:
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
            with self._record_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), **entry}) + "\n")
        except Exception as e:
            self._logger.warning(f"Could not write submissions.jsonl: {e}")

    def success_count(self) -> int:
        return sum(1 for e in self._submissions if e["ok"])

    def valid_hash_count(self) -> int:
        """Successful rows that carry a complete Git object id."""
        return sum(
            1
            for entry in self._submissions
            if entry.get("ok")
            and isinstance(entry.get("hash"), str)
            and _FULL_COMMIT_OID.fullmatch(entry["hash"]) is not None
        )

    def summary(self) -> list[dict[str, Any]]:
        """Per-submission rows for run metadata and ``list_submissions``."""
        return [
            {k: v for k, v in entry.items() if k != "event"}
            for entry in self._submissions
        ]

    def winner_hash(self) -> str | None:
        """Newest successful row with a complete Git object id."""
        winners = [
            entry["hash"]
            for entry in self._submissions
            if entry.get("ok")
            and isinstance(entry.get("hash"), str)
            and _FULL_COMMIT_OID.fullmatch(entry["hash"]) is not None
        ]
        return winners[-1] if winners else None

    async def restore_winner(self) -> bool:
        """Restore the winning commit; True once the workspace holds it."""
        winner = self.winner_hash()
        if not winner:
            self._logger.warning("Winner restore skipped: no valid submission exists.")
            return False
        try:
            result = await self._git_exec(
                utils.read_template("restore_winner.sh"),
                {"wd": self._workdir, "winner": winner},
                RESTORE_TIMEOUT_SEC,
            )
        except Exception as e:
            self._logger.warning(f"Winner restore failed: {e}")
            return False
        if result.return_code != 0:
            detail = tools.format_exec_output(result.stdout, result.stderr)
            self._logger.warning(
                f"Winner restore failed: {tools.truncate_middle(detail, 300)}"
            )
            return False
        self._logger.info(f"Restored winning submission {winner[:12]}.")
        return True

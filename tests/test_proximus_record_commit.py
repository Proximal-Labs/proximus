"""The record script against a REAL repository (every other suite fakes the
git exec): a commitless submit must commit the live workspace and pin THAT
commit — never silently bind to a stale HEAD that misses uncommitted work.
A clean workspace reuses HEAD; an explicit ref is pinned as-is and leaves
the tree untouched."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult

from harbor_proximus.submissions import SubmissionError, Submissions

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        shutil.which("git") is None or shutil.which("bash") is None,
        reason="requires git and bash on PATH",
    ),
]

logger = logging.getLogger(__name__)


def _base_env(home: Path) -> dict[str, str]:
    """Hermetic git: no user/system config (hooks, signing) may leak in."""
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "GIT_CONFIG_NOSYSTEM": "1",
    }


class LocalGitEnv:
    """Duck-typed environment that runs harness scripts FOR REAL with bash,
    the way a sandbox would: inputs as exec env vars, nothing spliced."""

    def __init__(self, base_env: dict[str, str]):
        self._base_env = base_env

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            env={**self._base_env, **(env or {})},
            cwd=cwd,
            timeout=timeout_sec,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def git(repo: Path, home: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        capture_output=True,
        text=True,
        check=True,
        env=_base_env(home),
    )
    return result.stdout.strip()


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Submissions]:
    """A real repo with a baseline commit, plus a store wired to run the
    scripts against it. Returns (repo, home, submissions)."""
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, home, "init", "-q")
    git(repo, home, "config", "user.email", "test@localhost")
    git(repo, home, "config", "user.name", "Test")
    (repo / "main.py").write_text("print('v1')\n")
    git(repo, home, "add", "-A")
    git(repo, home, "commit", "-qm", "baseline", "--no-verify")
    subs = Submissions(
        tmp_path / "logs",
        LocalGitEnv(_base_env(home)),  # type: ignore[arg-type]
        workdir=str(repo),
        elapsed_sec=lambda: 0.0,
        logger=logger,
    )
    return repo, home, subs


async def test_commitless_submit_on_dirty_workspace_commits_the_live_tree(workspace):
    repo, home, subs = workspace
    baseline = git(repo, home, "rev-parse", "HEAD")
    (repo / "main.py").write_text("print('v2')\n")  # modified, uncommitted
    (repo / "extra.py").write_text("print('new')\n")  # untracked

    submission = await subs.submit("")

    # A NEW commit, never the stale HEAD, and it is what got pinned.
    assert submission.hash != baseline
    assert len(submission.hash) == 40
    assert submission.harness_committed is True
    assert submission.description == "Proximus submission snapshot"
    pinned = git(repo, home, "rev-parse", "refs/proximus/submissions/1")
    assert pinned == submission.hash
    # The live work — modified AND untracked — is inside that commit.
    tree = git(repo, home, "ls-tree", "-r", "--name-only", submission.hash)
    assert "extra.py" in tree.splitlines()
    assert "print('v2')" in git(repo, home, "show", f"{submission.hash}:main.py")
    assert git(repo, home, "status", "--porcelain") == ""
    # The row says the harness made the commit (like the ``auto`` flag).
    row = json.loads((subs._record_path).read_text())
    assert row["ok"] is True
    assert row["hash"] == submission.hash
    assert row["harness_commit"] is True


async def test_commitless_submit_on_clean_workspace_reuses_head(workspace):
    repo, home, subs = workspace
    baseline = git(repo, home, "rev-parse", "HEAD")

    submission = await subs.submit("")

    assert submission.hash == baseline
    assert submission.harness_committed is False
    assert git(repo, home, "rev-parse", "refs/proximus/submissions/1") == baseline
    row = json.loads((subs._record_path).read_text())
    assert "harness_commit" not in row


async def test_explicit_commit_submit_pins_the_named_commit_unchanged(workspace):
    repo, home, subs = workspace
    baseline = git(repo, home, "rev-parse", "HEAD")
    (repo / "main.py").write_text("print('wip, not part of the submission')\n")

    submission = await subs.submit(baseline)

    # Pinned exactly as named; the harness committed nothing and the
    # uncommitted work is still live in the tree.
    assert submission.hash == baseline
    assert submission.harness_committed is False
    assert git(repo, home, "status", "--porcelain") != ""
    row = json.loads((subs._record_path).read_text())
    assert "harness_commit" not in row


async def test_explicit_unknown_ref_fails_with_guidance(workspace):
    repo, home, subs = workspace
    with pytest.raises(SubmissionError) as excinfo:
        await subs.submit("deadbeef")
    assert "not a commit in the workspace repository" in str(excinfo.value)
    assert git(repo, home, "status", "--porcelain") == ""  # nothing touched


async def test_auto_submit_on_dirty_tree_flags_auto_and_harness_commit(workspace):
    # The end-of-run auto-submission goes through the same argless path;
    # when the tree moved, the row carries BOTH flags.
    repo, home, subs = workspace
    (repo / "late.py").write_text("print('kill-time work')\n")

    await subs.submit("", auto=True)

    row = json.loads((subs._record_path).read_text())
    assert row["auto"] is True
    assert row["harness_commit"] is True


async def test_matches_winner_tracks_the_live_tree(workspace):
    # Dedup semantics: a bare ref matches the winner only while the live
    # tree still equals it — new work must read as "does not match" so a
    # confirm (or the exit auto-submission) records a fresh snapshot.
    repo, home, subs = workspace
    submission = await subs.submit("")
    assert await subs.matches_winner("", submission.hash) is True
    (repo / "more.py").write_text("print('newer work')\n")
    assert await subs.matches_winner("", submission.hash) is False

"""Unit tests for submissions: a submission IS a git commit. ``submit``
resolves the given ref (default HEAD) to a full SHA and pins it under
refs/proximus/submissions/; setup ensures a repository exists; the read-only
``list_submissions`` tool shows the store. Nothing is copied at submit time —
durability is whatever end-of-run workspace capture the runner does, which
carries the workspace ``.git``; at run end the winner is checked out so the
verifier and that capture see the submitted tree."""

import json
import logging

import litellm
import pytest
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from tests.test_proximus_loop import (
    FakeCompletion,
    FakeEnv,
    FakeMessage,
    FakeResponse,
    FakeToolCall,
    bash_response,
    make_agent,
    read_messages,
)

pytestmark = pytest.mark.unit

HEAD_HASH = "a" * 40
FIX2_HASH = "b" * 40


def git_submit_response(call_id: str = "call_g", **arguments) -> FakeResponse:
    return FakeResponse(
        FakeMessage(
            content="Submitting a commit.",
            tool_calls=[FakeToolCall(call_id, "submit", json.dumps(arguments))],
        )
    )


def list_response(call_id: str = "call_l") -> FakeResponse:
    return FakeResponse(
        FakeMessage(
            content="Reviewing my submissions.",
            tool_calls=[FakeToolCall(call_id, "list_submissions", "{}")],
        )
    )


class GitEnv(FakeEnv):
    """FakeEnv that answers the git steps: repo setup reports ok, and the
    record script resolves HEAD / fix2 to fixed hashes (badref fails with
    the script's guidance, exit 6). Script inputs arrive as exec env vars."""

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.all_users.append(user)
        ref = (env or {}).get("commit", "")
        # The confirm path's winner match (matches_winner.sh): a bare/HEAD
        # confirm matches a clean tree; an explicit ref matches when it
        # names the winner. Checked first: it also contains
        # `rev-parse -q --verify`.
        if 'echo "matched"' in command:
            self.all_commands.append(command)
            self.all_envs.append(dict(env or {}))
            if ref in ("", "HEAD", "head"):
                return ExecResult(stdout="matched", stderr=None, return_code=0)
            full = FIX2_HASH if ref == "fix2" else HEAD_HASH
            if ref != "badref" and full == (env or {}).get("winner"):
                return ExecResult(stdout="matched", stderr=None, return_code=0)
            return ExecResult(stdout=None, stderr=None, return_code=1)
        # `rev-parse -q --verify` + update-ref: the record script.
        if "rev-parse -q --verify" in command:
            self.all_commands.append(command)
            self.all_envs.append(dict(env or {}))
            if ref == "badref":
                return ExecResult(
                    stdout=None,
                    stderr=(
                        "'badref' is not a commit in the workspace repository. "
                        "Commit your work first (git add -A && git commit)..."
                    ),
                    return_code=6,
                )
            name = "fix2" if ref == "fix2" else "HEAD"
            full = FIX2_HASH if name == "fix2" else HEAD_HASH
            return ExecResult(
                stdout=f"subject=wip solution {name}\nhash={full}",
                stderr=None,
                return_code=0,
            )
        if "is-inside-work-tree" in command:  # ensure_git.sh
            self.all_commands.append(command)
            self.all_envs.append(dict(env or {}))
            return ExecResult(stdout="ok", stderr=None, return_code=0)
        return await super().exec(
            command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
        )


def finish_responses() -> list[FakeResponse]:
    return [git_submit_response(call_id=f"call_{n}") for n in range(2)]


# ----- surface -----


def test_surface(tmp_path):
    agent = make_agent(tmp_path)
    names = [s["function"]["name"] for s in agent._tool_schemas]
    assert "list_submissions" in names
    submit = next(s for s in agent._tool_schemas if s["function"]["name"] == "submit")
    assert "commit" in submit["function"]["parameters"]["properties"]
    assert submit["function"]["parameters"].get("required", []) == []
    # The prompt teaches submit-as-snapshot, neutrally.
    assert "record it with `submit`" in agent._system_prompt
    assert "multiple times" not in agent._system_prompt


# ----- the submit flow -----


async def test_submit_records_pins_and_confirms(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [git_submit_response(), git_submit_response(call_id="call_g2")]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = GitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    # Setup ensured the repository before the first turn.
    assert any("is-inside-work-tree" in c for c in env.all_commands)

    # Bare submit snapshots the live workspace (empty/HEAD), and the record
    # snippet pins the commit so a later rebase/reset or git gc cannot make
    # it unreachable before the end-of-run workspace capture.
    records = [
        (c, e)
        for c, e in zip(env.all_commands, env.all_envs)
        if 'git update-ref "refs/proximus/submissions/$idx"' in c
    ]
    assert len(records) == 1  # the confirm REUSED the record; nothing new
    assert records[0][1]["commit"] == ""
    assert records[0][1]["idx"] == "1"
    assert all(u == "root" for u in env.all_users if u is not None)

    # The response names the commit + subject and keeps the shared
    # continue-or-confirm flow; the second bare submit confirms by matching
    # the live tree to the pinned winner and reusing the armed record.
    assert any('echo "matched"' in c for c in env.all_commands)
    tools = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    assert f"recorded from commit {HEAD_HASH[:12]}" in tools[0]
    assert '"wip solution HEAD"' in tools[0]
    assert "confirm this as your final submission" in tools[0]
    assert "Final submission confirmed" in tools[1]

    # One command runs after the loop: the end-of-run restore checks out
    # the winner so the verifier and the workspace capture see the submitted tree.
    # It pins the pre-restore HEAD (end-state) and never resets hard.
    restore = env.all_commands[-1]
    assert "refs/proximus/end-state" in restore
    assert env.all_envs[-1]["winner"] == HEAD_HASH
    assert not any("reset --hard" in c for c in env.all_commands)

    # Store rows carry the commit subject; the winner is surfaced for
    # grading as submitted_commit.
    assert context.metadata is not None
    rows = context.metadata["submissions"]
    assert [r.get("description") for r in rows] == ["wip solution HEAD"]
    assert rows[0]["hash"] == HEAD_HASH
    assert context.metadata["submitted_commit"] == HEAD_HASH


async def test_explicit_commit_ref_is_submitted(tmp_path, monkeypatch):
    fake = FakeCompletion(
        [git_submit_response(commit="fix2"), git_submit_response(commit="fix2")]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = GitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    records = [
        e
        for c, e in zip(env.all_commands, env.all_envs)
        # matches_winner.sh also contains rev-parse; keep only record_commit.sh
        # calls (the exit auto-submission probes the winner with commit="").
        if "rev-parse -q --verify" in c and 'echo "matched"' not in c
    ]
    assert all(e["commit"] == "fix2" for e in records)
    tools = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    assert f"recorded from commit {FIX2_HASH[:12]}" in tools[0]
    assert context.metadata is not None
    assert context.metadata["submitted_commit"] == FIX2_HASH


async def test_confirm_with_a_new_commit_records_it(tmp_path, monkeypatch):
    # A second submit naming a DIFFERENT commit is new work, not the
    # protocol's "yes" — it must record (and end the run with it as winner).
    fake = FakeCompletion(
        [
            git_submit_response(),
            git_submit_response(commit="fix2", call_id="call_g2"),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = GitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    tools = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    assert f"recorded from commit {FIX2_HASH[:12]}" in tools[1]
    assert "Final submission confirmed" in tools[1]
    assert context.metadata is not None
    assert context.metadata["n_submissions"] == 2
    assert context.metadata["submitted_commit"] == FIX2_HASH


async def test_unknown_commit_fails_readably_and_keeps_prior_winner(
    tmp_path, monkeypatch
):
    # Bad ref first (arms but records nothing), work resets the gate, then a
    # good submission is recorded and confirmed.
    fake = FakeCompletion(
        [
            git_submit_response(commit="badref"),
            bash_response("git log --oneline"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = GitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    tools = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    assert "not a commit in the workspace repository" in tools[0]
    # No prior submission existed, so the fallback message stays truthful.
    assert "final workspace state" in tools[0].lower()
    assert context.metadata is not None
    # badref failed, HEAD recorded once; the confirm reused it.
    assert context.metadata["n_submissions"] == 1
    assert [r["ok"] for r in context.metadata["submissions"]] == [False, True]
    assert context.metadata["submitted_commit"] == HEAD_HASH


async def test_failed_record_after_a_good_one_names_the_prior_winner(
    tmp_path, monkeypatch
):
    # good submit -> bash -> bad submit: the failure message must say the
    # previous submission remains the result, not "final workspace state".
    fake = FakeCompletion(
        [
            git_submit_response(),
            bash_response("echo tinkering"),
            git_submit_response(commit="badref"),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = GitEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    tools = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    bad = next(c for c in tools if "not a commit" in c)
    assert "previous submission (#1) remains your recorded result" in bad
    assert "final workspace state" not in bad.lower()


async def test_malformed_commit_argument_neither_arms_nor_confirms(
    tmp_path, monkeypatch
):
    fake = FakeCompletion(
        [
            git_submit_response(),  # arms
            git_submit_response(commit=7),  # malformed: must not confirm
            git_submit_response(call_id="call_z"),  # confirms the pending one
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = GitEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    tools = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    assert "Invalid arguments for 'submit'" in tools[1]
    assert "Final submission confirmed" not in tools[1]
    assert "Final submission confirmed" in tools[2]
    # The malformed call recorded nothing: exactly ONE real record (the arming
    # submit) — the confirm reused it, and the exit auto-submission was skipped
    # because the tree matched the winner. (The old ==2 also counted the
    # confirm's matches_winner probe, which shares the rev-parse needle.)
    assert (
        sum(
            "rev-parse -q --verify" in c and 'echo "matched"' not in c
            for c in env.all_commands
        )
        == 1
    )


# ----- list_submissions -----


async def test_list_submissions_shows_store_rows_and_winner(tmp_path, monkeypatch):
    # bash between the submits: two consecutive submits would confirm-and-end
    # (the gate), and the listing itself also resets the gate.
    fake = FakeCompletion(
        [
            git_submit_response(),
            bash_response("echo improving"),
            git_submit_response(commit="fix2", call_id="call_g2"),
            list_response(),
            *finish_responses(),
        ]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600)
    env = GitEnv()
    await agent.run("Task", env, AgentContext())  # type: ignore[arg-type]

    tools = [m["content"] for m in read_messages(agent) if m.get("role") == "tool"]
    listing = next(c for c in tools if "current result" in c)
    assert f"#1  {HEAD_HASH[:12]}" in listing
    assert f"#2  {FIX2_HASH[:12]}" in listing
    assert '"wip solution fix2"' in listing
    # The winner at listing time is the newest successful submission.
    assert listing.index("<- current result") > listing.index(f"#2  {FIX2_HASH[:12]}")
    assert "most recent successful submission" in listing


# ----- the end-of-run restore -----


def _make_submissions(tmp_path, env):
    import logging

    from harbor_proximus.submissions import Submissions

    return Submissions(
        tmp_path,
        env,
        workdir="/app",
        elapsed_sec=lambda: 0.0,
        logger=logging.getLogger("test"),
    )


async def test_restore_winner_checks_out_newest_submission(tmp_path):
    env = GitEnv()
    subs = _make_submissions(tmp_path, env)
    await subs.submit("HEAD")

    await subs.restore_winner()

    restore = env.all_commands[-1]
    assert env.all_envs[-1]["winner"] == HEAD_HASH
    assert "refs/proximus/end-state" in restore
    assert "checkout -q -f" in restore


async def test_restore_winner_is_a_noop_without_a_submission(tmp_path):
    env = GitEnv()
    subs = _make_submissions(tmp_path, env)

    await subs.restore_winner()

    # No submission: nothing to restore, and no command reaches the sandbox
    # (all_commands stays empty — not even a cd).
    assert env.all_commands == []


async def test_restore_on_exit_false_keeps_kill_time_state(tmp_path, monkeypatch):
    # Offline-grading mode: pinning and telemetry are unchanged, but no
    # restore command reaches the sandbox — the workspace capture carries the
    # kill-time workspace, and grading goes through the pinned refs.
    fake = FakeCompletion(finish_responses())
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path, time_budget_sec=600, restore_on_exit=False)
    env = GitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    # The pin still happens at submit time.
    records = [c for c in env.all_commands if "rev-parse -q --verify" in c]
    assert 'git update-ref "refs/proximus/submissions/$idx"' in records[0]
    # But nothing touches the tree at run end.
    assert not any("refs/proximus/end-state" in c for c in env.all_commands)
    assert not any("checkout -q -f" in c for c in env.all_commands)
    # The grader's input is still surfaced.
    assert context.metadata is not None
    assert context.metadata["submitted_commit"] == HEAD_HASH


def test_restore_on_exit_defaults_to_true(tmp_path):
    assert make_agent(tmp_path).config.restore_on_exit is True


async def test_confirm_with_dirty_tree_records_only_the_model_snapshot(
    tmp_path, monkeypatch
):
    # Bare confirm after more uncommitted work records the second model state.
    # Finalization must not add a hidden third auto-submission.
    class DirtyGitEnv(GitEnv):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if 'echo "matched"' in command:
                self.all_users.append(user)
                self.all_commands.append(command)
                self.all_envs.append(dict(env or {}))
                return ExecResult(stdout=None, stderr=None, return_code=1)
            return await super().exec(
                command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
            )

    fake = FakeCompletion(
        [git_submit_response(), git_submit_response(call_id="call_g2")]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = DirtyGitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    records = [
        e
        for c, e in zip(env.all_commands, env.all_envs)
        if 'git update-ref "refs/proximus/submissions/$idx"' in c
    ]
    assert len(records) == 2
    assert context.metadata is not None
    assert context.metadata["n_submissions"] == 2
    assert all(not row.get("auto") for row in context.metadata["submissions"])


async def test_restore_winner_failure_never_raises(tmp_path, caplog):
    # Exit-path contract: a failed restore is logged, never raised — an
    # exception here would cost the end-of-run workspace capture.
    class FailingGitEnv(GitEnv):
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            if "refs/proximus/end-state" in command:
                self.all_commands.append(command)
                self.all_envs.append(dict(env or {}))
                return ExecResult(stdout=None, stderr="checkout failed", return_code=4)
            return await super().exec(
                command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user
            )

    env = FailingGitEnv()
    subs = _make_submissions(tmp_path, env)
    await subs.submit("HEAD")

    with caplog.at_level(logging.WARNING, logger="test"):
        await subs.restore_winner()

    assert "checkout failed" in caplog.text


# ----- end-of-run auto-submission -----


async def test_run_end_without_submit_auto_pins_final_state(tmp_path, monkeypatch):
    """A run that ends without the model ever submitting (here: a bash turn,
    then budget exhaustion) still leaves its last workspace state on the
    submissions rail — recorded via the exact argless submit() path, flagged
    ``auto`` in submissions.jsonl, winner for the exit restore."""
    from harbor_proximus.proximus import EMPTY_TURN_STALL_LIMIT
    from tests.test_proximus_loop import empty_response

    fake = FakeCompletion(
        [bash_response("echo work")]
        + [empty_response() for _ in range(EMPTY_TURN_STALL_LIMIT)]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)  # ends via the stall guard, never submitting
    env = GitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    assert context.metadata is not None
    subs = context.metadata["submissions"]
    assert [s["ok"] for s in subs] == [True]
    assert subs[0].get("auto") is True
    assert context.metadata["n_submissions"] == 1
    assert context.metadata["submitted_commit"] == HEAD_HASH
    rows = [
        json.loads(line)
        for line in (agent.logs_dir / "submissions.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["auto"] is True and rows[-1]["ok"] is True


async def test_clean_double_submit_end_gets_no_auto_submission(tmp_path, monkeypatch):
    """When the run ends via the confirm protocol and the tree still matches
    the winner, the exit path must NOT grow a duplicate submission."""
    fake = FakeCompletion(
        [git_submit_response(), git_submit_response(call_id="call_2")]
    )
    monkeypatch.setattr(litellm, "acompletion", fake)

    agent = make_agent(tmp_path)
    env = GitEnv()
    context = AgentContext()
    await agent.run("Task", env, context)  # type: ignore[arg-type]

    assert context.metadata is not None
    assert not any(s.get("auto") for s in context.metadata["submissions"])


async def test_ensure_git_threads_workspace_excludes(tmp_path):
    """The task's /app artifact excludes ride to ensure_git.sh as the
    exclude_patterns env var (newline-joined): grading quarantines those
    paths anyway, so tracking them would only bloat every submission."""
    import logging

    from harbor_proximus.submissions import Submissions

    env = GitEnv()
    store = Submissions(
        tmp_path,
        env,
        workdir="/app",
        elapsed_sec=lambda: 0.0,
        logger=logging.getLogger("test"),
        git_excludes=["**/.cache/**", "**/hf-cache/**"],
    )
    await store.ensure_git()
    ensure_env = env.all_envs[-1]
    assert ensure_env["exclude_patterns"] == "**/.cache/**\n**/hf-cache/**"


async def test_ensure_git_defaults_to_no_excludes(tmp_path):
    env = GitEnv()
    await _make_submissions(tmp_path, env).ensure_git()
    assert env.all_envs[-1]["exclude_patterns"] == ""

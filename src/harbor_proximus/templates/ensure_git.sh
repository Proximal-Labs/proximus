# Make the workspace a usable git repository for submissions (a submission
# IS a commit, so the model must be able to commit before its first turn).
# Idempotent: an existing repo is left alone apart from a local committer
# identity fallback (git commit refuses to run without one); a missing repo
# is initialized with the current state as a baseline commit, so HEAD exists
# before the model's first submit. A workspace that cannot host a repo (no
# git binary, missing dir) exits NONZERO — the run has no voluntary exit
# without submit, so the harness fails the trial at setup. On success the
# last stdout line is a status word for the debug log: ok / initialized.
#
# Inputs as exec env vars (submissions.py): wd — the workspace directory.
# Runs as root (see record_commit.sh): the agent user may not be able to
# invoke git at all.
command -v git >/dev/null 2>&1 || {
    echo "git is not installed in the task image" >&2
    exit 7
}
cd "$wd" || { echo "workspace directory missing: $wd" >&2; exit 3; }

status=ok
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git config user.email >/dev/null 2>&1 || git config user.email proximus@localhost
    git config user.name >/dev/null 2>&1 || git config user.name Proximus
else
    git init -q &&
        git config user.email proximus@localhost &&
        git config user.name Proximus
    status=initialized
fi

# .timer is harness runtime state, not the agent's work: the task image's
# timer daemon and reanchor_timer.sh own it, both as root, and the daemon
# rewrites it constantly. Left tracked it dirties every `git status` and, as
# root-owned files, can fail the model's own `git add -A` outright
# ("Permission denied"). Written to .git/info/exclude rather than .gitignore
# so the agent's own repo files are left untouched.
if [ -d .git ]; then
    mkdir -p .git/info
    if ! grep -qx '/.timer/' .git/info/exclude 2>/dev/null; then
        printf '/.timer/\n' >> .git/info/exclude
    fi
    # Already tracked from an earlier run or the task's own history: untrack
    # it (keeping the working copy) or the exclude has no effect.
    git rm -r --cached -q --ignore-unmatch .timer 2>/dev/null || true
fi

# Task-declared /app artifact excludes (exclude_patterns env var, one
# gitwildmatch pattern per line): grading quarantines these paths anyway, so
# tracking them only bloats every submission commit and the workspace capture —
# caches, model checkpoints, virtualenvs. Same .git/info/exclude home as
# .timer, appended BEFORE the baseline commit so a fresh repo never adds
# them at all. Tracked matches from a task-shipped repo are untracked with
# -X against ONLY the declared patterns — never --exclude-standard, which
# would also apply the task repo's own .gitignore and untrack force-added
# files the quarantine contract does not cover (phantom deletions in every
# submission).
if [ -d .git ] && [ -n "${exclude_patterns:-}" ]; then
    mkdir -p .git/info
    printf '%s\n' "$exclude_patterns" > .git/info/proximus-app-excludes
    while IFS= read -r pattern; do
        [ -n "$pattern" ] || continue
        if ! grep -qxF "$pattern" .git/info/exclude 2>/dev/null; then
            printf '%s\n' "$pattern" >> .git/info/exclude
        fi
    done < .git/info/proximus-app-excludes
    git ls-files -ci -X .git/info/proximus-app-excludes -z 2>/dev/null |
        xargs -0 -r git rm --cached -q --ignore-unmatch -- 2>/dev/null || true
fi

if [ "$status" = initialized ]; then
    git add -A 2>/dev/null
    # --no-verify: this repo was just created, so there are no hooks to
    # respect; an empty workspace yields no commit, which is fine (the
    # model's first commit creates HEAD). stdout is dropped too: with -q,
    # "nothing to commit" still prints there and would pollute the status.
    git commit -qm "Initial workspace state (proximus baseline)" \
        --no-verify >/dev/null 2>&1
fi

# Root-created .git must not lock the agent out on tasks where the model
# can still run git.
if [ -d .git ]; then
    owner=$(stat -c '%u:%g' . 2>/dev/null) || owner=""
    if [ -n "$owner" ]; then
        chown -R "$owner" .git 2>/dev/null || true
    fi
fi
echo "$status"

# Pin $commit as submission #$idx (a submission IS a git commit).
# Inputs arrive as exec ENV VARS — wd, commit, idx — never spliced into this
# script, so there is nothing to shell-quote. Runs as root: the agent user
# often cannot invoke git (task images wrap or hide it), so harness git
# must not inherit default_user.
#
# Empty / HEAD means "snapshot the live workspace": the model cannot commit
# when git is disabled, so HEAD is stale. An explicit ref that resolves is
# pinned as-is; an explicit ref that does not resolve fails (the model
# named a commit that is not in this repo).
# stdout contract (parsed by submissions.py): `subject=<s>` then `hash=<sha>`,
# plus `committed=1` when the harness itself made the commit.
cd "$wd" || { echo "workspace directory missing" >&2; exit 3; }
command -v git >/dev/null 2>&1 || {
    echo "git is not installed in the task image" >&2
    exit 7
}

git config user.email >/dev/null 2>&1 || git config user.email proximus@localhost
git config user.name >/dev/null 2>&1 || git config user.name Proximus

snapshot=0
made_commit=0
case "$commit" in
    ""|HEAD|head) snapshot=1 ;;
esac

if [ "$snapshot" -eq 0 ]; then
    full=$(git rev-parse -q --verify "$commit^{commit}") || {
        echo "'$commit' is not a commit in the workspace repository." \
            "Pass a valid commit hash, or omit commit to snapshot" \
            "the current workspace." >&2
        exit 6
    }
else
    # Stage everything the exclude list allows, then commit if the tree
    # moved; a clean tree with HEAD reuses that commit (confirm-reuse).
    git add -A 2>/dev/null || true
    if git rev-parse -q --verify HEAD^{commit} >/dev/null \
        && git diff --cached --quiet \
        && git diff --quiet; then
        full=$(git rev-parse HEAD)
    else
        git commit -qm "Proximus submission snapshot" --no-verify --allow-empty \
            >/dev/null 2>&1 || {
            echo "could not snapshot the current workspace" >&2
            exit 5
        }
        full=$(git rev-parse HEAD)
        made_commit=1
    fi
fi

git update-ref "refs/proximus/submissions/$idx" "$full" || {
    echo "could not pin refs/proximus/submissions/$idx" >&2
    exit 5
}

# Root-created objects must not lock the agent out of .git on tasks
# where the model can still run git.
if [ -d .git ]; then
    owner=$(stat -c '%u:%g' . 2>/dev/null) || owner=""
    if [ -n "$owner" ]; then
        chown -R "$owner" .git 2>/dev/null || true
    fi
fi

git log -1 --format='subject=%s' "$full"
echo "hash=$full"
if [ "$made_commit" -eq 1 ]; then
    echo "committed=1"
fi

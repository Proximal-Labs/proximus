# Check out the winning submission so grading and the end-of-run workspace
# capture see the submitted tree, not whatever the last turn left behind. The
# pre-restore HEAD is pinned under refs/proximus/end-state first so the
# kill-time committed state stays reachable in the repository. checkout -f
# rewrites TRACKED files only: untracked files survive the restore.
# Inputs as exec env vars: wd, winner. stdout contract: `restored`.
# Runs as root (see record_commit.sh).
cd "$wd" || { echo "workspace directory missing" >&2; exit 3; }
git update-ref refs/proximus/end-state HEAD || true
git checkout -q -f "$winner" || { echo "checkout failed" >&2; exit 4; }
echo "restored"

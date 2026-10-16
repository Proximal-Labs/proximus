# Exit 0 when a confirming submit names the current winner. A bare/HEAD
# $commit means "snapshot the live workspace", so the live tree (tracked +
# untracked, excluding ignored) must match $winner — the agent user often
# cannot commit, so HEAD may be stale while the tree moved. An explicit
# $commit matches when it resolves to $winner.
# Inputs as exec env vars: wd, winner, commit. stdout contract: `matched`.
# Runs as root (see record_commit.sh).
cd "$wd" || { echo "workspace directory missing" >&2; exit 3; }
git rev-parse -q --verify "$winner^{commit}" >/dev/null || exit 1
case "$commit" in
    ""|HEAD|head)
        git diff --quiet "$winner" || exit 1
        untracked=$(git ls-files --others --exclude-standard)
        [ -z "$untracked" ] || exit 1
        ;;
    *)
        full=$(git rev-parse -q --verify "$commit^{commit}") || exit 1
        [ "$full" = "$winner" ] || exit 1
        ;;
esac
echo "matched"

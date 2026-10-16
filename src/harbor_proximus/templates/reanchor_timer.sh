# Re-anchor the task's own timer so its countdown means "seconds until
# YOUR run ends". Two task conventions, newest first:
#   1. sandbox-timer (FrontierSWE v2 tasks): /usr/local/bin/sandbox-timer computes
#      remaining = TASK_BUDGET_SECS - (now - /sandbox-timer/start), with a
#      root-owned first-wins anchor written at container boot. Reseed the
#      ANCHOR: boot-to-agent-start time (image pull, artifact staging) must
#      not be billed to the model.
#   2. /app/timer.sh (older task sets): restart the daemon seeded with the
#      remaining seconds.
# Prints the fresh countdown value as proof, or $missing when the task
# ships no timer at all.
#
# Inputs, prepended as variable assignments by timer.py:
#   remaining  seconds until THIS run ends, measured now (not the full
#              budget: on any re-arm after time has passed, seeding the
#              full budget would over-promise)
#   missing    sentinel to print when there is no timer script (timer.py
#              matches on it to tell "no timer" from "failed to start")
if command -v sandbox-timer >/dev/null 2>&1 && [ -n "${TASK_BUDGET_SECS:-}" ]; then
  b="${TASK_BUDGET_SECS%.*}"
  mkdir -p /sandbox-timer
  # remaining(t) = b - (t - anchor)  =>  anchor = now - b + remaining
  echo $(( $(date +%s) - b + remaining )) > /sandbox-timer/start
  sandbox-timer remaining 2>/dev/null || printf '%s\n' "$remaining"
  exit 0
fi
if [ ! -x /app/timer.sh ]; then
  printf '%s\n' "$missing"
  exit 0
fi
mkdir -p /app/.timer
pid_file=/app/.timer/timer.pid
pid="$(cat "$pid_file" 2>/dev/null || true)"
if [ -n "$pid" ]; then
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
fi
# A SIGKILLed daemon leaves its lock behind; clear both so the fresh one is
# not turned away by its own stale predecessor. Alert markers reset too, so
# the 30/10/5-minute warnings fire against the new deadline. The countdown
# files go as well: the value read back below then proves the new daemon
# started, instead of echoing the dead one's last write.
rm -f "$pid_file"
rm -rf /app/.timer/.timer.lock
rm -f /app/.timer/alert_30min /app/.timer/alert_10min /app/.timer/alert_5min
rm -f /app/.timer/remaining_secs /app/.timer/elapsed_secs
env -u BASH_ENV -u ENV TASK_BUDGET_SECS="$remaining" HARBOR_AGENT_TIMEOUT_SEC="$remaining" setsid /app/timer.sh >/tmp/proximus-timer.log 2>&1 &
sleep 1
cat /app/.timer/remaining_secs 2>/dev/null || true

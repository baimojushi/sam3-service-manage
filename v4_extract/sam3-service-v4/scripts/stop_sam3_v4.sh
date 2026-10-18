#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="${APP_ROOT:-/opt/molink/app}"
PID_FILE="${PID_FILE:-$APP_ROOT/run/sam3-v4.pid}"
if [[ ! -f "$PID_FILE" ]]; then
  echo "SAM3 v4 PID file not found"; exit 0
fi
pid="$(cat "$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid PID file; refusing action" >&2; exit 1
fi
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Stale PID file removed"; exit 0
fi
if [[ ! -r "/proc/$pid/cmdline" ]] || ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "sam3_service_v4.run"; then
  echo "Process $pid is not sam3_service_v4.run, refusing to stop" >&2
  exit 1
fi
kill -TERM "$pid"
for n in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "SAM3 v4 stopped"
    exit 0
  fi
  sleep 1
done
echo "Process $pid still alive. Inspect before any forced termination." >&2
exit 1

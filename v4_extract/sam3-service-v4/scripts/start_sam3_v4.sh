#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="${APP_ROOT:-/opt/molink/app}"
PROJECT_DIR="${PROJECT_DIR:-$APP_ROOT/sam3-service-v4}"
PYTHON="${PYTHON:-$APP_ROOT/.venv/bin/python3}"
PID_FILE="${PID_FILE:-$APP_ROOT/run/sam3-v4.pid}"
LOG_FILE="${LOG_FILE:-$APP_ROOT/logs/sam3-v4.log}"
PORT="${SAM3_VIDEO_PORT:-6014}"
mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SAM3_VIDEO_PORT="$PORT"
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq 'sam3_service_v4.run'; then
      echo "SAM3 v4 already running: PID $pid"; exit 0
    fi
    echo "PID file refers to a different process: $pid. Refusing to kill or replace it." >&2
    exit 1
  fi
fi
# Fail if any other service has the target port. Do not pkill broad patterns.
if "$PYTHON" - "$PORT" <<'PY'
import socket, sys
sock=socket.socket()
try:
    sock.settimeout(.5)
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
else:
    sys.exit(0)
finally:
    sock.close()
PY
then
  echo "Port $PORT is occupied; inspect ss -ltnp before proceeding." >&2
  exit 1
fi
cd "$PROJECT_DIR"
nohup "$PYTHON" -u -m sam3_service_v4.run >> "$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
echo "Started SAM3 v4 PID=$pid (loading model, health/ready stays 503 until done)"
for n in $(seq 1 240); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "SAM3 process exited while starting. Log: $LOG_FILE" >&2
    exit 1
  fi
  if curl -fSs --max-time 2 "http://127.0.0.1:$PORT/health/ready" >/dev/null 2>&1; then
    echo "SAM3 v4 ready on port $PORT"
    exit 0
  fi
  sleep 1
done
echo "SAM3 v4 did not reach ready status. Inspect $LOG_FILE" >&2
exit 1

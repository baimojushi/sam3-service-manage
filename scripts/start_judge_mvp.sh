#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"
PORT="${JUDGE_PORT:-6012}"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/run"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Load production environment first, then local overrides
if [ -f /etc/molink/gpu.env ]; then
  set +e; source /etc/molink/gpu.env; set -e
fi
if [ -f "$ROOT/.env.local" ]; then
  set -a
  source "$ROOT/.env.local"
  set +a
fi

source /opt/molink/app/.venv/bin/activate

# Check if already running
if python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(0.35)
try:
    s.connect(('127.0.0.1', int(sys.argv[1])))
    print('1')
except Exception:
    print('0')
finally:
    s.close()
" "$PORT" | grep -qx "1"; then
  echo "judge_mvp already running on port $PORT"
  exit 0
fi

cd "$ROOT"
echo "Starting judge_mvp on port $PORT..."
nohup /opt/molink/app/.venv/bin/python -m uvicorn judge_mvp.service:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  --log-level info \
  > "$LOG_DIR/judge.log" 2>&1 &

echo $! > "$PID_DIR/judge.pid"

# Wait for health
for i in $(seq 1 30); do
  sleep 1
  code=$(python3 -c "
import http.client, sys
try:
    conn = http.client.HTTPConnection('127.0.0.1', int(sys.argv[1]), timeout=3)
    conn.request('GET', '/health')
    resp = conn.getresponse()
    print(resp.status)
except Exception:
    print('0')
" "$PORT" 2>/dev/null || echo "0")
  if [ "$code" = "200" ]; then
    echo "judge_mvp healthy on port $PORT (pid $(cat "$PID_DIR/judge.pid"))"
    exit 0
  fi
done

echo "ERROR: judge_mvp failed to start. Check $LOG_DIR/judge.log"
exit 1
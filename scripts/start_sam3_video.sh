#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/app}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
PID_DIR="${PID_DIR:-$ROOT/run}"
mkdir -p "$LOG_DIR" "$PID_DIR"

SAM3_SERVICE_FILE="${SAM3_SERVICE_FILE:-/mnt/e/项目/深度相机/sam3_video_service.py}"
SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-/mnt/e/models/sam3/models/facebook--sam3/snapshots/master/sam3.pt}"
SAM3_VIDEO_PORT="${SAM3_VIDEO_PORT:-6014}"

export PYTHONPATH="/opt/sam3-src:${PYTHONPATH:-}"
export SAM3_MODEL_PATH
export ALLOW_UNAUTH_LOCAL=1
export SAM3_VIDEO_PORT

cd "$ROOT"

echo "[start:sam3] Starting SAM3 video service on port $SAM3_VIDEO_PORT"
echo "[start:sam3] Service file: $SAM3_SERVICE_FILE"
echo "[start:sam3] Model: $SAM3_MODEL_PATH"

nohup /opt/molink/app/.venv/bin/python -u "$SAM3_SERVICE_FILE" > "$LOG_DIR/sam3.log" 2>&1 &
echo $! > "$PID_DIR/sam3.pid"

# Wait for HTTP server to start (port listening)
for _ in $(seq 1 30); do
    if python3 -c "import socket, sys; s = socket.socket(); s.settimeout(0.5); s.connect(('127.0.0.1', int(sys.argv[1]))); print('1')" "$SAM3_VIDEO_PORT" 2>/dev/null; then
        echo "[start:sam3] HTTP server is up on port $SAM3_VIDEO_PORT"
        exit 0
    fi
    sleep 1
done

echo "[warn:sam3] port $SAM3_VIDEO_PORT not listening after 30s; check $LOG_DIR/sam3.log"
exit 1
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"
ACTION="${1:-status}"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/run"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Load production environment first, then local overrides
[ -f /etc/molink/gpu.env ] && source /etc/molink/gpu.env
[ -f "$ROOT/.env.local" ] && source "$ROOT/.env.local"

# name|port|health|fallback-start-scripts|kind
SERVICE_SPECS=(
  "metric3d|6006|/health|start_giant_auto.sh,start_giant.sh,start.sh,services/metric3d_api/start_giant_auto.sh,services/metric3d_api/start_giant.sh,services/metric3d_api/start.sh,metric3d_ctl.sh|api"
  "semantic|6011|/health|scripts/start_semantic_v1_sam.sh,start_semantic_mvp.sh|api"
  "hanging|6010|/health|scripts/start_hanging_v1.sh,start_hanging_mvp.sh|api"
  "judge|6012|/health|scripts/start_judge_mvp.sh|api"
  "sam3|6014|/health|scripts/start_sam3_video.sh|api"
)
MVP_SERVICES="${MVP_SERVICES:-metric3d,semantic,hanging,judge,sam3}"

_service_enabled() {
  local name="$1"
  local csv=",$MVP_SERVICES,"
  [ "${csv#*,${name},}" != "$csv" ]
}

_have_systemctl() {
  command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files >/dev/null 2>&1
}

_unit_for_name() {
  case "$1" in
    metric3d) echo "metric3d-api.service" ;;
    semantic) echo "semantic-mvp.service" ;;
    hanging) echo "hanging-mvp.service" ;;
    judge) echo "judge-mvp.service" ;;
    sam3) echo "sam3-video.service" ;;
    *) echo "" ;;
  esac
}

_have_unit() {
  local unit="$1"
  [ -n "$unit" ] || return 1
  systemctl list-unit-files 2>/dev/null | awk '{print $1}' | grep -qx "$unit"
}

_port_open() {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(0.35)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    print("1")
except Exception:
    print("0")
finally:
    s.close()
PY
}

_http_health() {
  python3 - "$1" "$2" <<'PY'
import http.client, os, sys
port = int(sys.argv[1])
path = sys.argv[2]
try:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=float(os.getenv("MVP_HEALTH_TIMEOUT", "5.0")))
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read(180).decode("utf-8", "ignore").replace("\n", " ")
    print(f"HTTP {resp.status} {body}")
except Exception as e:
    print(f"DOWN {type(e).__name__}: {e}")
PY
}

_find_start_script() {
  local spec="$1"
  local scripts_csv
  scripts_csv=$(echo "$spec" | cut -d'|' -f4)
  IFS=',' read -ra scripts <<< "$scripts_csv"
  for rel in "${scripts[@]}"; do
    local cand="$ROOT/$rel"
    if [ -f "$cand" ] && [ -x "$cand" ]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

_start_one() {
  local name="$1"
  local spec
  for s in "${SERVICE_SPECS[@]}"; do
    if [[ "$s" == "$name|"* ]]; then
      spec="$s"
      break
    fi
  done
  [ -n "$spec" ] || { echo "[err:$name] no spec"; return 1; }

  local port health scripts kind
  port=$(echo "$spec" | cut -d'|' -f2)
  health=$(echo "$spec" | cut -d'|' -f3)
  scripts=$(echo "$spec" | cut -d'|' -f4)
  kind=$(echo "$spec" | cut -d'|' -f5)

  if ! _service_enabled "$name"; then
    echo "[skip:$name] not in MVP_SERVICES"
    return 0
  fi

  local start_script
  start_script=$(_find_start_script "$spec")
  if [ -z "$start_script" ]; then
    echo "[err:$name] no start script found among: $scripts"
    return 1
  fi

  echo "[start:$name] $start_script -> $LOG_DIR/$name.log"
  nohup bash "$start_script" > "$LOG_DIR/$name.log" 2>&1 &
  echo $! > "$PID_DIR/$name.pid"

  local retries=25
  while [ $retries -gt 0 ]; do
    sleep 1
    if python3 - "$port" <<'PY' 2>/dev/null; then
      echo "[ok:$name] port $port is listening"
      return 0
    fi
    retries=$((retries - 1))
  done

  echo "[warn:$name] port $port not listening after 25s; check $LOG_DIR/$name.log"
  return 1
}

_stop_one() {
  local name="$1"
  [ -f "$PID_DIR/$name.pid" ] && kill "$(cat "$PID_DIR/$name.pid")" 2>/dev/null || true
  rm -f "$PID_DIR/$name.pid"

  case "$name" in
    semantic)
      pkill -f 'uvicorn semantic_mvp.service:app' || true
      pkill -f 'semantic_mvp.service:app' || true
      ;;
    hanging)
      pkill -f 'uvicorn hanging_mvp.service:app' || true
      pkill -f 'hanging_mvp.service:app' || true
      ;;
    metric3d)
      pkill -f 'uvicorn.*metric3d' || true
      pkill -f 'metric3d.*uvicorn' || true
      pkill -f 'metric3d_api.*app' || true
      ;;
    judge)
      pkill -f 'uvicorn judge_mvp.service:app' || true
      pkill -f 'judge_mvp.service:app' || true
      ;;
    sam3)
      pkill -f 'uvicorn.*6014' || true
      pkill -f 'sam3_video_service' || true
      ;;
  esac
}

_force_stop_port() {
  local port="$1"
  local name="$2"
  case "$name" in
    metric3d)  pkill -f 'uvicorn.*6006' 2>/dev/null || true ;;
    semantic)  pkill -f 'uvicorn.*6011' 2>/dev/null || true ;;
    hanging)   pkill -f 'uvicorn.*6010' 2>/dev/null || true ;;
    judge)     pkill -f 'uvicorn.*6012' 2>/dev/null || true ;;
    sam3)      pkill -f 'uvicorn.*6014' 2>/dev/null || true ;;
  esac
}

status() {
  echo "== MVP API status =="
  echo "service filter: $MVP_SERVICES"
  for spec in "${SERVICE_SPECS[@]}"; do
    local name port health scripts kind
    name=$(echo "$spec" | cut -d'|' -f1)
    port=$(echo "$spec" | cut -d'|' -f2)
    health=$(echo "$spec" | cut -d'|' -f3)
    scripts=$(echo "$spec" | cut -d'|' -f4)
    kind=$(echo "$spec" | cut -d'|' -f5)

    if ! _service_enabled "$name"; then continue; fi

    local hres
    hres=$(_http_health "$port" "$health")
    if [[ "$hres" == HTTP\ 200* ]]; then
      echo "[UP]   $name port=$port health=$hres"
      [ -f "$PID_DIR/$name.pid" ] && echo "       pidfile: $(cat "$PID_DIR/$name.pid")"
      [ -f "$LOG_DIR/$name.log" ] && echo "       log: $LOG_DIR/$name.log"
    else
      echo "[DOWN] $name port=$port health=$hres"
      [ -f "$LOG_DIR/$name.log" ] && echo "       log: $LOG_DIR/$name.log"
      [ -f "$PID_DIR/$name.pid" ] && echo "       pidfile: $(cat "$PID_DIR/$name.pid")"
    fi
  done

  echo
  echo "== Matching processes =="
  for spec in "${SERVICE_SPECS[@]}"; do
    local name port health scripts kind
    name=$(echo "$spec" | cut -d'|' -f1)
    port=$(echo "$spec" | cut -d'|' -f2)
    if ! _service_enabled "$name"; then continue; fi
    pgrep -af "uvicorn.*$port|$name" 2>/dev/null | grep -v grep | head -5
  done
}

start() {
  for spec in "${SERVICE_SPECS[@]}"; do
    local name
    name=$(echo "$spec" | cut -d'|' -f1)
    _service_enabled "$name" || continue
    _start_one "$name"
  done
  # worker if defined
  if _service_enabled "worker"; then
    echo "[start:worker] python3 -m worker.main -> $LOG_DIR/worker.log"
    nohup python3 -m worker.main > "$LOG_DIR/worker.log" 2>&1 &
    echo $! > "$PID_DIR/worker.pid"
  fi
}

stop() {
  for spec in "${SERVICE_SPECS[@]}"; do
    local name
    name=$(echo "$spec" | cut -d'|' -f1)
    _service_enabled "$name" || continue
    _stop_one "$name"
  done
  [ -f "$PID_DIR/worker.pid" ] && kill "$(cat "$PID_DIR/worker.pid")" 2>/dev/null || true
  rm -f "$PID_DIR/worker.pid"
}

restart() {
  stop
  sleep 2
  start
}

case "$ACTION" in
  status) status ;;
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  *) echo "Usage: $0 {status|start|stop|restart}" >&2; exit 1 ;;
esac
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_DIR="$ROOT_DIR/.dev/local"
LOG_DIR="$DEV_DIR/logs"
PID_DIR="$DEV_DIR/pids"
REDIS_MODE_FILE="$DEV_DIR/redis.mode"

mkdir -p "$LOG_DIR" "$PID_DIR"

compose_cmd() {
  if command -v docker-compose >/dev/null 2>&1; then
    echo docker-compose
  elif docker compose version >/dev/null 2>&1; then
    echo "docker compose"
  else
    return 1
  fi
}

load_env() {
  if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  fi

  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  export SESSION_DB_PATH="${SESSION_DB_PATH:-$ROOT_DIR/sessions.db}"
  export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://127.0.0.1:6379/0}"
  export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://127.0.0.1:6379/1}"
  export PYTHONUNBUFFERED=1
  export QUANT_DATABASE__HOST="${QUANT_DATABASE__HOST:-localhost}"
  export QUANT_DATABASE__PORT="${QUANT_DATABASE__PORT:-8123}"
  export QUANT_DATABASE__USERNAME="${QUANT_DATABASE__USERNAME:-default}"
  export QUANT_DATABASE__PASSWORD="${QUANT_DATABASE__PASSWORD:-}"
  export API_PORT="${API_PORT:-8000}"
  export UI_PORT="${UI_PORT:-5173}"
  export FLOWER_PORT="${FLOWER_PORT:-5555}"
  export CELERY_WORKER_CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-2}"
  export CELERY_QUEUES="${CELERY_QUEUES:-backtest,default,automation}"
}

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1" >&2
    exit 1
  fi
}

is_port_open() {
  python3 - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
try:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        print("1" if sock.connect_ex(("127.0.0.1", port)) == 0 else "0")
except OSError:
    print("0")
PY
}

pid_is_running() {
  local pid="$1"
  [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1
}

start_background() {
  local name="$1"
  local workdir="$2"
  local command="$3"
  local pidfile="$PID_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"

  if [ -f "$pidfile" ] && pid_is_running "$(cat "$pidfile")"; then
    echo "[INFO] $name already running (pid $(cat "$pidfile"))"
    return
  fi

  rm -f "$pidfile"
  (
    cd "$workdir"
    nohup /bin/zsh -lc "$command" >"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )
  echo "[INFO] Started $name (pid $(cat "$pidfile"))"
  echo "[INFO] Logs: $logfile"
}

stop_background() {
  local name="$1"
  local pidfile="$PID_DIR/$name.pid"

  if [ ! -f "$pidfile" ]; then
    echo "[INFO] $name not running"
    return
  fi

  local pid
  pid="$(cat "$pidfile")"
  if pid_is_running "$pid"; then
    pkill -TERM -P "$pid" >/dev/null 2>&1 || true
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if pid_is_running "$pid"; then
      pkill -KILL -P "$pid" >/dev/null 2>&1 || true
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    echo "[INFO] Stopped $name"
  else
    echo "[INFO] $name already stopped"
  fi
  rm -f "$pidfile"
}

start_redis() {
  if [ "$(is_port_open 6379)" = "1" ]; then
    echo "reuse" > "$REDIS_MODE_FILE"
    echo "[INFO] Reusing existing Redis on 127.0.0.1:6379"
    return
  fi

  if command -v redis-server >/dev/null 2>&1; then
    mkdir -p "$DEV_DIR/redis"
    redis-server \
      --port 6379 \
      --save '' \
      --appendonly no \
      --daemonize yes \
      --pidfile "$PID_DIR/redis.pid" \
      --logfile "$LOG_DIR/redis.log" \
      --dir "$DEV_DIR/redis"
    echo "local" > "$REDIS_MODE_FILE"
    echo "[INFO] Started local redis-server"
    return
  fi

  local compose
  compose="$(compose_cmd)"
  (cd "$ROOT_DIR" && $compose up -d redis)
  echo "docker" > "$REDIS_MODE_FILE"
  echo "[INFO] Started Redis via docker compose"
}

stop_redis() {
  if [ ! -f "$REDIS_MODE_FILE" ]; then
    return
  fi

  local mode
  mode="$(cat "$REDIS_MODE_FILE")"
  case "$mode" in
    local)
      if [ -f "$PID_DIR/redis.pid" ]; then
        local pid
        pid="$(cat "$PID_DIR/redis.pid")"
        if pid_is_running "$pid"; then
          kill "$pid" >/dev/null 2>&1 || true
        fi
        rm -f "$PID_DIR/redis.pid"
      fi
      echo "[INFO] Stopped local redis-server"
      ;;
    docker)
      local compose
      compose="$(compose_cmd)"
      (cd "$ROOT_DIR" && $compose stop redis >/dev/null)
      echo "[INFO] Stopped docker compose Redis"
      ;;
    reuse)
      echo "[INFO] Redis was pre-existing; leaving it running"
      ;;
  esac

  rm -f "$REDIS_MODE_FILE"
}

show_status() {
  for name in api worker ui flower; do
    local pidfile="$PID_DIR/$name.pid"
    if [ -f "$pidfile" ] && pid_is_running "$(cat "$pidfile")"; then
      echo "[UP]   $name (pid $(cat "$pidfile"))"
    else
      echo "[DOWN] $name"
    fi
  done

  if [ -f "$REDIS_MODE_FILE" ]; then
    echo "[INFO] redis mode: $(cat "$REDIS_MODE_FILE")"
  elif [ "$(is_port_open 6379)" = "1" ]; then
    echo "[INFO] redis mode: external"
  else
    echo "[DOWN] redis"
  fi

  echo
  echo "API:    http://127.0.0.1:${API_PORT:-8000}/docs"
  echo "UI:     http://127.0.0.1:${UI_PORT:-5173}"
  echo "Flower: http://127.0.0.1:${FLOWER_PORT:-5555}"
}

logs() {
  local name="${1:-api}"
  if [ "$name" = "redis" ]; then
    local mode="none"
    [ -f "$REDIS_MODE_FILE" ] && mode="$(cat "$REDIS_MODE_FILE")"
    case "$mode" in
      local|reuse)
        local logfile="$LOG_DIR/redis.log"
        if [ -f "$logfile" ]; then
          tail -f "$logfile"
          return
        fi
        ;;
      docker)
        local compose
        compose="$(compose_cmd)"
        (cd "$ROOT_DIR" && $compose logs -f redis)
        return
        ;;
    esac
  fi

  local logfile="$LOG_DIR/$name.log"
  if [ ! -f "$logfile" ]; then
    echo "[ERROR] No log file for $name" >&2
    exit 1
  fi
  tail -f "$logfile"
}

up() {
  load_env
  ensure_cmd /bin/zsh
  ensure_cmd npm
  [ -x "$ROOT_DIR/.venv/bin/python" ] || { echo "[ERROR] Missing .venv/bin/python" >&2; exit 1; }
  [ -x "$ROOT_DIR/.venv/bin/celery" ] || { echo "[ERROR] Missing .venv/bin/celery" >&2; exit 1; }

  start_redis

  start_background "api" "$ROOT_DIR" \
    "export PYTHONPATH='$PYTHONPATH' SESSION_DB_PATH='$SESSION_DB_PATH' CELERY_BROKER_URL='$CELERY_BROKER_URL' CELERY_RESULT_BACKEND='$CELERY_RESULT_BACKEND' QUANT_DATABASE__HOST='$QUANT_DATABASE__HOST' QUANT_DATABASE__PORT='$QUANT_DATABASE__PORT' QUANT_DATABASE__USERNAME='$QUANT_DATABASE__USERNAME' QUANT_DATABASE__PASSWORD='$QUANT_DATABASE__PASSWORD' PYTHONUNBUFFERED=1; '$ROOT_DIR/.venv/bin/python' -m uvicorn src.api.server:app --reload --host 0.0.0.0 --port '$API_PORT'"

  start_background "worker" "$ROOT_DIR" \
    "export PYTHONPATH='$PYTHONPATH' SESSION_DB_PATH='$SESSION_DB_PATH' CELERY_BROKER_URL='$CELERY_BROKER_URL' CELERY_RESULT_BACKEND='$CELERY_RESULT_BACKEND' QUANT_DATABASE__HOST='$QUANT_DATABASE__HOST' QUANT_DATABASE__PORT='$QUANT_DATABASE__PORT' QUANT_DATABASE__USERNAME='$QUANT_DATABASE__USERNAME' QUANT_DATABASE__PASSWORD='$QUANT_DATABASE__PASSWORD' PYTHONUNBUFFERED=1; '$ROOT_DIR/.venv/bin/celery' -A src.tasks.celery_app worker --loglevel=info --concurrency='$CELERY_WORKER_CONCURRENCY' --queues='$CELERY_QUEUES' --max-tasks-per-child=100 --time-limit=3600 --soft-time-limit=3000"

  start_background "ui" "$ROOT_DIR/ui" \
    "npm run dev -- --host 0.0.0.0 --port '$UI_PORT'"

  if [ "${START_FLOWER:-0}" = "1" ]; then
    start_background "flower" "$ROOT_DIR" \
      "export PYTHONPATH='$PYTHONPATH' CELERY_BROKER_URL='$CELERY_BROKER_URL' CELERY_RESULT_BACKEND='$CELERY_RESULT_BACKEND'; '$ROOT_DIR/.venv/bin/celery' -A src.tasks.celery_app flower --port='$FLOWER_PORT'"
  fi

  echo
  echo "[INFO] Local debug stack is up"
  show_status
}

down() {
  stop_background "flower"
  stop_background "ui"
  stop_background "worker"
  stop_background "api"
  stop_redis
}

restart() {
  down
  up
}

case "${1:-up}" in
  up)
    up
    ;;
  down)
    down
    ;;
  restart)
    restart
    ;;
  status)
    load_env
    show_status
    ;;
  logs)
    logs "${2:-api}"
    ;;
  *)
    echo "Usage: $0 {up|down|restart|status|logs [api|worker|ui|flower|redis]}"
    exit 1
    ;;
esac

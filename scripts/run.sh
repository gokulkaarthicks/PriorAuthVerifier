#!/usr/bin/env bash
# Bash job control gives each service its own process group, including descendants.
# Compatible with macOS Bash 3.2 (does not require wait -n or external setsid).
set -euo pipefail
set -m
cd "$(dirname "$0")/.."
pids=()
cleanup() {
  trap '' INT TERM
  local pid
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  # Bound shutdown even if a service is stuck in model inference.
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi
    done
    if [ "$alive" -eq 0 ]; then break; fi
    sleep 0.5
  done
  for pid in "${pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
prepare() {
  "$@" &
  pids=("$!")
  wait "${pids[0]}"
  pids=()
}
prepare uv sync
prepare uv run python scripts/download_models.py
# npm ci uses the committed lockfile and ensures installed dependencies match it.
prepare npm --prefix frontend ci --no-audit --no-fund
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 &
pids=("$!")
npm --prefix frontend run dev &
pids+=("$!")
printf '\nFrontend: http://localhost:5173\nAPI: http://localhost:8000\nSwagger: http://localhost:8000/docs\nCtrl+C stops both services.\n\n'
while true; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      printf 'A service stopped (exit %s); shutting down both services.\n' "$status" >&2
      # An unexpected clean exit is still a failed combined development session.
      if [ "$status" -eq 0 ]; then status=1; fi
      exit "$status"
    fi
  done
  sleep 0.5 &
  wait "$!"
done

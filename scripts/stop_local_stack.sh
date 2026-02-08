#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.local_stack"
PID_DIR="$ROOT_DIR/.local_stack/pids"
STACK_LOG="$RUN_DIR/stack.log"

log_stack() {
  local message="$1"
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf '[%s] %s\n' "$ts" "$message" | tee -a "$STACK_LOG"
}

kill_tree() {
  local pid="$1"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

stop_one() {
  local name="$1"
  local pid_file="$PID_DIR/$name.pid"
  if [[ ! -f "$pid_file" ]]; then
    log_stack "[$name] not running (no pid file)"
    return 0
  fi

  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill_tree "$pid"
    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    log_stack "[$name] stopped (pid=$pid)"
  else
    log_stack "[$name] stale pid file (pid=$pid)"
  fi
  rm -f "$pid_file"
}

stop_one "mcp-qc"
stop_one "mcp-render"
stop_one "mcp-lipsync"
stop_one "mcp-xtts"
stop_one "mcp-memory"
stop_one "mcp-assets"
stop_one "vllm"

log_stack "Local stack stopped."

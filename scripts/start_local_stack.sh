#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.local_stack"
PID_DIR="$RUN_DIR/pids"
STACK_LOG="$RUN_DIR/stack.log"

log_stack() {
  local message="$1"
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf '[%s] %s\n' "$ts" "$message" | tee -a "$STACK_LOG"
}

if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    [[ "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key//[[:space:]]/}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    # UID/GID are readonly shell vars in bash; keep environment intact by skipping them.
    if [[ "$key" == "UID" || "$key" == "GID" ]]; then
      continue
    fi
    export "$key=$value"
  done <"$ROOT_DIR/.env"
fi

if [[ ! -f "$RUN_DIR/orchestrator_env.sh" ]]; then
  log_stack "missing env file: $RUN_DIR/orchestrator_env.sh"
  exit 1
fi
source "$RUN_DIR/orchestrator_env.sh"

mkdir -p "$PID_DIR" "$ARTIFACT_ROOT" "$DATA_ROOT/sqlite" "$DATA_ROOT/assets" "$DATA_ROOT/tts/speakers" "$DATA_ROOT/tts/piper_voices" "$DATA_ROOT/vllm-cache" "$VLLM_HF_HOME" "$VLLM_HF_HUB_CACHE" "$VLLM_TRANSFORMERS_CACHE"

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

start_service() {
  local name="$1"
  shift
  local pid_file="$PID_DIR/$name.pid"

  if is_running "$pid_file"; then
    log_stack "[$name] already running (pid=$(cat "$pid_file"))"
    return 0
  fi
  rm -f "$pid_file"

  (
    cd "$ROOT_DIR"
    {
      printf '[%s] [%s] cmd:' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$name"
      printf ' %q' "$@"
      printf '\n'
    } >>"$STACK_LOG"
    nohup "$@" >>"$STACK_LOG" 2>&1 &
    echo $! >"$pid_file"
  )
  log_stack "[$name] started (pid=$(cat "$pid_file"))"
  sleep 0.3
  if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    log_stack "[$name] exited immediately after start"
    rm -f "$pid_file"
    return 1
  fi
}

wait_port() {
  local host="$1"
  local port="$2"
  local timeout_sec="$3"
  local label="$4"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1.0)
s.connect(("${host}", ${port}))
s.close()
PY
    then
      log_stack "[$label] healthy on ${host}:${port}"
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= timeout_sec )); then
      log_stack "[$label] timed out waiting for ${host}:${port}"
      return 1
    fi
    sleep 1
  done
}

if ! command -v vllm >/dev/null 2>&1; then
  log_stack "[vllm] binary not found in PATH. Install vLLM CLI or fix PATH."
  exit 1
fi

start_service "vllm" env \
  HF_HOME="$VLLM_HF_HOME" HF_HUB_CACHE="$VLLM_HF_HUB_CACHE" HUGGINGFACE_HUB_CACHE="$VLLM_HF_HUB_CACHE" TRANSFORMERS_CACHE="$VLLM_TRANSFORMERS_CACHE" \
  vllm serve "$VLLM_MODEL" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --download-dir "$VLLM_DOWNLOAD_DIR" \
  --quantization "$VLLM_QUANTIZATION" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"

# Bring vLLM fully online before other services that may touch GPU memory.
wait_port "127.0.0.1" "$VLLM_PORT" "$VLLM_START_TIMEOUT_SEC" "vllm"

start_service "mcp-assets" env \
  MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_ASSETS_HOST" MCP_PORT="$MCP_ASSETS_PORT" MCP_PATH="$MCP_ASSETS_PATH" \
  ARTIFACT_ROOT="$ARTIFACT_ROOT" ASSET_CATALOG_PATH="$ASSET_CATALOG_PATH" \
  OPENVERSE_CLIENT_ID="${OPENVERSE_CLIENT_ID:-}" OPENVERSE_CLIENT_SECRET="${OPENVERSE_CLIENT_SECRET:-}" OPENVERSE_CLIENT_NAME="${OPENVERSE_CLIENT_NAME:-}" \
  "$PYTHON_BIN" -m mcp_servers.assets.mcp_server

start_service "mcp-memory" env \
  MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_MEMORY_HOST" MCP_PORT="$MCP_MEMORY_PORT" MCP_PATH="$MCP_MEMORY_PATH" \
  MEMORY_DB_PATH="$MEMORY_DB_PATH" \
  "$PYTHON_BIN" -m mcp_servers.memory.mcp_server

start_service "mcp-xtts" env \
  MCP_HOST="$MCP_XTTS_HOST" MCP_PORT="$MCP_XTTS_PORT" MCP_PATH="$MCP_XTTS_PATH" \
  XTTS_DEVICE="$XTTS_DEVICE" XTTS_MODEL="$XTTS_MODEL" XTTS_LANGUAGE="$XTTS_LANGUAGE" \
  COQUI_TOS_AGREED="$COQUI_TOS_AGREED" XTTS_SPEAKER_WAV_DIR="$XTTS_SPEAKER_WAV_DIR" \
  TTS_HOME="$TTS_HOME" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
  TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  ARTIFACT_ROOT="$ARTIFACT_ROOT" ARTIFACT_AUDIO_ROOT="$ARTIFACT_AUDIO_ROOT" \
  "$PYTHON_BIN" -m mcp_servers.qwen_tts.mcp_server

if [[ -f "$WAV2LIP_SCRIPT" ]]; then
  start_service "mcp-lipsync" env \
    MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_LIPSYNC_HOST" MCP_PORT="$MCP_LIPSYNC_PORT" MCP_PATH="$MCP_LIPSYNC_PATH" \
    WAV2LIP_SCRIPT="$WAV2LIP_SCRIPT" WAV2LIP_CHECKPOINT="$WAV2LIP_CHECKPOINT" \
    NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR" ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    "$PYTHON_BIN" -m mcp_servers.lipsync.mcp_server
else
  log_stack "[mcp-lipsync] skipped: missing $WAV2LIP_SCRIPT"
fi

start_service "mcp-render" env \
  MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_RENDER_HOST" MCP_PORT="$MCP_RENDER_PORT" MCP_PATH="$MCP_RENDER_PATH" \
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  "$PYTHON_BIN" -m mcp_servers.render.mcp_server

start_service "mcp-qc" env \
  MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_QC_HOST" MCP_PORT="$MCP_QC_PORT" MCP_PATH="$MCP_QC_PATH" \
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  "$PYTHON_BIN" -m mcp_servers.qc.mcp_server

wait_port "127.0.0.1" "$MCP_ASSETS_PORT" 30 "mcp-assets"
wait_port "127.0.0.1" "$MCP_MEMORY_PORT" 30 "mcp-memory"
wait_port "127.0.0.1" "$MCP_XTTS_PORT" 120 "mcp-xtts"
wait_port "127.0.0.1" "$MCP_RENDER_PORT" 30 "mcp-render"
wait_port "127.0.0.1" "$MCP_QC_PORT" 30 "mcp-qc"
if [[ -f "$PID_DIR/mcp-lipsync.pid" ]]; then
  wait_port "127.0.0.1" "$MCP_LIPSYNC_PORT" 60 "mcp-lipsync"
fi

log_stack "Stack is up. Source env for orchestrator:"
log_stack "  source $RUN_DIR/orchestrator_env.sh"
log_stack "Logs:"
log_stack "  tail -f $STACK_LOG"
tail -f /home/zodrak/AITHTR/aithtr/.local_stack/stack.log

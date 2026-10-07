#!/usr/bin/env bash
# Run ANY small GGUF model from Hugging Face as a CPU-only llama.cpp server on
# port 9090, next to your main model on :8080. Used to try small models as the
# book writer's checking model (tools/verdict_eval.py, tools/extractor_eval.py).
#
#   REPO=ibm-granite/granite-4.2-3b-GGUF QUANT=Q6_K ALIAS=granite tools/run-small-model.sh -d
#   tools/run-small-model.sh status | stop | download
#
# Environment (REPO is required):
#   REPO       Hugging Face repo, e.g. LiquidAI/LFM2.5-2.6B-GGUF
#   QUANT      quantisation to pick from the repo's files   (default Q6_K)
#   FILE       exact file name, if QUANT matches more than one
#   ALIAS      model name the API reports                    (default: small)
#   DEVICE     cpu (default) or gpu (offload all layers to the first GPU)
#   PORT, CTX, THREADS, EXTRA_ARGS (extra llama-server flags), FORCE=1
#
# Memory is the thing to watch (your main server holds most of the RAM): the
# script refuses to start unless enough is free, binds to 127.0.0.1 only, keeps
# the model off the GPU (--device none) and uses a small context.

set -euo pipefail

REPO="${REPO:-}"
QUANT="${QUANT:-Q6_K}"
PORT="${PORT:-9090}"
CTX="${CTX:-8192}"
THREADS="${THREADS:-3}"           # your main model uses -t 6 on 6 physical cores
ALIAS="${ALIAS:-small}"
HOST="${HOST_BIND:-127.0.0.1}"
KV_KIB_PER_TOKEN="${KV_KIB_PER_TOKEN:-100}"   # generous q8_0 KV estimate
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
STATE_DIR="${STATE_DIR:-$HOME}"
LOG="$STATE_DIR/small-model.log"
PIDFILE="$STATE_DIR/.small-model.pid"

log() { printf '[small-model] %s\n' "$*"; }
die() { printf '[small-model] ERROR: %s\n' "$*" >&2; exit 1; }

mem_available_mib() { awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo; }

server_pid() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid; pid="$(cat "$PIDFILE")"
  kill -0 "$pid" 2>/dev/null && grep -q llama-server "/proc/$pid/cmdline" 2>/dev/null \
    && echo "$pid"
}

# ------------------------------------------------------------------ download
# Picks the file for QUANT from the repo (names differ between repos) and
# prints "name size sha256".
resolve_file() {
  local api="https://huggingface.co/api/models/$REPO/tree/main"
  curl -fsS -m 30 "$api" | QUANT="$QUANT" FILE="${FILE:-}" python3 -c '
import json, os, sys
quant, want = os.environ["QUANT"].lower(), os.environ["FILE"]
files = [f for f in json.load(sys.stdin)
         if f.get("path", "").lower().endswith(".gguf")
         and "mmproj" not in f["path"].lower() and "-0000" not in f["path"]]
if want:
    pick = [f for f in files if f["path"] == want]
else:
    pick = sorted((f for f in files if quant in f["path"].lower()),
                  key=lambda f: len(f["path"]))
if not pick:
    names = ", ".join(sorted(f["path"] for f in files)) or "(no .gguf files)"
    sys.exit("no file matches; available: " + names)
f = pick[0]
print(f["path"], f.get("size", 0), (f.get("lfs") or {}).get("oid", "-"))
'
}

download() {
  [[ -n "$REPO" ]] || die "set REPO=<huggingface repo>, e.g. REPO=LiquidAI/LFM2.5-2.6B-GGUF"
  mkdir -p "$MODEL_DIR"
  local spec file size sha
  spec="$(resolve_file)" || die "$REPO: $spec"
  read -r file size sha <<<"$spec"
  MODEL="$MODEL_DIR/$file"

  local fresh=0
  if [[ -f "$MODEL" && "$(stat -c %s "$MODEL")" == "$size" ]]; then
    log "model already present: $MODEL ($((size / 1048576)) MiB)"
  else
    local free_mib; free_mib="$(df -Pm "$MODEL_DIR" | awk 'NR==2 {print $4}')"
    (( free_mib > size / 1048576 + 1024 )) || die "not enough disk space in $MODEL_DIR"
    log "downloading $file ($((size / 1048576)) MiB) -> $MODEL (resumable)"
    curl -fL -C - --progress-bar -o "$MODEL" \
      "https://huggingface.co/$REPO/resolve/main/$file" || die "download failed; rerun to resume"
    fresh=1
  fi
  if [[ "$sha" != "-" && ( "$fresh" == 1 || "${VERIFY:-0}" == "1" ) ]]; then
    log "verifying sha256..."
    [[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "$sha" ]] \
      || die "checksum mismatch for $MODEL (delete it and rerun)"
    log "checksum OK"
  fi
}

# --------------------------------------------------------------- memory guard
check_memory() {
  local file_mib kv_mib need avail
  file_mib=$(( $(stat -c %s "$MODEL") / 1048576 ))
  kv_mib=$(( CTX * KV_KIB_PER_TOKEN / 1024 ))
  need=$(( file_mib + kv_mib + 600 ))          # + compute buffers/overhead
  avail="$(mem_available_mib)"
  log "model ${file_mib} MiB + KV ~${kv_mib} MiB + overhead => need ~${need} MiB;" \
      "available ${avail} MiB (swap used: $(free -m | awk '/^Swap:/ {print $3}') MiB)"
  if (( avail < need + 1024 )); then
    if [[ "${FORCE:-0}" == "1" ]]; then
      log "WARNING: low memory, continuing because FORCE=1"
    else
      die "not enough free RAM (want ${need} MiB + 1 GiB headroom for your main server)." \
          "Try a smaller QUANT or CTX=4096, free memory, or FORCE=1."
    fi
  fi
}

# --------------------------------------------------------------------- server
port_in_use() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}\$"; }

prepare() {
  [[ -n "$LLAMA_SERVER" && -x "$LLAMA_SERVER" ]] || die "llama-server not found: put it on PATH or set LLAMA_SERVER=/path/to/llama-server"
  download
  check_memory
  port_in_use && die "port $PORT is already in use"
  # DEVICE=cpu (default): never touch the GPU. DEVICE=gpu: offload every layer
  # to the first GPU (use when the big model is stopped and the GPU is free).
  local device_args=(--device none -ngl 0)
  [[ "${DEVICE:-cpu}" == "gpu" ]] && device_args=(-ngl 99)
  SERVER_ARGS=(
    -m "$MODEL" -a "$ALIAS" -c "$CTX" -t "$THREADS"
    "${device_args[@]}"
    --parallel 1             # one slot: parallel slots split memory bandwidth
    -fa on -ctk q8_0 -ctv q8_0
    -b 512 -ub 512           # smaller batches, smaller compute buffers
    --jinja
    --host "$HOST" --port "$PORT"
  )
  # shellcheck disable=SC2206
  [[ -z "${EXTRA_ARGS:-}" ]] || SERVER_ARGS+=(${EXTRA_ARGS})
}

cmd_foreground() {
  prepare
  log "starting on http://$HOST:$PORT (model name '$ALIAS', ${THREADS} threads, ctx $CTX)"
  exec "$LLAMA_SERVER" "${SERVER_ARGS[@]}"
}

cmd_daemon() {
  if pid="$(server_pid)"; then log "already running (pid $pid)"; return 0; fi
  prepare
  log "starting in the background; log: $LOG"
  nohup "$LLAMA_SERVER" "${SERVER_ARGS[@]}" >"$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  local i
  for i in $(seq 1 180); do                     # up to ~3 minutes to load
    if ! server_pid >/dev/null; then die "server exited; last log lines: $(tail -3 "$LOG" | tr '\n' ' ')"; fi
    if curl -fs -m 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      log "ready: http://$HOST:$PORT  (pid $(cat "$PIDFILE"))"
      cmd_status; return 0
    fi
    sleep 1
  done
  die "not ready after 3 minutes; see $LOG"
}

cmd_stop() {
  if pid="$(server_pid)"; then
    log "stopping pid $pid"; kill "$pid"
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
    rm -f "$PIDFILE"; log "stopped"
  else
    rm -f "$PIDFILE"; log "not running"
  fi
}

cmd_status() {
  if pid="$(server_pid)"; then
    log "running: pid $pid, RSS $(awk '/VmRSS/ {printf "%.2f GiB", $2/1048576}' "/proc/$pid/status")," \
        "port $PORT"
  else
    log "not running"
  fi
  log "MemAvailable: $(mem_available_mib) MiB"
}

case "${1:-}" in
  ""|run|start)  cmd_foreground ;;
  -d|--daemon|daemon) cmd_daemon ;;
  stop)          cmd_stop ;;
  status)        cmd_status ;;
  download)      download ;;
  -h|--help|help) sed -n '2,20p' "$0" ;;
  *) die "unknown command '$1' (try --help)" ;;
esac

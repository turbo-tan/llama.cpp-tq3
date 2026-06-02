#!/usr/bin/env bash
set -euo pipefail

TAN_LLAMA_ROOT="${TAN_LLAMA_ROOT:-/home/awee/code/tan_llama}"
SERVER_BIN="${LLAMA_SERVER_BIN:-$TAN_LLAMA_ROOT/.work/llama-rotatek-private-main/build/bin/llama-server}"
MODEL="${MODEL:-/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4L2-legacy-qTQ3_4S.gguf}"
OUT_DIR="${OUT_DIR:-$TAN_LLAMA_ROOT/artifacts/tq3_4l2_legacy_53_rc_witness_$(date +%Y%m%d_%H%M%S)}"
PORT="${BENCHLOOP_PORT:-18123}"

EXPECTED_SERVER_SHA="6dca0d053f04a97dd21c59070362c4faaca8c2621166a76b638ac3edd319aa9b"
EXPECTED_MODEL_SHA="38304c17bb2afc9fbb6970006921322b6a7d1987eda2c6edb6703c7076aa7e1e"

server_sha="$(sha256sum "$SERVER_BIN" | awk '{print $1}')"
model_sha="$(sha256sum "$MODEL" | awk '{print $1}')"

if [[ "$server_sha" != "$EXPECTED_SERVER_SHA" ]]; then
    echo "server sha mismatch: $server_sha != $EXPECTED_SERVER_SHA" >&2
    exit 1
fi

if [[ "$model_sha" != "$EXPECTED_MODEL_SHA" ]]; then
    echo "model sha mismatch: $model_sha != $EXPECTED_MODEL_SHA" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
preset="$(mktemp "$OUT_DIR/preset.XXXXXX.json")"
trap 'rm -f "$preset"' EXIT

cat > "$preset" <<JSON
[
  {
    "key": "packed",
    "label": "Qwen3.6-27B-MTP-TQ3_4L2 legacy 53 RC",
    "path": "$MODEL",
    "base_args": [
      "-ngl", "99",
      "-ctk", "q4_0",
      "-ctv", "tq3_0",
      "--spec-type", "draft-mtp",
      "--spec-draft-n-min", "1",
      "--spec-draft-n-max", "2",
      "--spec-draft-p-min", "1.0",
      "--spec-draft-ngl", "99",
      "--spec-draft-type-k", "q4_0",
      "--spec-draft-type-v", "tq3_0"
    ]
  }
]
JSON

BENCHLOOP_PORT="$PORT" \
QWEN35_HARD_MAX_TOKENS=5000 \
QWEN36_MODEL_PRESETS_JSON="$preset" \
LLAMA_SERVER_BIN="$SERVER_BIN" \
python3 "$TAN_LLAMA_ROOT/scripts/run_qwen35_hard_coder_bench.py" \
    --output-dir "$OUT_DIR" \
    --models packed \
    --tasks async_rate_limiter

echo "$OUT_DIR"

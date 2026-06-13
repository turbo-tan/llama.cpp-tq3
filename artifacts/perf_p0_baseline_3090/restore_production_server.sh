#!/bin/bash
# Exact production server command captured 2026-06-10 before Phase 0 bench session
setsid /home/awee/code/llm-launch/release/bin/llama-server \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  -a qwen36-27b-mtp-tq3_4s-hybrid --host 0.0.0.0 --port 8085 \
  --threads 8 --threads-batch 8 --metrics --gpu-layers 99 -np 1 -c 65536 \
  -ctk q4_0 -ctv tq3_0 --spec-type draft-mtp --spec-draft-n-min 1 \
  --spec-draft-n-max 2 --spec-draft-p-min 1.0 --gpu-layers-draft 99 \
  --spec-draft-type-k q4_0 --spec-draft-type-v tq3_0 --cache-ram 0 --jinja \
  --mmproj /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/mmproj-BF16.gguf \
  </dev/null >>/tmp/llama-server-8085.log 2>&1 &
systemctl --user start llama-server.service  # supervisor restore

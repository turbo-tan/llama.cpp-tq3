# Phase 0 probes against the production server (2026-06-10)

Server: /home/awee/code/llm-launch/release/bin/llama-server (pre-rebase build)
Model: Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf (13.39 GiB)
GPU: RTX 3090 24GB, server resident at 18.0 GiB
Config: ngl 99, c 65536, K=q4_0 V=tq3_0, spec-type draft-mtp, n-min 1, n-max 2,
p-min 1.0, draft KV q4_0/tq3_0, mmproj loaded

## Probe 1: TG-dominated (prompt_n=16, n_predict=256, temp 0, no cache)
- predicted_per_second: 47.21
- draft_n: 174, draft_n_accepted: 167 (96.0%)

## Probe 2: PP-dominated (prompt_n=2210, n_predict=64, temp 0, no cache)
- prompt_per_second: 633.8
- predicted_per_second: 53.17
- draft_n: 43, draft_n_accepted: 41 (95.3%)

## Findings
1. The "failed to fit on 3090 at ngl=99" smoke failure is explained: this server
   occupies 18 GiB; a second instance cannot load. Not a model-size problem.
2. MTP acceptance at p-min=1.0, n-max=2 is 95-96%. Drafting fired on ~68% of
   decode steps (174 drafts over 256 tokens). The operating point is conservative:
   acceptance this high means deeper drafts (n-max 3-5) and/or lower p-min should
   raise effective TG. A1 sweep is the highest-value next experiment.
3. PP 633.8 t/s at 2.2K prompt on the 3090 with 64K ctx server config (old build).
4. Non-MTP TG baseline cannot be measured without exclusive GPU access.

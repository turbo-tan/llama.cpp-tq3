# HF Upload SOP

This is the standard release flow for `TQ3_4S` GGUF uploads.

## Required Release Assets

- `README.md`
- `thumbnail.png`
- main model GGUF
- `mmproj.gguf` for multimodal model lines

For Qwopus multimodal releases, `mmproj.gguf` should be uploaded together with the model.

## Validation Before Upload

Use the public runtime first:

- repo:
  - `turbo-tan/llama.cpp-tq3`

Minimum checks:

1. `llama-simple-chat` coherence smoke
2. `llama-server --reasoning off` strict smoke
3. `llama-bench`:
   - `pp2048`
   - `tg128`

Recommended strict server smoke:

```bash
PORT=8096
MODEL=/path/to/model.gguf

./build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 --port $PORT \
  -ngl 99 -c 4096 -np 1 \
  -ctk q8_0 -ctv q8_0 -fa on \
  --no-warmup --jinja \
  --reasoning off --reasoning-budget 0 --reasoning-format deepseek
```

Then request:

```bash
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"Write ONLY the word ok."}],"max_tokens":32,"temperature":0}'
```

Expected result:

- assistant content should be exactly `ok`

## Naming

Use final filenames before upload.

Example:

- `Qwopus3.5-27B-v3-Abliterated-TQ3_4S.gguf`
- `Qwopus3.5-27B-v3-Abliterated-mmproj.gguf`

Avoid temporary names like:

- `*-local-YYYYMMDD.gguf`

## Upload Workflow

Model-specific helpers should upload:

1. `README.md`
2. `thumbnail.png`
3. main model GGUF
4. `mmproj.gguf` if present for that model line

Current Abliterated helper:

- [upload_hf_qwopus35_27b_v3_abliterated_tq3_4s.py](/home/awee/code/tan_llama/scripts/upload_hf_qwopus35_27b_v3_abliterated_tq3_4s.py)

Run with:

```bash
HF_TOKEN=... python3 /home/awee/code/tan_llama/scripts/upload_hf_qwopus35_27b_v3_abliterated_tq3_4s.py
```

## Release Record

Store a small artifact text file with:

- quantization source
- output filename
- smoke-test result
- `pp2048`
- `tg128`

Keep it under:

- `artifacts/`

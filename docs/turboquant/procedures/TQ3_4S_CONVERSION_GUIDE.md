# TQ3_4S Quantization Guide

## Prerequisites

```bash
# Build the TurboQuant runtime
cd ~/code/llama.cpp-tq3
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build --target llama-quantize --target llama-bench --target llama-server -j$(nproc)
```

## Dense Models (e.g. Qwen3.5-27B, Qwopus-27B)

All weights equally active → uniform TQ3_4S.

```bash
./build/bin/llama-quantize --allow-requantize \
  --output-tensor-type q6_K \
  --token-embedding-type q6_K \
  source-Q8_0.gguf output-TQ3_4S.gguf TQ3_4S
```

Typical result: 27B → 12.9 GiB (4.06 BPW), fits ngl 99 on 16GB.

## MoE Models (e.g. Qwen3.6-35B-A3B, Qwen3.5-35B-A3B)

98% of params are in expert MLPs, but only 8/256 active per token → compress experts aggressively.

```bash
./build/bin/llama-quantize --allow-requantize \
  --tensor-type ffn_gate_exps=q2_K \
  --tensor-type ffn_up_exps=q2_K \
  --tensor-type ffn_down_exps=q3_K \
  --output-tensor-type q6_K \
  --token-embedding-type q6_K \
  source-Q8_0.gguf output-TQ3_4S.gguf TQ3_4S
```

Typical result: 35B → 12.4 GiB (3.07 BPW), fits ngl 99 on 16GB.

## Source Model

Always quantize from Q8_0 for best quality. Download from unsloth:

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/MODEL-GGUF', 'MODEL-Q8_0.gguf', local_dir='./models')
"
```

## Testing

### 1. Kill existing servers (SOP)

```bash
pkill -9 -f llama-server 2>/dev/null; sleep 2
```

### 2. Bench

```bash
# Dense model (ngl 99)
./build/bin/llama-bench -m output-TQ3_4S.gguf -ngl 99 -p 512 -n 128 -r 3

# MoE model (ngl 99, tq3_0 V cache)
./build/bin/llama-bench -m output-TQ3_4S.gguf -ngl 99 -fa 1 -ctk q4_0 -ctv tq3_0 -p 512 -n 128 -r 3
```

### 3. Quality (10-question smoke test)

Start server (use exec wrapper for background):

```bash
cat > /tmp/run_server.sh << 'EOF'
#!/bin/bash
exec /path/to/llama-server \
  -m output-TQ3_4S.gguf \
  -ngl 99 -c 4096 -np 1 -fa on -ctk q4_0 -ctv tq3_0 \
  --port 8085 --jinja \
  --reasoning off --reasoning-budget 0 --reasoning-format deepseek \
  > /tmp/server.log 2>&1
EOF
chmod +x /tmp/run_server.sh
/tmp/run_server.sh &
# Poll until ready
for i in $(seq 1 24); do sleep 5; curl -s http://localhost:8085/health | grep -q ok && echo "UP" && break; done
```

Run questions:

```bash
ask() {
  curl -s http://localhost:8085/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":128,\"temperature\":0}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['message'].get('content',''))"
}
ask "What is the capital of France? One sentence."
ask "What is 2+2? Just the number."
ask "Write a Python function to reverse a string."
ask "Explain gravity in one sentence."
ask "What year did WW2 end?"
ask "List 3 prime numbers under 20."
ask "Boiling point of water in Celsius?"
ask "Who wrote Romeo and Juliet? One sentence."
ask "Largest planet in our solar system?"
ask "Translate hello to Spanish. One word."
```

Expected: 10/10 correct.

## HF Upload

```bash
pip install huggingface_hub
python3 << 'PYEOF'
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo = "YTan2000/Model-Name-TQ3_4S"
create_repo(repo, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=repo)
api.upload_file(path_or_fileobj="thumbnail.png", path_in_repo="thumbnail.png", repo_id=repo)
api.upload_file(path_or_fileobj="mmproj-BF16.gguf", path_in_repo="mmproj-BF16.gguf", repo_id=repo)
api.upload_file(path_or_fileobj="output-TQ3_4S.gguf", path_in_repo="Model-Name-TQ3_4S.gguf", repo_id=repo)
PYEOF
```

## Thumbnail

```python
from PIL import Image, ImageDraw, ImageFont
W, H = 1200, 630
img = Image.new('RGB', (W, H))
for x in range(W):
    for y in range(H):
        t = x / W; s = y / H
        r = int(140*(1-t)*0.7 + 50*t + 40*s*(1-t))
        g = int(160*(1-t)*0.8 + 80*t + 30*s*(1-t))
        b = int(80*(1-t) + 200*t + 40*s)
        img.putpixel((x, y), (min(r,255), min(g,255), min(b,255)))
draw = ImageDraw.Draw(img)
draw.rounded_rectangle([(20, 20), (W-20, H-20)], radius=20, outline=(255,255,255,60), width=1)
small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
big_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56)
draw.text((55, 55), "YTan2000", fill=(220, 220, 220), font=small_font)
draw.text((55, 130), "/Model-Name-TQ3_4S", fill=(255, 255, 255), font=big_font)
img.save('thumbnail.png')
```

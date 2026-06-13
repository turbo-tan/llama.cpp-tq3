# Reddit Dashboard — TQ3_4S (2026-04-01)

## Core Result

`Qwen3.5-27B`, `wiki.test.raw`, full pass, `c=2048`

| Format | PPL | Size |
|--------|-----|------|
| `TQ3_4S` | `6.8224 +/- 0.04534` | `12.9 GiB` |
| `Q3_K_S` | `6.8630 +/- 0.04583` | `11.4 GiB` |
| `TQ3_1S` | `6.9807 +/- 0.04690` | `12.9 GiB` |
| `EXL3 3.0bpw` | `7.027580` | `~13.0 GiB` |

## Tight Summary

- `TQ3_4S` beats `Q3_K_S` on full-pass `c=2048` PPL by about `0.0406`
- `TQ3_4S` beats `TQ3_1S` by about `0.1583`
- `TQ3_4S` beats local `EXL3 3.0bpw` by about `0.2052`
- size is `12.9 GiB` for the clean base `TQ3_4S` model

## Caveats

- `Q3_K_S` is still smaller: `11.4 GiB`
- `EXL3 3.0bpw` result is from a local `145 x 2048` eval, not `llama-perplexity`
- current public claim should be about quality and runtime support, not kernel speed leadership

## Safer Headlines

1. `TQ3_4S now beats Q3_K_S on Qwen3.5-27B full-pass c=2048`
2. `Full-pass result: TQ3_4S reaches 6.8224 PPL on Qwen3.5-27B`
3. `TQ3_4S improves the original TQ3_1S line and edges out Q3_K_S on full-pass PPL`

## Punchier Headlines

1. `TurboQuant weights just cleared Q3_K_S on full-pass 27B`
2. `TQ3_4S beats Q3_K_S on Qwen3.5-27B at full-pass c=2048`
3. `From TQ3_1S to TQ3_4S: now ahead of Q3_K_S on the full 27B pass`

## One-Paragraph Reddit Version

`TQ3_4S` is now on a full-pass `c=2048` result for `Qwen3.5-27B`: `6.8224 +/- 0.04534` on `wiki.test.raw`. That puts it ahead of my local `Q3_K_S` reference at `6.8630 +/- 0.04583`, ahead of the older `TQ3_1S` release at `6.9807 +/- 0.04690`, and ahead of my local `EXL3 3.0bpw` run at `7.027580`. The clean base `TQ3_4S` GGUF is about `12.9 GiB`. The remaining weakness is speed, not quality.

## Short Comment Reply

`Updated result: the clean base TQ3_4S model is now at 6.8224 +/- 0.04534 on the full-pass c=2048 run for Qwen3.5-27B. My local Q3_K_S reference is 6.8630 +/- 0.04583, so TQ3_4S is now slightly ahead on that gate.`

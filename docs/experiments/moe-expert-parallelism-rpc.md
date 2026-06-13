# MoE Expert Parallelism over RPC: Experiment Plan

## Executive Summary

**Goal**: Run a MoE model larger than 24GB by splitting experts across two machines (local RTX 3090 + remote RTX 3090 via 1Gbps RPC).

**Key Insight**: Unlike dense models where every layer must communicate (pipeline parallelism), MoE models can use **expert parallelism** — only communicate when a token needs an expert on the remote machine.

**Expected Result**: ~100-130 tok/s for models up to ~40GB (vs 140 tok/s single-GPU for 35B TQ3_4S that fits in 24GB).

---

## 1. Background: Why This Works

### Current MoE Implementation in llama.cpp

```cpp
// In llama-graph.cpp, build_moe_ffn():
// 1. Router computes expert probabilities
logits = build_lora_mm(gate_inp, cur);  // [n_expert, n_tokens]
probs = ggml_soft_max(ctx0, logits);

// 2. Select top-K experts
selected_experts = ggml_argsort_top_k(ctx0, probs, n_expert_used);

// 3. Compute ONLY the selected experts (this is the key!)
up = build_lora_mm_id(up_exps, cur, selected_experts);    // [n_ff, n_expert_used, n_tokens]
gate = build_lora_mm_id(gate_exps, cur, selected_experts);
// ... activation ...
down = build_lora_mm_id(down_exps, cur, selected_experts);
```

**Key observation**: `ggml_mul_mat_id` only computes the selected experts. If we can split experts across machines and only send tokens to the remote machine when needed, we achieve expert parallelism!

### Communication Analysis

For Qwopus3.6-35B-A3B with 128 experts, top-3 routing:

```
If experts split 80/48 (local/remote):
  P(all 3 experts local) = (80/128)³ = 24.4%
  P(at least 1 remote) = 75.6%
  Average remote experts per token = 3 × (48/128) = 1.125

Communication per token:
  - Best case (all local): 0 bytes
  - Worst case (all remote): 3 × hidden_size × 2 bytes = 3 × 5120 × 2 = 30 KB
  - Average: ~1.1 × 5120 × 2 = 11 KB sent + 11 KB received = 22 KB

On 1Gbps (125 MB/s):
  - Time per communication: 22 KB / 125 MB/s = 0.176 ms
  - Plus latency: ~0.5-1ms round trip
  - Total: ~0.7-1.2ms per token for network

Compare to compute time:
  - At 140 tok/s, each token takes ~7ms
  - Network adds ~1ms = ~14% overhead
  - Expected: ~120 tok/s (very reasonable!)
```

### Why This is Better Than Pipeline Parallelism

| Approach | Communication per token | On 1Gbps |
|----------|------------------------|----------|
| Pipeline (current RPC) | ~64 MB (all layers) | 3.9 tok/s |
| Expert Parallel (proposed) | ~22 KB average | ~120 tok/s |

**Expert parallelism does ~3000× less communication!**

---

## 2. Implementation Strategy

### Option A: Minimal Modification (Recommended for Experiment)

Instead of modifying ggml's internal graph, we intercept at a higher level:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Modified llama-server                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load model with experts split:                              │
│     - Local: layers 0-N with experts 0-79                       │
│     - Remote: experts 80-127 (stored on remote machine)         │
│                                                                  │
│  2. During inference, for each MoE layer:                       │
│     a. Router selects top-K experts (on local GPU)              │
│     b. Partition: local_experts, remote_experts                 │
│     c. Compute local experts normally                           │
│     d. If remote_experts > 0:                                   │
│        - Send hidden_states[remote_tokens] to remote            │
│        - Send remote_expert_ids to remote                       │
│        - Remote computes and returns results                    │
│        - Merge results into output                              │
│     e. Continue with next layer                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Option B: ggml-Level Modification (More Complex)

Modify `ggml_mul_mat_id` to support remote experts. This requires:
- New ggml operation: `ggml_mul_mat_id_rpc`
- Custom CUDA kernel that handles the split
- More invasive but potentially more efficient

**Recommendation**: Start with Option A for the experiment.

---

## 3. Detailed Implementation Plan

### Phase 1: Model Preparation (Week 1)

#### 3.1 Split Expert Weights

```python
# Script: split_moe_experts.py
# Input: Full model GGUF
# Output: 
#   - local_model.gguf (all layers, but only experts 0-79)
#   - remote_experts.gguf (only experts 80-127 for all MoE layers)

def split_experts(input_gguf, local_gguf, remote_gguf, local_experts=80):
    """
    For each MoE layer, split:
      - ffn_gate_exps.weight: [n_ff, n_embd, n_expert] → [n_ff, n_embd, local_experts] + [n_ff, n_embd, remote_experts]
      - ffn_up_exps.weight: same
      - ffn_down_exps.weight: same
    """
    # Load model
    model = load_gguf(input_gguf)
    
    local_tensors = {}
    remote_tensors = {}
    
    for name, tensor in model.tensors.items():
        if "ffn_gate_exps" in name or "ffn_up_exps" in name or "ffn_down_exps" in name:
            # This is an expert weight tensor
            # Shape: [dim1, dim2, n_expert]
            local_tensors[name] = tensor[:, :, :local_experts]
            remote_tensors[name] = tensor[:, :, local_experts:]
        else:
            # Non-expert weights go to local only
            local_tensors[name] = tensor
    
    save_gguf(local_gguf, local_tensors)
    save_gguf(remote_gguf, remote_tensors)
```

#### 3.2 Remote Server

```cpp
// Remote server: expert_worker.cpp
// Listens for requests, computes specific experts, returns results

struct ExpertRequest {
    uint32_t layer_id;
    uint32_t n_tokens;
    uint32_t n_experts_requested;  // number of unique experts needed
    float hidden_states[];         // [n_tokens, n_embd]
    int32_t expert_ids[];          // which experts to compute
};

struct ExpertResponse {
    uint32_t n_tokens;
    uint32_t n_experts_requested;
    float results[];               // [n_tokens, n_experts_requested, n_ff]
};

// Main loop
while (true) {
    ExpertRequest req = receive_from_network();
    
    // Load expert weights for this layer (if not cached)
    auto& experts = load_experts_for_layer(req.layer_id);
    
    // Compute only the requested experts
    auto results = compute_experts(req.hidden_states, req.expert_ids, experts);
    
    send_response(ExpertResponse{results});
}
```

### Phase 2: Core Implementation (Week 2)

#### 3.3 Modified MoE Forward Pass

```cpp
// In llama-graph.cpp or a new file llama-moe-rpc.cpp

ggml_tensor* build_moe_ffn_with_rpc(
    ggml_tensor* cur,
    // ... other params ...
    int local_expert_start,
    int local_expert_end,
    rpc_client* remote_expert_server) const {
    
    // 1. Router (runs locally)
    ggml_tensor* logits = build_lora_mm(gate_inp, cur);
    ggml_tensor* probs = ggml_soft_max(ctx0, logits);
    ggml_tensor* selected_experts = ggml_argsort_top_k(ctx0, probs, n_expert_used);
    
    // 2. Partition experts into local and remote
    // This needs to happen at runtime, not graph build time!
    // We need a custom ggml operation or post-processing
    
    // 3. Compute local experts (existing code path)
    ggml_tensor* local_result = compute_local_experts(cur, selected_experts, 
                                                       local_expert_start, local_expert_end);
    
    // 4. Compute remote experts (new code path)
    ggml_tensor* remote_result = nullptr;
    if (has_remote_experts(selected_experts, local_expert_start, local_expert_end)) {
        // This triggers actual network communication during graph execution
        remote_result = compute_remote_experts(cur, selected_experts,
                                                local_expert_start, local_expert_end,
                                                remote_expert_server);
    }
    
    // 5. Merge results
    return merge_local_remote_results(local_result, remote_result, selected_experts,
                                       local_expert_start, local_expert_end);
}
```

#### 3.4 Key Challenge: Static Graph vs Dynamic Routing

**Problem**: llama.cpp builds a static computation graph before inference. But expert routing is dynamic — we don't know which experts will be selected until runtime.

**Solution**: Build the graph to handle the worst case (all experts remote), then use runtime checks to skip unnecessary work.

```cpp
// Alternative: Use ggml's custom operation mechanism
// Register a new operation that handles the RPC internally

struct ggml_tensor* ggml_mul_mat_id_rpc(
    struct ggml_context* ctx,
    struct ggml_tensor* weight,      // local expert weights
    struct ggml_tensor* src,         // input hidden states
    struct ggml_tensor* ids,         // selected expert ids
    rpc_client* remote,              // remote expert server
    int local_expert_start,
    int local_expert_end) {
    
    struct ggml_tensor* result = ggml_new_tensor_3d(ctx, src->type, 
        weight->ne[0], ids->ne[0], src->ne[1]);
    
    // Store RPC info in tensor metadata
    result->op = GGML_OP_MUL_MAT_ID_RPC;
    result->src[0] = weight;
    result->src[1] = src;
    result->src[2] = ids;
    // ... store remote, local_expert_start, local_expert_end ...
    
    return result;
}

// In ggml-cuda.cu, implement the kernel
void ggml_cuda_mul_mat_id_rpc(ggml_tensor* dst) {
    // 1. Determine which selected experts are local vs remote
    // 2. Compute local experts with existing kernel
    // 3. For remote experts:
    //    a. Gather hidden states for tokens needing remote experts
    //    b. Send to remote server
    //    c. Receive results
    //    d. Scatter results back
}
```

### Phase 3: Testing & Benchmarking (Week 3)

#### 3.5 Test Configuration

```bash
# Local machine
./bin/llama-server \
    -m /models/qwopus-35b-local.gguf \
    --rpc-experts 192.168.1.77:50052:80-127 \
    -c 8192 \
    --port 8085

# Remote machine (existing rpc-server, no changes needed if we use custom protocol)
./bin/rpc-server --host 0.0.0.0 --port 50052

# OR: Dedicated expert worker on remote
./bin/expert-worker \
    --model /models/qwopus-35b-remote-experts.gguf \
    --host 0.0.0.0 \
    --port 50053
```

#### 3.6 Benchmark Metrics

| Metric | Target | Baseline (single GPU) |
|--------|--------|----------------------|
| Token generation | >100 tok/s | 140 tok/s |
| Prompt processing | >50 tok/s | 95 tok/s |
| Network usage | <50 KB/token | N/A |
| Memory (local) | <24 GB | 21.4 GB |
| Memory (remote) | <24 GB | N/A |

---

## 4. Simplified Experiment: "Poor Man's Expert Parallelism"

If the full implementation is too complex, here's a simpler approach:

### 4.1 Concept

Instead of modifying llama.cpp, use **two separate llama-server instances** with a custom router:

```
                    ┌─────────────────────────────────────────┐
                    │           Custom Router (Python)         │
                    │                                          │
User ──────────────►│  1. Receive prompt                       │
                    │  2. Send to local llama-server           │
                    │  3. Get intermediate hidden states       │
                    │  4. For MoE layers, route to remote      │
                    │  5. Merge and continue                   │
                    └─────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    Local llama-server    Remote llama-server
    (experts 0-79)        (experts 80-127)
```

### 4.2 Even Simpler: Model Splitting with Manual Routing

```python
# Simple experiment: manually split model and measure
# This doesn't require modifying llama.cpp at all!

# Step 1: Create two model files
# - model_local.gguf: all layers, experts 0-79 have real weights, 80-127 are zeros
# - model_remote.gguf: only expert weights for 80-127

# Step 2: Run local inference, intercept at MoE layers
# Step 3: For tokens that need remote experts, call remote server
# Step 4: Merge results

# This can be done with llama.cpp's Python bindings!
```

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Network latency too high | Medium | High | Use 10Gbps if available; batch remote calls |
| Graph build complexity | High | Medium | Start with "poor man's" approach |
| Expert load imbalance | Low | Medium | Dynamic expert placement based on routing stats |
| Memory overhead | Low | Low | Only cache active expert weights |

---

## 6. Success Criteria

### Minimum Viable Experiment (Week 1-2)
- [ ] Successfully split a MoE model into local/remote expert files
- [ ] Remote server can load and serve specific experts
- [ ] Basic test: single token generation works correctly

### Full Experiment (Week 3-4)
- [ ] Token generation >100 tok/s for 40GB+ model
- [ ] Memory usage <24GB on each machine
- [ ] No accuracy degradation vs single-machine inference

### Production Ready (Future)
- [ ] Support for multiple remote machines
- [ ] Dynamic expert placement based on routing patterns
- [ ] Integration with existing llama.cpp RPC protocol

---

## 7. Next Steps

1. **Immediate**: Verify model architecture (how many experts, which layers are MoE)
2. **Day 1**: Write expert splitting script
3. **Day 2-3**: Implement basic remote expert server
4. **Day 4-5**: Integrate with llama-server
5. **Day 6-7**: Benchmark and iterate

---

## Appendix: Model Architecture Analysis

Need to check:
```bash
# Check model architecture
./bin/llama-cli --model /models/Qwopus3.6-35B-A3B-v1-TQ3_4S.gguf --verbose-prompt 2>&1 | grep -i expert

# Expected output:
# n_expert = 128
# n_expert_used = 3 (or similar)
# moe layers: every N layers or specific layer indices
```

---

*Document created: 2026-06-09*
*Status: Planning*
*Author: Collaborative experiment design*

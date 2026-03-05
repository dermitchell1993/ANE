# ANE 7B–13B Production Roadmap for OpenClaw

**Target**: Production-grade fine-tuning and inference of 7B–13B class LLMs on 24 GB M2 MacBook, ANE-only compute (GPU/CPU for orchestration only).

**Baseline**: Stories110M — 12-layer, dim=768, hidden=2048, 109M params, 91–106 ms/step on M3/M4.

**Architecture Constraint**: Apple Neural Engine 16-core, ~15.8 FP16 TFLOPS (M2 rated), 24 GB unified memory (20 GB usable budget).

---

## Memory Budget — M2 24 GB

| Component | 7B Q4 (group128) | 13B Q3 (group64) |
|---|---|---|
| Model weights (packed) | 3.5 GB | 4.9 GB |
| Dequant buffer (2 layers fp16) | 0.44 GB | 0.78 GB |
| KV cache (seq=2048, fp16) | 1.0 GB | 1.6 GB |
| Activation scratch (1 layer) | 0.34 GB | 0.52 GB |
| IOSurface pool (6 kernels) | 0.25 GB | 0.35 GB |
| Python/OS/bridge overhead | 1.5 GB | 1.5 GB |
| **Total** | **7.03 GB** | **9.65 GB** |
| **Headroom to 20 GB cap** | **12.97 GB** | **10.35 GB** |
| LoRA adapters (rank-16, fp16) | +0.10 GB | +0.15 GB |
| Gradient checkpoints (LoRA) | +0.80 GB | +1.20 GB |
| **Total w/ LoRA training** | **7.93 GB** | **11.00 GB** |

### Derivation

**7B Q4 weights**: 6.7B params × 4 bits / 8 = 3.35 GB + scales (group128: 6.7B/128 × 2B = 105 MB) ≈ 3.5 GB

**KV cache (7B)**: 32 layers × 2 (K+V) × 32 heads × 128 head_dim × 2048 seq × 2 bytes = 1.07 GB

**Activation scratch**: max(dim × seq × fp16, hidden × seq × fp16) per kernel invocation. 7B: max(4096 × 2048 × 2, 11008 × 2048 × 2) ≈ 43 MB per buffer, ~8 buffers active = 344 MB

---

## Estimated Performance — M2 ANE

| Metric | 7B Q4 | 13B Q3 | Notes |
|---|---|---|---|
| **Inference tok/s (prefill, seq=512)** | 18–25 | 8–12 | Bound by ANE matmul throughput |
| **Inference tok/s (decode, seq=1)** | 12–18 | 5–9 | Bound by weight load bandwidth |
| **Training tok/s (LoRA, seq=256)** | 3–6 | 1–3 | Forward + LoRA backward |
| Weight dequant per layer (NEON) | 0.8 ms | 1.2 ms | Q4→fp16 via vld1q + shift |
| ANE kernel exec per layer | 1.5–2.5 ms | 2.5–4.0 ms | Fused SDPA + FFN |
| Weight reload per layer | 0.2–0.5 ms | 0.3–0.6 ms | IOSurface rewrite, no recompile |
| Full forward pass (32/40 layers) | 55–90 ms | 120–200 ms | Sum of per-layer times |
| ANE utilization estimate | 25–40% | 20–35% | Limited by weight I/O latency |

### Derivation

**ANE matmul throughput**: M2 measured at ~10 FP16 TFLOPS sustained (extrapolated from M3 Pro at 15 TFLOPS, M2 rated lower). 7B forward FLOPs per token ≈ 2 × 6.7B = 13.4 GFLOP. At 10 TFLOPS: 13.4/10000 = 1.34 ms compute. With overhead: 2–3 ms.

**Decode bottleneck**: At decode (seq=1), each layer requires loading ~110 MB fp16 weights into ANE. M2 unified memory bandwidth: 100 GB/s. Load time: 110MB / 100GB/s = 1.1 ms per layer. At 32 layers: 35.2 ms minimum → ~28 tok/s theoretical ceiling. Realistic with overheads: 12–18 tok/s.

---

## Phase 1 — Stable Multi-Layer Stacking (Days 1–3)

### Goal
Prove the weight-swap architecture at 32-layer scale. Validate that a single compiled MIL kernel can serve all layers by reloading weights between invocations.

### 1.1 Parameterized Kernel Generator

**Current state**: `stories_mil.h` generates MIL with hardcoded `DIM=768`, `HIDDEN=2048`, `HEADS=12`.

**Deliverable**: `mil_gen_llama.h` — parameterized MIL generator accepting `(dim, hidden_dim, n_heads, n_kv_heads, head_dim, max_seq)`.

**New/Modified MIL ops required**:
| Op | Status | Notes |
|---|---|---|
| `conv` (1×1) | ✅ Exists | Linear projections (Q/K/V/O/W1/W2/W3) |
| `matmul` | ✅ Exists | Attention scores, output |
| `softmax` | ✅ Exists | Attention weights |
| `reduce_sum` | ✅ Exists | RMSNorm |
| `pow` | ✅ Exists | RMSNorm (rsqrt via pow(-0.5)) |
| `mul` / `add` / `sub` | ✅ Exists | Element-wise |
| `reshape` / `transpose` | ✅ Exists | Head reshaping |
| `slice_by_size` | ✅ Exists | Tensor slicing |
| `concat` | ✅ Exists | Forward taps |
| `cast` (fp32↔fp16) | ✅ Exists | I/O conversion |
| **`sin` / `cos`** | 🆕 **NEW** | RoPE embedding |
| **`gather`** | 🆕 **NEW** | RoPE frequency lookup (if table approach) |

### 1.2 RoPE as MIL Op

**Architecture**: Precompute `cos_table[max_seq, head_dim/2]` and `sin_table[max_seq, head_dim/2]` as `const()` tensors baked into the SDPA kernel. Apply after Q/K projection, before attention.

**MIL implementation** (inside fused SDPA kernel):
```
// After Q projection: q [1, n_heads, seq, head_dim]
// Split even/odd pairs
q_even = slice(q, begin=[0,0,0,0], size=[1,H,S,HD/2], stride=[1,1,1,2])
q_odd  = slice(q, begin=[0,0,0,1], size=[1,H,S,HD/2], stride=[1,1,1,2])
// cos_tab, sin_tab: [1, 1, max_seq, HD/2] baked constants
q_rot_even = sub(mul(q_even, cos_tab), mul(q_odd, sin_tab))
q_rot_odd  = add(mul(q_even, sin_tab), mul(q_odd, cos_tab))
// Interleave back
q_rotated = concat(q_rot_even, q_rot_odd, interleave=true)
```

**Fallback**: If ANE rejects strided slice, use reshape to `[1, H, S, HD/2, 2]` → split on last axis → rotate → reshape back. MIL `split` + `concat` with explicit shapes.

### 1.3 Dynamic Layer Count via Weight Swap

**Critical experiment**: Validate `unload → rewrite weight.bin → load` at 7B layer scale.

**Deliverable**: `test_weight_swap_7b.m` — compile one SDPA kernel with dim=4096 shapes, time the weight swap cycle with realistic data sizes.

**Go/no-go gate**: If weight swap takes >10 ms/layer, abandon layer iteration — fall back to chunked compilation (compile 30 layers, exec() restart, compile remaining).

### 1.4 Python Bridge (`bridge/ane_model.py`)

**Deliverable**: ctypes wrapper around `libane_bridge.dylib`:

```python
class ANEModel:
    def __init__(self, config: ModelConfig)
    def compile_kernels(self) -> None
    def load_layer_weights(self, layer_idx: int, weights: dict[str, np.ndarray]) -> None
    def forward(self, input_ids: np.ndarray) -> np.ndarray  # logits
    def decode_token(self, token_id: int, pos: int) -> np.ndarray  # single token
```

**Model loading**: Parse GGUF or SafeTensors via pure-Python readers (no HuggingFace dependency). Extract per-layer weight tensors, convert to fp16, pack into ANE blob format.

### 1.5 Residual Add + RMSNorm Fusion

**Current state**: Residual add is on CPU (`for (int i = 0; i < S * d; i++) x[i] += o_out[i]`). RMSNorm is fused into forward kernels.

**Deliverable**: Fuse residual add into the FFN kernel input:
```
// FFN kernel now takes TWO inputs: x_residual and attn_output
// First op: x = add(x_residual, attn_output)
// Then: rmsnorm(x) → FFN → output + ffn_taps
```

This eliminates one CPU→ANE→CPU round-trip per layer.

### Phase 1 Files

| File | Description |
|---|---|
| `roadmap/mil_gen_llama.h` | Parameterized MIL generator for LLaMA-family models |
| `roadmap/test_weight_swap_7b.m` | Weight swap benchmark at 7B layer dimensions |
| `bridge/ane_model.py` | Python ctypes wrapper for model loading + inference |
| `bridge/gguf_reader.py` | GGUF format parser (weights + metadata) |
| `roadmap/rope_mil_gen.h` | RoPE embedding as MIL constant tables |

---

## Phase 2 — Production Op Coverage + Quantization (Days 4–10)

### 2.1 Full SwiGLU on ANE

**Current state**: SiLU activation is on CPU: `silu_f(h1) * h3`.

**Deliverable**: Fused SwiGLU inside FFN kernel:
```
h1 = conv(W1, xnorm)    // gate projection
h3 = conv(W3, xnorm)    // up projection
// SiLU(h1) = h1 * sigmoid(h1)
neg_h1 = mul(h1, const(-1))
exp_neg = exp(neg_h1)    // MIL 'exp' op — NEW, verify ANE support
one_plus = add(exp_neg, const(1.0))
sigmoid = real_div(const(1.0), one_plus)  // MIL 'real_div' op
silu = mul(h1, sigmoid)
swiglu = mul(silu, h3)
ffn_out = conv(W2, swiglu)
```

**New MIL ops needed**:
| Op | Status | Notes |
|---|---|---|
| `exp` | 🔍 **VERIFY** | May compile but fall back to CPU |
| `real_div` | 🔍 **VERIFY** | Scalar/tensor division |
| `sigmoid` | 🔍 **VERIFY** | Native MIL op, likely ANE-native |

**Fallback**: If `exp` doesn't run on ANE, approximate SiLU with a piecewise polynomial: `silu(x) ≈ x * clamp(0.5 + 0.25*x, 0, 1)` for |x| < 4. Known to produce <1% error for typical activation ranges.

### 2.2 GQA (Grouped Query Attention)

**Current state**: Assumes `n_kv_heads == n_heads`. Stories110M uses MHA (12 Q = 12 KV).

**Deliverable**: Parameterized SDPA kernel supporting `n_kv_heads < n_heads`:

```
// Q: [1, n_heads, seq, head_dim]
// K: [1, n_kv_heads, seq, head_dim]
// V: [1, n_kv_heads, seq, head_dim]
// Repeat K/V to match Q head count
group_size = n_heads / n_kv_heads  // e.g., 4 for 32Q/8KV
// tile(K, reps=[1, group_size, 1, 1]) → [1, n_heads, seq, head_dim]
K_expanded = tile(x=K, reps=tile_reps)
V_expanded = tile(x=V, reps=tile_reps)
// Then standard SDPA
scores = matmul(Q, transpose(K_expanded)) * scale
```

**MIL ops needed**:
| Op | Status | Notes |
|---|---|---|
| `tile` | 🆕 **NEW** | Repeat K/V for GQA head expansion |

**Alternative**: If `tile` isn't supported, use `concat` of the same tensor repeated `group_size` times. Less elegant but guaranteed to work.

### 2.3 Causal Masking (Already Working)

The existing SDPA kernel bakes a causal mask as a `const()` BLOBFILE: upper-triangle filled with -65504 (fp16 -inf). This scales to any sequence length by regenerating the blob. No changes needed beyond parameterizing `max_seq`.

### 2.4 4-bit / 8-bit Weight Packing

**Architecture decision**: Dequant on CPU during weight reload, NOT inside ANE kernel.

**Rationale**:
- ANE MIL has no native INT4 type
- Building custom dequant in MIL would require `bit_shift` / `bitwise_and` ops that may not exist
- CPU-side NEON dequant of one 7B layer (55MB Q4 → 110MB fp16) takes <1ms
- Keeps ANE kernels pure fp16 — proven, debugged, cross-generation compatible

**Storage format** (custom `.anepak`):
```
Header:
  magic: "ANEP"
  version: 1
  n_layers: uint32
  quant_type: uint8 (0=fp16, 1=q8, 2=q4, 3=q3)
  group_size: uint32
Per-layer:
  offset: uint64
  compressed_size: uint64
  scales: fp16[n_groups]
  zeros: fp16[n_groups]  (asymmetric quant)
  data: uint8[compressed_size]  (packed nibbles for Q4)
```

**Dequant kernel** (NEON intrinsics):
```c
// Q4 group dequant: 32 weights per NEON pass
// Input: 16 bytes (32 nibbles) + 1 fp16 scale + 1 fp16 zero
// Output: 32 fp16 values
void dequant_q4_group_neon(const uint8_t *src, _Float16 scale, _Float16 zero,
                            _Float16 *dst, int group_size);
```

### 2.5 LoRA Adapter Weights

**Architecture**: LoRA adapters stored as separate constant blobs in the MIL program.

For each adapted projection (Q, K, V, O per layer):
```
// Original: y = conv(W, x)
// LoRA:     y = conv(W, x) + conv(B, conv(A, x))
// Where A: [rank, dim, 1, 1], B: [dim, rank, 1, 1]
// Bake A and B as additional const() weight blobs
```

**MIL modification**:
```
// After main projection conv
tensor<fp16, [rank, dim, 1, 1]> Aq = const()[name=string("lora_a_q"), ...BLOBFILE...];
tensor<fp16, [dim, rank, 1, 1]> Bq = const()[name=string("lora_b_q"), ...BLOBFILE...];
tensor<fp16, [1, rank, 1, S]> lora_down = conv(..., weight=Aq, x=xn);
tensor<fp16, [1, dim, 1, S]> lora_up = conv(..., weight=Bq, x=lora_down);
tensor<fp16, [1, dim, 1, S]> q_final = add(x=q_base, y=lora_up);  // merge
```

**Training**: Freeze base weights. Only accumulate gradients for A and B matrices. At rank-16 for 7B:
- Per projection: 4096 × 16 × 2 × 2 = 256 KB
- Per layer (QKVO): 1 MB
- Total (32 layers): 32 MB trainable params

### 2.6 Custom Dequant Kernel on ANE (Experimental)

If CPU dequant becomes the bottleneck, attempt an ANE-native dequant:

```
// Pack Q4 weights as uint8 tensor, bake scale/zero as fp16 tensors
// MIL: cast uint8 → fp16, then bit-extract
tensor<uint8, [1, N/2, 1, 1]> packed = const()[...];
tensor<fp16, [1, N/2, 1, 1]> packed_f = cast(dtype="fp16", x=packed);
// Low nibble: mod(packed_f, 16)
// High nibble: floor_div(packed_f, 16)
// Dequant: (nibble - 8) * scale
```

**Risk**: ANE may not support `uint8` inputs or the `mod`/`floor_div` ops. This is experimental — CPU dequant is the safe path.

### Phase 2 Files

| File | Description |
|---|---|
| `roadmap/quant_pack.h` | Q4/Q8 weight packing + NEON dequant routines |
| `roadmap/quant_pack.py` | Python quantization: fp16/fp32 model → .anepak |
| `roadmap/lora_mil_gen.h` | LoRA-augmented MIL kernel generator |
| `roadmap/safetensors_reader.py` | SafeTensors format parser |

---

## Phase 3 — Swarm-Ready Production (Days 11–21)

### 3.1 Zero-Downtime Hot-Reload

**Problem**: Current architecture requires `exec()` restart to overcome the ~119 compile limit. For a 24/7 bot node, this means downtime.

**Solution**: Weight-only reload without recompile.

**Implementation**:
1. Compile the ~6 kernel programs ONCE at startup (SDPA, FFN, FFN_bwd, SDPA_bwd1, SDPA_bwd2, QKV_bwd)
2. For model swaps: unload current weights → write new weights to tmpDir → load
3. If weight reload doesn't work (per Phase 1 gate test): maintain a pool of 2 compiled model instances, swap atomically

**Bridge API**:
```c
// Hot-reload: swap model weights without recompile
int ane_bridge_hot_reload(ANEKernelHandle *kernel,
                          const char **weight_names,
                          const uint8_t **weight_datas,
                          const size_t *weight_lens,
                          int n_weights);
```

### 3.2 Telemetry (JouleWork-Style)

**Deliverable**: `bridge/telemetry.h` — per-invocation metrics:

```c
typedef struct {
    double wall_ms;        // Wall clock time
    double ane_ms;         // ANE compute time (from perfStats)
    double cpu_ms;         // CPU overhead (wall - ane)
    double dequant_ms;     // Weight dequant time
    double reload_ms;      // Weight reload time
    uint64_t tokens;       // Tokens processed
    double tok_per_sec;    // Throughput
    double watts_ane;      // ANE power draw (via IOReport)
    double watts_cpu;      // CPU power draw
    double joules_per_tok; // Energy efficiency
    size_t mem_used_mb;    // Current unified memory usage
    size_t mem_peak_mb;    // Peak unified memory usage
} ANETelemetry;
```

**Power measurement**: Use `IOReport` framework (private but widely used by powermetrics/asitop):
```c
// Sample ANE power channel from IOReport
CFDictionaryRef channel = IOReportCopyChannelsInGroup(
    CFSTR("Energy Model"), NULL, NULL, NULL, NULL);
```

### 3.3 Inference Server Mode

**Deliverable**: `bridge/serve.py` — HTTP/WebSocket inference endpoint.

```python
# HTTP endpoint
POST /v1/completions
{
    "prompt": "Once upon a time",
    "max_tokens": 256,
    "temperature": 0.7
}

# WebSocket endpoint for streaming
WS /v1/stream
→ {"token": "Once", "logprob": -0.5, "latency_ms": 45.2}
→ {"token": " upon", "logprob": -0.3, "latency_ms": 42.1}
```

**Implementation**: `asyncio` + `websockets` (no heavy framework). Single-threaded decode loop with async I/O for network.

### 3.4 Memory Safety (20 GB Cap)

**Implementation**: Memory watchdog thread:

```c
// Check unified memory usage every 100ms
// If approaching 20 GB: reduce KV cache, shorten sequence length
typedef struct {
    size_t hard_cap_mb;     // 20480 (20 GB)
    size_t soft_cap_mb;     // 18432 (18 GB — start evicting)
    size_t current_mb;
    int seq_len_current;    // Dynamic: starts at max, shrinks under pressure
    int seq_len_min;        // Never go below 256
    bool checkpoint_active; // Activation checkpointing enabled
} MemoryGuard;
```

**Dynamic sequence length**: If memory pressure exceeds soft cap, halve the KV cache (discard oldest positions) and reduce effective seq_len. Log a warning to telemetry.

**Activation checkpointing**: For LoRA training, only store activations for 2 layers at a time. Recompute intermediate activations during backward pass. Trades compute for memory: ~3× per-layer computation but ~16× memory savings on activations.

### 3.5 Crash Recovery + Watchdog

**Deliverable**: `bridge/watchdog.py` — systemd-compatible watchdog for 24/7 operation.

```python
class ANEWatchdog:
    def __init__(self, model_path, config):
        self.heartbeat_interval = 5.0  # seconds
        self.max_restart_count = 10
        self.restart_backoff = [1, 2, 4, 8, 16, 32, 60]  # seconds

    def run(self):
        while True:
            proc = self.spawn_inference_server()
            self.monitor(proc)  # blocks until crash or shutdown
            if self.should_restart():
                self.save_state()  # persist KV cache + position
                self.restart_with_backoff()
```

**State persistence**: On clean shutdown or crash recovery, serialize:
- Current KV cache (mmap'd file for instant reload)
- Token position counter
- LoRA adapter weights (if training in progress)
- Telemetry accumulator

### 3.6 OpenClaw Swarm Integration

**How each "claw" registers its ANE model**:

```python
# OpenClaw skill manifest (in claw's skill.yaml)
name: "ane-inference"
type: "llm-endpoint"
capabilities:
  - text-generation
  - text-completion
hardware:
  accelerator: "ane"
  memory_gb: 24
  chip: "m2"
model:
  name: "llama-7b-q4"
  format: "anepak"
  max_seq: 2048
  quant: "q4_group128"
endpoint:
  http: "http://localhost:8741/v1/completions"
  ws: "ws://localhost:8741/v1/stream"
telemetry:
  report_interval: 10  # seconds
  metrics: ["tok_per_sec", "mem_used_mb", "watts_ane", "joules_per_tok"]
```

**Registration flow**:
1. Claw starts → loads model via `bridge/ane_model.py`
2. Compiles kernels (one-time, ~30s for 6 kernel programs)
3. Starts inference server on local port
4. Registers with swarm controller via Lobster workflow:
   ```
   lobster register-skill --manifest skill.yaml --node-id $(hostname)
   ```
5. Swarm controller routes inference requests to available claws
6. Each claw reports telemetry every `report_interval` seconds
7. Swarm controller load-balances based on current tok/s and memory pressure

**Multi-claw coordination**:
- Different claws can host different models (7B on 24GB M2, 13B on 64GB M2 Max)
- Swarm controller picks the best available claw for each request based on model match + current load
- If a claw crashes, watchdog restarts it; swarm controller re-routes traffic to surviving claws

### Phase 3 Files

| File | Description |
|---|---|
| `bridge/telemetry.h` | C telemetry struct + IOReport power sampling |
| `bridge/telemetry.py` | Python telemetry wrapper |
| `bridge/serve.py` | HTTP/WS inference server |
| `bridge/watchdog.py` | Crash recovery + systemd watchdog |
| `bridge/memory_guard.h` | Memory pressure monitor + dynamic seq_len |
| `bridge/openclaw_manifest.yaml` | OpenClaw skill manifest template |

---

## Kernel Fusion Summary — All Phases

### Forward Kernels (inference)

| Kernel | Ops Fused | Weights Baked | I/O |
|---|---|---|---|
| **k_sdpa** | RMSNorm → Q/K/V conv → RoPE → reshape → matmul → scale → mask → softmax → matmul → reshape → O conv | rms_att, Wq, Wk, Wv, Wo, cos_tab, sin_tab, causal_mask | In: x [dim,S] → Out: o_out [dim,S] + taps |
| **k_ffn** | residual_add → RMSNorm → W1 conv → W3 conv → SiLU → mul → W2 conv | rms_ffn, W1, W2, W3 | In: x_res [dim,S] + attn_out [dim,S] → Out: ffn_out [dim,S] + taps |
| **k_cls** | conv (embed @ x) | embed [vocab,dim] | In: x [dim,S] → Out: logits [vocab,S] |
| **k_softmax** | softmax(axis=1) | — | In: logits [vocab,S] → Out: probs [vocab,S] |
| **k_rmsnorm_final** | RMSNorm | rms_final | In: x [dim,S] → Out: xnorm [dim,S] |

### Backward Kernels (LoRA training)

| Kernel | Ops Fused | Weights Baked |
|---|---|---|
| **k_lora_bwd** | LoRA A/B gradient accumulation | Base W (frozen), A, B |
| **k_ffn_bwd** | W2^T + SiLU_bwd + W1^T + W3^T | W2^T, W1^T, W3^T |
| **k_sdpa_bwd1** | Wo^T + SDPA backward part 1 | Wo^T, mask |
| **k_sdpa_bwd2** | SDPA backward part 2 | — |
| **k_qkv_bwd** | Wq^T + Wk^T + Wv^T | Wq^T, Wk^T, Wv^T |
| **k_rmsnorm_bwd** | RMSNorm backward | rms_w |

**Total compiled kernel programs**: 11 (6 forward + 5 backward). Well under the ~119 compile limit. All layers share the same compiled programs, swapping only weights.

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|---|---|---|---|
| Weight reload doesn't work at 7B scale | Architecture dead | Medium | Phase 1 gate test. Fallback: chunked compilation |
| M2 channel constraint (like M3 Pro ch=512) | Can't do dim=4096 in one kernel | Medium | Tile into 8× ch=512 sub-convolutions |
| ANE rejects `sin`/`cos` MIL ops | No on-ANE RoPE | Low | Precompute rotated Q/K on CPU, pass as kernel input |
| `exp` op falls back to CPU inside ANE kernel | Slow SiLU | Medium | Polynomial SiLU approximation |
| 24 GB not enough for 13B Q3 + training | Can't train 13B | Low | Reduce to inference-only for 13B; train only 7B |
| Private API changes in macOS 27 | Everything breaks | Medium | Pin macOS version; contribute to public API advocacy |
| ANE compile limit changes | More/fewer kernels allowed | Low | Weight-swap architecture is limit-independent |

---

## Decision Gates

| Gate | Day | Criteria | Pass → | Fail → |
|---|---|---|---|---|
| **G1: Weight swap speed** | Day 2 | <10 ms/layer at dim=4096 | Continue Phase 1 | Pivot to chunked compilation |
| **G2: RoPE on ANE** | Day 3 | sin/cos compile + correct output | Full on-ANE forward | CPU RoPE fallback (acceptable) |
| **G3: 7B forward pass** | Day 8 | End-to-end correct logits | Proceed to server mode | Debug kernel accuracy |
| **G4: 7B decode speed** | Day 10 | >10 tok/s on M2 | Production viable | Investigate speculative decoding |
| **G5: 24/7 stability** | Day 18 | 72h continuous run, 0 crashes | Ship it | Harden watchdog + memory guard |


# ANE Training Cheatsheet

Everything we know about running training workloads on Apple Neural Engine via CoreML MIL kernels compiled at runtime. All findings from direct probing on M1 / M1 Pro, macOS 26.x. No Apple documentation exists for any of this.

---

## The Model

ANE is a fixed-function matrix/vector engine. It runs CoreML models — not arbitrary compute shaders. We generate MIL (Model Intermediate Language) text at runtime, compile via `modelWithMILText:`, and execute with `evaluateWithModel:`. Each unique tensor shape needs its own compiled kernel.

Weights are **runtime IOSurface inputs** — no baked constants, no recompilation when weights change. Compile once at startup, train forever.

---

## Tensor Layout

- **Use 3D tensors `[1, M, N]`** — NOT 4D. Two runtime 4D inputs fail.
- **`[1, C, S]` layout** (channel-first): standard for most ops (LN, matmul, GELU, add, sigmoid)
- **`[1, S, C]` layout** (spatial-first): required for depthwise broadcast `[1,S,C] * [1,1,C]`
- These two layouts are **incompatible** — no ANE transpose op exists
- Layout conflict: depthwise needs `[S,C]`, LayerNorm needs `[C,S]` → CPU copy between them

---

## IOSurface Rules

### Slot size ordering (CRITICAL — violating = silent zeros)
```
Inputs:  ioInputs[0].bytes  ≤  ioInputs[1].bytes  ≤  ...   (ascending)
Outputs: ioOutputs[0].bytes  ≥  ioOutputs[1].bytes  ≥  ...   (descending)
```
No error, no warning — just wrong output (zeros).

### Minimum buffer size
~2048 bytes. Tensors smaller than this (e.g. `W[32,3]` = 192 bytes) must fall back to CPU.

**Exception**: depthwise conv shifted-copies pattern (ane_dw.h) has NO 2048 minimum when slot rule is satisfied and sizes are exact.

### Multi-output constraint
2+ outputs fail when output surfaces are much smaller than input surfaces with mixed input sizes. Keep dW kernels as separate dispatches.

### Max inputs
Tested up to **2×K×K = 98 inputs** (depthwise 7×7) — works. 4 runtime inputs confirmed for matmul chains.

---

## Matmul

### What works
```
matmul(W[1,Cout,Cin], X[1,Cin,S]) → Y[1,Cout,S]
       ↑ slot0 (weight)  ↑ slot1 (activation)
```
- **W MUST be slot0, X MUST be slot1** — this is semantic, not just size-based
- Any matmul where the "data" tensor is in slot0 silently returns zeros
- Slot rule: `Cout×Cin ≤ Cin×S` → `Cout ≤ S` (usually true)
- Tested sizes: C=96 Co=384 S=1024 ✅, C=384 Co=1536 S=1024 ✅, C=192 S=4096 ✅

### Dimension constraints
- **Ci (inner/contraction dim) must be a multiple of 32** — Ci=16,48,80,112 give eval=0
- Ci=32,64,96,128,192,384... all work ✓
- Co and S have no observed alignment requirement
- Sparse weight matrices (mostly zeros) may give incorrect results due to tiling alignment

### What fails
- `matmul(X[C,S], P[S,S4])` with P in slot1 → compiles, **eval=0** (zeros)
- Matmul with baked `const` tensor as an operand → InvalidMILProgram
- `[S,C] @ [C,1]` at real sizes (C=96, S≥32) → eval=0
- `[S,C] @ [C,S4]` → slot rule violated OR eval=0

---

## Broadcast Rules

### What works ✅
- `[1,C,S] * [1,1,S]` — broadcast scalar-per-spatial-position over channels
- `[1,S,C] * [1,1,C]` — broadcast scalar-per-channel over spatial positions
  - **CRITICAL**: large tensor `[1,S,C]` MUST be slot0, small `[1,1,C]` slot1
  - This reverses the usual slot ordering — use DECREASING size order for dw-style inputs

### What fails ❌
- `[1,C,S] * [1,C,1]` — per-channel scaling in C-first layout → **ALWAYS eval=0**
- `[1,C,S] * [1,1,1]` scalar broadcast — compile ok, **wrong results**
- `[C,1,S] * [C,1,1]` — batch-dim broadcast → eval=0
- Any broadcast where the small tensor is in slot0 (in non-dw pattern) → may eval=0

---

## MIL Ops That Work on ANE

| Op | MIL name | Notes |
|----|----------|-------|
| Matrix multiply | `matmul` | 3D only, W=slot0 X=slot1, Ci multiple of 32 |
| Add | `add` | Elementwise, broadcast `[1,1,S]→[1,C,S]` OK |
| Subtract | `sub` | Elementwise, broadcast OK |
| Multiply | `mul` | Elementwise, see broadcast rules above |
| Divide | `real_div` | Elementwise, see sqrt caveats below |
| Sigmoid | `sigmoid` | Works in fusions |
| Tanh | `tanh` | Used in GELU approximation |
| Square root | `sqrt` | Works alone, **broken in fusions** (see below) |
| Const | `const` | Baked scalars/tensors (eps, coefficients) |
| Conv (baked W) | `conv` | W as MIL const — works for all conv types |
| Conv (runtime W, groups=1) | `conv` | ⚠️ Works for small sizes only |

### Fusions that work
- `matmul → add → sigmoid → mul → matmul → add` (full FFN+SiLU+residual, 1 dispatch)
- `sub → mul(x,x) → matmul` (LN variance path, 3 ops fused)
- `add → mul`, `sub → mul`, `mul → add` (any chain of basic arithmetic)
- `sigmoid(x) * x` = SiLU (2 ops)
- `tanh`-based GELU approximation (9 ops in one kernel)

## MIL Ops That Do NOT Work

| Op | Status |
|----|--------|
| `rsqrt` | **Not supported** — compile fails |
| `pow` | **Not supported** — compile fails |
| `reduce_sum` / `reduce_mean` / `reduce_max` | **Fails** — no reduction ops on ANE |
| `reshape`, `transpose` | **Fails** — InvalidMILProgram |
| `slice_by_size`, `slice_by_index` | **Fails** |
| `gather`, `gather_nd` | **Fails** |
| `concat`, `split`, `pad`, `crop` | **Fails** |
| `conv` with runtime W (depthwise, groups=C) | **Fails** — InvalidMILProgram |
| `relu` | ⚠️ Works but only for small N |
| `gelu` (flat `[1,N]`) | ❌ Fails at N>16384; use `[1,C,S]` 3D instead |

**Bottom line**: ANE does matmul + elementwise. No reductions, no indexing, no reshaping, no grouped conv with runtime weights.

---

## The sqrt Bug

sqrt works alone. sqrt breaks in fusions involving broadcast or division.

| Pattern | Result |
|---------|--------|
| `sqrt(A)` alone | ✅ Correct |
| `add(A,B) → sqrt` | ✅ Correct |
| `sqrt → mul` same-size `[1,1,S]` | ✅ Correct |
| `sqrt → mul` broadcast `[1,1,S] → [1,C,S]` | ❌ Only c=0 gets a value, rest zeros |
| `sqrt(A) → real_div(sqrt, B)` | ❌ Computes `B/sqrt(A)` instead of `sqrt(A)/B` |
| `sqrt → real_div(ONE, sqrt)` in fusion | ❌ Compile fails |
| `add → sqrt → real_div(DIFF, sqrt)` | ❌ Wrong values |

**Root cause**: sqrt output cannot participate in broadcast. Likely a hardware routing limitation — sqrt uses a dedicated functional unit (Newton-Raphson iterator) whose output doesn't feed back into the broadcast-capable datapath.

**Impact on LayerNorm**: Cannot fuse `sqrt → reciprocal → broadcast multiply` into one kernel. The 5-dispatch LN decomposition is the hard limit.

---

## MIL Variable Names

Reserved/problematic names (cause silent eval=0 or compile failure):
- ❌ `var`, `diff`, `mean`, `std`, `eps`, `x`, `y`, `mul`, `add`, `rsqrt`, `c`, `C`
- ✅ Use descriptive uppercase: `MN`, `VR`, `DIFF`, `RSTD`, `SQ`, `INP`, `X3`, `ARG`, `TH`

---

## GELU

- Must use 3D `[1,C,S]` tensor — NOT flat `[1,N]` when `N > ~16384`
- Implemented as tanh approximation: 9 MIL ops in one kernel
- CPU fallback for backward and for tensors > ~50K elements

---

## LayerNorm Decomposition

ANE has no reduction ops. The trick: `matmul([1,1,C] ones, [1,C,S] x)` = sum over C → `[1,1,S]`. This ONLY works in `[C,S]` layout.

```
                         ┌─────────────────────────────────────────┐
                         │  k1b_2_3a (FUSED: sub + square + matmul) │
   x[C,S] ──┬── k1a ──►│  MN[1,S] ──►  DIFF = x - mean           │
             │  (mean    │  x[C,S]  ──►  SQ = DIFF²                │
             │  matmul)  │  W[1,S]  ──►  VAR = W @ SQ              │
             │           └──────────┬──────────────────┬────────────┘
             │                      │ DIFF[C,S]        │ VAR[1,S]
             │                      │                  ▼
             │                      │            k3b: add(VAR,EPS) → sqrt
             │                      │                  │ SQRTV[1,S]
             │                      │                  ▼
             │                      │            k3c: real_div(1, SQRTV)
             │                      │                  │ RSTD[1,S]
             │                      ▼                  ▼
             │                   k4: mul(DIFF, RSTD) → NORM[C,S]
             │
             └── (also used by k1b_2_3a via rewire)
```

**5 ANE dispatches** (was 7 before fusing k1b+k2+k3a). Can't fuse further due to sqrt broadcast bug.

**Performance**: ~1.3ms at C=96 S=1024, ~1.7ms at S=16384.

**CPU threshold**: When `C × S > 50K`, CPU NEON LayerNorm is faster than 5 ANE dispatches.

---

## Depthwise Conv on ANE (ane_dw.h)

Runtime depthwise conv with `groups=C` fails as a `conv` op. Workaround: decompose into shifted copies + broadcast mul + add chain.

Single kernel with `2×K×K` inputs in `[1,S,C]` layout:
- Slots 0..KK-1: shifted input copies `K_k[1,S,C]` (large — NO 2048 minimum)
- Slots KK..2KK-1: weight slices `W_k[1,1,C]` (small — NO 2048 minimum)
- MIL: `T_k = mul(K_k, W_k)` then chain of adds `A0=T0+T1, A1=A0+T2, ...`
- **Slot ordering DECREASING**: large inputs first (opposite of normal rule!)
- CPU pre-computes K×K shifted copies of input (the only CPU work)

**Performance**: ~2.2ms at C=96 S=1024 K=7

**Layout conflict**: dw needs `[S,C]`, LN needs `[C,S]`. No ANE transpose → CPU copy between them.

**In practice**: depthwise 7×7 runs on CPU NEON instead. Faster than ANE dw + layout transpose overhead.

---

## Upsample 2x (CANNOT do on ANE)

Nearest-2x upsample requires a scatter matrix P where:
- `P[S, 4S]` (in `[C,S]` layout): P must be slot1 → eval=0 always
- `P[4S, S]` (in `[S,C]` layout): P must be slot0 → `4S×S > S×C` violates slot rule

**Conclusion**: No ANE upsample. Use CPU (4 memcpy patterns — negligible cost).

---

## Proven Training Pattern

Compile once, train forever. No recompilation when weights change.

### Per-layer kernel set
```
K1: fused_matmul    (W, x, dy) → (y, dx)      # fwd + bwd_dx in one kernel
K2: dW              (x, dy) → dW               # weight gradient
K3: adam_m          (dW, m) → m_new             # momentum update
K4: adam_v          (dW, v) → v_new             # variance update
K5: adam_w          (W, m, v) → W_new           # weight update
```

### Weight ping-pong
```
Step N:   K1.in[0] = W_A (current)  →  K5.out[0] = W_B (updated)
Step N+1: K1.in[0] = W_B (current)  →  K5.out[0] = W_A (updated)
Swap via ane_rewire() — zero copy, just pointer swap.
```

### Activation chaining
```
K1[layer0].out[0]  →  K1[layer1].in[1]    (via ane_rewire)
K1[layer1].out[1]  →  K1[layer0].in[2]    (gradient flows backward)
```

### Optimizer state
```
memcpy(K3.out → K3.in[1])   # m state carry-forward
memcpy(K4.out → K4.in[1])   # v state carry-forward
```

---

## What Runs Where

### On ANE
- All pointwise matmuls (fwd, bwd_dx, dW, projections)
- LayerNorm forward (5 dispatches)
- Activations: SiLU, GELU (tanh approx), sigmoid
- Residual adds
- Adam optimizer (m, v, w updates)
- Fused FFN blocks (pw1+silu+pw2+residual = 1 dispatch)

### On CPU (NEON vectorized)
- LayerNorm backward (requires reduction over C — `sum_c(dy)`, `sum_c(dy*norm)`)
- Depthwise convolution fwd + bwd (runtime W on ANE = InvalidMILProgram)
- Embed / head layers (weight matrices below 2048-byte minimum)
- GELU backward (tanh derivative, ~9 ops, small tensors)
- Any matmul where `C_out > C_in` violates slot ordering
- Layout transpose between `[C,S]` ↔ `[S,C]`

### On CPU (scalar)
- Loss computation (MSE/L1, reduction over all elements)
- Learning rate schedule
- Gradient clipping

---

## Fused Kernels (Production)

| Kernel | Ops fused | Dispatches saved | File |
|--------|-----------|-----------------|------|
| `fused_matmul` | fwd Y=W@X + bwd dX=Wᵀ@dY | 2→1 | `ane_matmul_bwd.h` |
| `fused_pw1_silu_pw2_add` | pw1 matmul + SiLU + pw2 matmul + residual | 4→1 | `ane_fused_pw1_silu_pw2_add.h` |
| `fused_silu_dw` | recompute SiLU + dW matmul | 2→1 | `ane_fused_silu_dw.h` |
| `k1b_2_3a` (LN) | sub + square + var matmul | 3→1 | `ane_ln.h` |

---

## Performance (ConvNeXt UNet, 256×256, M1)

```
Total step:     ~400ms
  ANE matmuls:  ~200ms (pointwise expand/contract, projections)
  ANE LN:       ~100ms (5 dispatches × 24 LN instances)
  CPU NEON dw:  ~50ms  (depthwise 7×7, 8 blocks)
  CPU NEON bwd: ~30ms  (LN backward, GELU backward)
  Adam:         ~20ms
```

Startup compile: ~2s for all kernels (cached after first run).

---

## ane_rewire — The Key Primitive

```c
// Swap kernel B's input slot 0 to use kernel A's output
IOSurfaceRef new_ins[2] = { A->ioOutputs[0], NULL };  // NULL = don't change
ane_rewire(B, new_ins, NULL);
ane_eval(B);  // now reads from A's output surface
```

- Must call after changing any IOSurface — pointer assignment alone does NOT update `_ANERequest`
- Changes take effect on next `ane_eval()`
- Used for: activation chaining, gradient routing, weight ping-pong
- Zero-copy — just swaps IOSurface pointers and rebuilds the ANE request
- Rewiring before first eval is fine; rewiring after eval works too

---

## _ANEChainingRequest (On-Chip Chaining)

Kernel A output → Kernel B input without CPU roundtrip. Verified working:
```
Input: all 1.0  →  A(W@x): 7.6875  →  B(W@(W@x)): 59.0625 = 7.6875²  ✓
```

Limited to kernels executed within the same evaluation context.

---

## Compilation

- `ane_compile(milData, nil, nInputs, inputSizes, nOutputs, outputSizes)`
- Returns NULL on failure — always check
- Compilation is slow (~100-500ms per kernel) but done once
- MIL text with large baked `const` tensors: compile time scales with tensor size
- Baked `const` as matmul operand → InvalidMILProgram

---

## Common Pitfalls

1. **Silent zeros**: Almost always slot ordering violation. Check input ascending / output descending.
2. **Silent zeros from matmul**: "Data" tensor in slot0 instead of slot1. Slot0 = weight, slot1 = activation.
3. **Silent zeros from Ci alignment**: Inner dimension not a multiple of 32. Ci=48,80 → zeros.
4. **Silent zeros from broadcast**: `[1,C,1]` broadcast in C-first layout always fails. Only `[1,1,S]` broadcast works.
5. **InvalidMILProgram**: Using an unsupported op (depthwise conv, reduction, reshape) with runtime inputs.
6. **Wrong values after sqrt**: sqrt in a fusion feeding broadcast or real_div. Keep sqrt isolated.
7. **Hangs**: Elementwise op on >16384 elements. Use CPU fallback for large tensors.
8. **Compile returns nil**: Empty weights dict `@{}` passed to `modelWithMILText:`. Always pass non-nil.
9. **dW zeros**: Multi-output kernel with mixed input sizes and small outputs. Keep dW as separate dispatch.
10. **Variable name collision**: Using MIL reserved words (`var`, `mean`, `x`, `mul`, etc.) as tensor names → silent failure.

---

## File Reference

| File | Purpose |
|------|---------|
| `ane_runtime.h` | Core API: `ane_compile()`, `ane_eval()`, `ane_rewire()` |
| `ane_mil_gen.h` | MIL text generators for matmul, adam, etc. |
| `modules/ops/ane_ln.h` | LayerNorm forward (5-kernel decomposition) |
| `modules/ops/ane_ln_bwd.h` | LayerNorm backward (CPU NEON) |
| `modules/ops/ane_matmul_bwd.h` | Matmul backward (fused fwd+bwd_dx + dW + adam) |
| `modules/ops/ane_adam.h` | Adam optimizer (3 ANE kernels) |
| `modules/ops/ane_gelu.h` | GELU forward (tanh approximation) |
| `modules/ops/ane_silu.h` | SiLU forward |
| `modules/ops/ane_fused_pw1_silu_pw2_add.h` | Fused FFN block (4→1 dispatch) |
| `modules/ops/ane_fused_silu_dw.h` | Fused SiLU recompute + dW |
| `modules/ops/ane_dw_bwd.h` | Depthwise conv backward (CPU NEON) |
| `modules/blocks/ane_convnext.h` | ConvNeXt block (forward) |
| `modules/blocks/ane_convnext_bwd.h` | ConvNeXt block (backward) |
| `train_unet.m` | Full training: ConvNeXt UNet, 256×256, ~420ms/step |

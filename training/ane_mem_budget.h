// ane_mem_budget.h — Conservative memory planner for M1/M2 ANE training
// Caps batch/seq/hidden dims to fit within unified memory without OOM
// Auto-enables gradient checkpointing when memory is tight
#pragma once
#include "ane_hw_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// ============================================================
// Memory budget configuration
// ============================================================
typedef struct {
    int batch_size;
    int seq_len;
    int hidden_dim;
    int n_layers;
    int vocab_size;
    int n_heads;
    bool gradient_checkpointing;     // recompute activations in backward
    int checkpoint_interval;         // checkpoint every N layers (2 = every other)
    int max_compiles_per_cycle;      // ANE compile limit before exec() restart
    int accum_steps;                 // gradient accumulation steps
    size_t estimated_peak_mb;        // estimated peak memory usage in MB
    bool reduced_precision_grads;    // use fp16 for gradient accumulators
} ANEMemBudget;

// ============================================================
// Estimate memory usage for a Stories110M-class model
//
// Memory breakdown per layer:
//   Weights:     4*d*d + 2*hd*d + d*hd + 2*d = ~7.5M floats for dim=768, hd=2048
//   Activations: ~12 buffers of S*d or S*hd each
//   Gradients:   same as weights
//   Adam state:  2x weights (m, v)
//
// Total per layer ≈ (weights + grads + 2*adam) + activations
//   = 4 * 7.5M * 4 bytes + 12 * S * max(d, hd) * 4 bytes
// ============================================================
static size_t _ane_estimate_peak_mb(int batch, int seq, int dim, int hidden, int layers, int vocab) {
    size_t params_per_layer = (size_t)(4 * dim * dim + 2 * hidden * dim + dim * hidden + 2 * dim);
    size_t total_params = params_per_layer * layers + (size_t)vocab * dim * 2 + dim;

    // Weights + gradients + Adam (m,v) = 4x params
    size_t weight_bytes = total_params * 4 * 4;

    // Activations: ~12 buffers per layer, each S*max(d,hd)
    int max_dim = hidden > dim ? hidden : dim;
    size_t acts_per_layer = (size_t)12 * batch * seq * max_dim * 4;
    size_t act_bytes = acts_per_layer * layers;

    // Logits buffer
    size_t logit_bytes = (size_t)batch * seq * vocab * 4;

    // IOSurface buffers (fp16, double-buffered input+output per kernel)
    size_t io_bytes = (size_t)2 * 7 * layers * batch * seq * max_dim * 2;

    return (weight_bytes + act_bytes + logit_bytes + io_bytes) / (1024 * 1024);
}

// ============================================================
// Compute budget with gradient checkpointing savings
// ============================================================
static size_t _ane_estimate_checkpointed_mb(int batch, int seq, int dim, int hidden,
                                              int layers, int vocab, int ckpt_interval) {
    size_t params_per_layer = (size_t)(4 * dim * dim + 2 * hidden * dim + dim * hidden + 2 * dim);
    size_t total_params = params_per_layer * layers + (size_t)vocab * dim * 2 + dim;
    size_t weight_bytes = total_params * 4 * 4;

    // With checkpointing: only keep activations for checkpoint_interval layers
    int max_dim = hidden > dim ? hidden : dim;
    int kept_layers = (layers + ckpt_interval - 1) / ckpt_interval;
    size_t acts_per_layer = (size_t)12 * batch * seq * max_dim * 4;
    size_t act_bytes = acts_per_layer * kept_layers;

    size_t logit_bytes = (size_t)batch * seq * vocab * 4;
    size_t io_bytes = (size_t)2 * 7 * layers * batch * seq * max_dim * 2;

    return (weight_bytes + act_bytes + logit_bytes + io_bytes) / (1024 * 1024);
}

// ============================================================
// M2MemoryBudget — primary entry point
//
// Given available unified memory, compute safe training parameters
// for Stories110M (12-layer, dim=768, hidden=2048, vocab=32000)
//
// Default: availableUnifiedGB=24 (M2 MacBook Pro max tier)
// ============================================================
static ANEMemBudget M2MemoryBudget(int availableUnifiedGB) {
    ANEChipProfile prof = ANEVersionDetect();
    ANEMemBudget b = {0};

    // Start with maximum dims for the chip
    b.batch_size = 1;
    b.n_layers = 12;   // Stories110M fixed
    b.vocab_size = 32000;
    b.n_heads = 12;

    // Seq and hidden constrained by chip profile
    b.seq_len = prof.max_seq_len;
    b.hidden_dim = prof.max_hidden_dim;

    // Stories110M has fixed dim=768, hidden=2048 — clamp to model spec
    if (b.hidden_dim > 2048) b.hidden_dim = 2048;
    if (b.seq_len > 1024) b.seq_len = 1024;

    // M2-specific caps per the task spec
    if (prof.gen == ANE_CHIP_M2 || prof.gen == ANE_CHIP_M1 || prof.gen == ANE_CHIP_UNKNOWN) {
        if (b.seq_len > 512) b.seq_len = 512;
        if (b.hidden_dim > 4096) b.hidden_dim = 4096;
        b.batch_size = 1;  // forced batch=1 on M1/M2
    }

    // Compile budget from chip profile
    b.max_compiles_per_cycle = prof.max_compiles;

    // Estimate unconstrained memory
    size_t peak_mb = _ane_estimate_peak_mb(b.batch_size, b.seq_len, 768,
                                            b.hidden_dim, b.n_layers, b.vocab_size);

    size_t available_mb = (size_t)availableUnifiedGB * 1024;
    // Reserve 30% for system + ANE compiler overhead
    size_t usable_mb = (available_mb * 70) / 100;

    printf("[ANE Budget] Chip: %s, Available: %d GB (%zu MB usable)\n",
           prof.name, availableUnifiedGB, usable_mb);
    printf("[ANE Budget] Initial estimate: %zu MB peak\n", peak_mb);

    // If it fits, no checkpointing needed
    if (peak_mb <= usable_mb) {
        b.gradient_checkpointing = false;
        b.checkpoint_interval = 0;
        b.estimated_peak_mb = peak_mb;
        b.accum_steps = 10;
        b.reduced_precision_grads = false;
        printf("[ANE Budget] Fits in memory — no gradient checkpointing needed\n");
    } else {
        // Enable gradient checkpointing
        b.gradient_checkpointing = true;

        // Try intervals: 2, 3, 4, 6
        int intervals[] = {2, 3, 4, 6};
        for (int i = 0; i < 4; i++) {
            size_t ckpt_mb = _ane_estimate_checkpointed_mb(
                b.batch_size, b.seq_len, 768, b.hidden_dim,
                b.n_layers, b.vocab_size, intervals[i]);
            if (ckpt_mb <= usable_mb) {
                b.checkpoint_interval = intervals[i];
                b.estimated_peak_mb = ckpt_mb;
                break;
            }
        }

        // If still doesn't fit, reduce seq_len
        if (b.estimated_peak_mb == 0 || b.estimated_peak_mb > usable_mb) {
            b.seq_len = 256;
            b.checkpoint_interval = 2;
            b.estimated_peak_mb = _ane_estimate_checkpointed_mb(
                b.batch_size, b.seq_len, 768, b.hidden_dim,
                b.n_layers, b.vocab_size, 2);
        }

        // Last resort: reduce seq_len further and use fp16 grads
        if (b.estimated_peak_mb > usable_mb) {
            b.seq_len = 128;
            b.reduced_precision_grads = true;
            b.estimated_peak_mb = _ane_estimate_checkpointed_mb(
                b.batch_size, b.seq_len, 768, b.hidden_dim,
                b.n_layers, b.vocab_size, 2);
        }

        // Increase accum steps to compensate for smaller effective batch
        b.accum_steps = (b.seq_len >= 256) ? 10 : 20;

        printf("[ANE Budget] Gradient checkpointing: interval=%d\n", b.checkpoint_interval);
        printf("[ANE Budget] Adjusted: seq=%d, peak=%zu MB\n", b.seq_len, b.estimated_peak_mb);
    }

    // Validate final configuration
    if (b.seq_len < 16) {
        fprintf(stderr, "[ANE Budget] FATAL: Cannot fit model in %d GB — seq_len reduced to %d\n",
                availableUnifiedGB, b.seq_len);
        b.seq_len = 16; // absolute minimum
    }

    printf("[ANE Budget] Final: batch=%d, seq=%d, hidden=%d, layers=%d, "
           "ckpt=%s (interval=%d), accum=%d, peak=%zu MB\n",
           b.batch_size, b.seq_len, b.hidden_dim, b.n_layers,
           b.gradient_checkpointing ? "ON" : "OFF", b.checkpoint_interval,
           b.accum_steps, b.estimated_peak_mb);

    return b;
}

// ============================================================
// Convenience: default M2 budget (24 GB)
// ============================================================
static ANEMemBudget M2DefaultBudget(void) {
    return M2MemoryBudget(24);
}

// ============================================================
// Convenience: auto-detect memory and compute budget
// ============================================================
static ANEMemBudget ANEAutoBudget(void) {
    int gb = _ane_detect_memory_gb();
    if (gb < 8) gb = 8;  // sanity floor
    return M2MemoryBudget(gb);
}


// gradient_checkpoint.h — Activation checkpointing for deep models
// Trades compute for memory: recompute forward activations during backward
// instead of storing all layers' activations simultaneously
#pragma once
#include "model_config.h"

// ===== Checkpoint policies =====

typedef enum {
    CKPT_ALL,           // save all layers' activations (current behavior)
    CKPT_BOUNDARY,      // save only group boundary activations, recompute within group
    CKPT_SQRT,          // save every √N layers (optimal memory/compute tradeoff)
    CKPT_EVERY_N,       // save every N-th layer (configurable)
    CKPT_NONE           // save nothing, recompute everything (max memory savings)
} CheckpointPolicy;

typedef struct {
    CheckpointPolicy policy;
    int interval;           // for CKPT_EVERY_N: save every N layers
    int n_layers;           // total layers in model
    int n_groups;           // layer groups in pipeline
    int layers_per_group;   // layers per group (from pipeline plan)
    // Derived
    int n_checkpointed;     // how many layers have saved activations
    bool *is_saved;         // per-layer: true if activation is saved (not recomputed)
} CheckpointManager;

// ===== Initialization =====

static CheckpointManager checkpoint_init(CheckpointPolicy policy, const ModelConfig *cfg,
                                          const PipelinePlan *plan) {
    CheckpointManager cm = {0};
    cm.policy = policy;
    cm.n_layers = cfg->dims.n_layers;
    cm.n_groups = plan->n_groups;
    cm.layers_per_group = (plan->n_groups > 0) ? plan->groups[0].n_layers : cfg->dims.n_layers;
    cm.is_saved = (bool *)calloc(cfg->dims.n_layers, sizeof(bool));

    switch (policy) {
    case CKPT_ALL:
        // Save everything — no recompute needed
        for (int i = 0; i < cm.n_layers; i++) cm.is_saved[i] = true;
        cm.n_checkpointed = cm.n_layers;
        break;

    case CKPT_BOUNDARY:
        // Save only the input to each layer group
        for (int g = 0; g < plan->n_groups; g++) {
            cm.is_saved[plan->groups[g].start_layer] = true;
        }
        // Always save the last layer's output (needed for loss backward)
        cm.is_saved[cm.n_layers - 1] = true;
        cm.n_checkpointed = plan->n_groups + 1;
        break;

    case CKPT_SQRT: {
        // Save every √N layers — optimal memory/compute balance
        int interval = (int)sqrtf((float)cm.n_layers);
        if (interval < 1) interval = 1;
        cm.interval = interval;
        for (int i = 0; i < cm.n_layers; i += interval) cm.is_saved[i] = true;
        cm.is_saved[cm.n_layers - 1] = true;
        cm.n_checkpointed = (cm.n_layers + interval - 1) / interval;
        break;
    }

    case CKPT_EVERY_N:
        // Caller should set cm.interval before using
        cm.interval = 4;   // default
        for (int i = 0; i < cm.n_layers; i += cm.interval) cm.is_saved[i] = true;
        cm.is_saved[cm.n_layers - 1] = true;
        cm.n_checkpointed = (cm.n_layers + cm.interval - 1) / cm.interval;
        break;

    case CKPT_NONE:
        // Save nothing except layer 0 input (needed as recompute starting point)
        cm.is_saved[0] = true;
        cm.n_checkpointed = 1;
        break;
    }

    return cm;
}

static void checkpoint_free(CheckpointManager *cm) {
    free(cm->is_saved);
    cm->is_saved = NULL;
}

// ===== Query functions =====

// Should we save this layer's activations during forward pass?
static bool checkpoint_should_save(const CheckpointManager *cm, int layer_idx) {
    if (layer_idx < 0 || layer_idx >= cm->n_layers) return false;
    return cm->is_saved[layer_idx];
}

// Does this layer need forward recompute during backward pass?
static bool checkpoint_needs_recompute(const CheckpointManager *cm, int layer_idx) {
    return !checkpoint_should_save(cm, layer_idx);
}

// Find the nearest saved checkpoint before this layer (for recompute starting point)
static int checkpoint_nearest_saved_before(const CheckpointManager *cm, int layer_idx) {
    for (int i = layer_idx; i >= 0; i--) {
        if (cm->is_saved[i]) return i;
    }
    return 0;   // fallback to first layer
}

// How many layers need recompute between the nearest checkpoint and this layer?
static int checkpoint_recompute_depth(const CheckpointManager *cm, int layer_idx) {
    int saved = checkpoint_nearest_saved_before(cm, layer_idx);
    return layer_idx - saved;
}

// ===== Memory estimation =====

// Memory for saved activations only (bytes)
static size_t checkpoint_saved_memory(const CheckpointManager *cm, const ModelDims *d) {
    return (size_t)cm->n_checkpointed * layer_activation_bytes(d);
}

// Memory savings vs. saving all layers (bytes)
static size_t checkpoint_memory_saved(const CheckpointManager *cm, const ModelDims *d) {
    size_t all = (size_t)cm->n_layers * layer_activation_bytes(d);
    size_t used = checkpoint_saved_memory(cm, d);
    return all - used;
}

// Extra forward FLOPs due to recompute (fraction of 1.0)
static double checkpoint_recompute_overhead(const CheckpointManager *cm) {
    int recomputed = cm->n_layers - cm->n_checkpointed;
    return (double)recomputed / (double)cm->n_layers;
}

// ===== Pretty-print =====

static const char *checkpoint_policy_name(CheckpointPolicy p) {
    switch (p) {
        case CKPT_ALL: return "ALL";
        case CKPT_BOUNDARY: return "BOUNDARY";
        case CKPT_SQRT: return "SQRT";
        case CKPT_EVERY_N: return "EVERY_N";
        case CKPT_NONE: return "NONE";
        default: return "UNKNOWN";
    }
}

static void checkpoint_print(const CheckpointManager *cm, const ModelDims *d) {
    printf("=== Checkpoint Policy: %s ===\n", checkpoint_policy_name(cm->policy));
    printf("  %d/%d layers checkpointed", cm->n_checkpointed, cm->n_layers);
    if (cm->policy == CKPT_SQRT || cm->policy == CKPT_EVERY_N)
        printf(" (interval=%d)", cm->interval);
    printf("\n");
    printf("  Activation memory: %.1fMB (saved) / %.1fMB (all)\n",
           checkpoint_saved_memory(cm, d) / 1e6,
           (double)cm->n_layers * layer_activation_bytes(d) / 1e6);
    printf("  Memory savings: %.1fMB (%.0f%%)\n",
           checkpoint_memory_saved(cm, d) / 1e6,
           100.0 * checkpoint_memory_saved(cm, d) / ((double)cm->n_layers * layer_activation_bytes(d)));
    printf("  Recompute overhead: %.0f%% extra forward FLOPs\n",
           100.0 * checkpoint_recompute_overhead(cm));
    printf("  Saved layers: ");
    for (int i = 0; i < cm->n_layers; i++) {
        if (cm->is_saved[i]) printf("%d ", i);
    }
    printf("\n");
}


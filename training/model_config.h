// model_config.h — Parameterized model configuration for pipeline training
// Replaces hardcoded #defines with portable structs + preset configs
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// ===== Model configuration =====

typedef struct {
    int dim;            // model dimension (embedding/residual width)
    int hidden_dim;     // FFN hidden dimension
    int n_heads;        // number of attention heads
    int n_kv_heads;     // number of KV heads (for GQA; == n_heads for MHA)
    int n_layers;       // total transformer layers
    int vocab_size;     // vocabulary size
    int seq_len;        // maximum sequence length
    // Derived (computed by model_config_init)
    int head_dim;       // dim / n_heads
    int kv_dim;         // head_dim * n_kv_heads
    int score_ch;       // n_heads * seq_len (attention score channels for SDPA bwd)
} ModelDims;

typedef struct {
    int compile_budget;     // max ANE compilations per process (~119)
    int kernels_per_layer;  // weight-bearing kernels per layer (currently 5)
    int static_per_layer;   // weight-free kernels per layer (sdpaBwd2 = 1)
    int accum_steps;        // gradient accumulation steps per compile batch
    float headroom_pct;     // safety margin as fraction of budget (0.0-1.0, default 0.10)
} CompileConfig;

typedef struct {
    ModelDims dims;
    CompileConfig compile;
    const char *name;       // human-readable model name
} ModelConfig;

// ===== Layer group for pipeline scheduling =====

typedef struct {
    int start_layer;        // first layer index (inclusive)
    int end_layer;          // last layer index (exclusive)
    int n_layers;           // end_layer - start_layer
    int weight_kernels;     // weight-bearing kernels in this group
    int static_kernels;     // weight-free kernels in this group
    int total_kernels;      // weight_kernels + static_kernels
} LayerGroup;

typedef struct {
    LayerGroup *groups;
    int n_groups;
    int total_exec_restarts;    // estimated exec() restarts per training step
} PipelinePlan;

// ===== Derived dimension helpers =====

static void model_dims_init(ModelDims *d) {
    d->head_dim = (d->n_heads > 0) ? d->dim / d->n_heads : 0;
    d->kv_dim = d->head_dim * d->n_kv_heads;
    d->score_ch = d->n_heads * d->seq_len;
}

// ===== Per-layer memory sizes (bytes) =====

// Weight sizes in floats (fp32)
static inline size_t wq_size(const ModelDims *d) { return (size_t)d->dim * d->dim; }
static inline size_t wo_size(const ModelDims *d) { return (size_t)d->dim * d->dim; }
static inline size_t w1_size(const ModelDims *d) { return (size_t)d->hidden_dim * d->dim; }
static inline size_t w2_size(const ModelDims *d) { return (size_t)d->dim * d->hidden_dim; }
static inline size_t w3_size(const ModelDims *d) { return (size_t)d->hidden_dim * d->dim; }

static inline size_t layer_weight_floats(const ModelDims *d) {
    return 4 * wq_size(d)    // Wq, Wk, Wv, Wo
         + w1_size(d) + w2_size(d) + w3_size(d)   // W1, W2, W3
         + 2 * (size_t)d->dim;                     // rms_att, rms_ffn
}

static inline size_t layer_weight_bytes(const ModelDims *d) {
    return layer_weight_floats(d) * sizeof(float);
}

// Adam state: 2x weight size (m + v vectors)
static inline size_t layer_adam_bytes(const ModelDims *d) {
    return 2 * layer_weight_bytes(d);
}

// Activation buffers per layer (saved for backward)
static inline size_t layer_activation_floats(const ModelDims *d) {
    int S = d->seq_len, D = d->dim, H = d->hidden_dim;
    // layer_in, xnorm, Q, K, V, attn_out, o_out, x2, x2norm = 9 * D*S
    // h1, h3, silu_out = 3 * H*S
    // ffn_out = D*S
    return (size_t)(10 * D * S + 3 * H * S);
}

static inline size_t layer_activation_bytes(const ModelDims *d) {
    return layer_activation_floats(d) * sizeof(float);
}

// Gradient accumulators per layer
static inline size_t layer_gradient_bytes(const ModelDims *d) {
    return layer_weight_bytes(d);   // same layout as weights
}

// Total model memory (weights + adam + activations + gradients for all layers)
static inline size_t total_model_bytes(const ModelConfig *cfg) {
    const ModelDims *d = &cfg->dims;
    size_t per_layer = layer_weight_bytes(d) + layer_adam_bytes(d)
                     + layer_activation_bytes(d) + layer_gradient_bytes(d);
    size_t global = (size_t)d->dim * sizeof(float)                  // rms_final
                  + (size_t)d->vocab_size * d->dim * sizeof(float)  // embed
                  + (size_t)d->dim * 2 * sizeof(float)              // rms_final adam
                  + (size_t)d->vocab_size * d->dim * 2 * sizeof(float)  // embed adam
                  + (size_t)d->dim * sizeof(float)                  // rms_final grad
                  + (size_t)d->vocab_size * d->dim * sizeof(float); // embed grad
    return per_layer * d->n_layers + global;
}

// ===== Pipeline planning =====

// Compute how many layers can fit in one compile batch
static int max_layers_per_compile(const CompileConfig *cc) {
    float headroom = (cc->headroom_pct > 0.0f && cc->headroom_pct < 1.0f)
                   ? cc->headroom_pct : 0.10f;
    int usable = (int)(cc->compile_budget * (1.0f - headroom));
    int per_layer = cc->kernels_per_layer + cc->static_per_layer;
    if (per_layer <= 0) return 1;
    return usable / per_layer;
}

// Compute optimal layer groups for a model given compile budget
// Returns a PipelinePlan (caller must free plan.groups)
static PipelinePlan compute_pipeline_plan(const ModelConfig *cfg) {
    PipelinePlan plan = {0};
    int max_per = max_layers_per_compile(&cfg->compile);
    if (max_per <= 0) max_per = 1;

    // Clamp to actual layer count
    int group_size = (max_per < cfg->dims.n_layers) ? max_per : cfg->dims.n_layers;

    plan.n_groups = (cfg->dims.n_layers + group_size - 1) / group_size;
    plan.groups = (LayerGroup *)calloc(plan.n_groups, sizeof(LayerGroup));

    int kpl = cfg->compile.kernels_per_layer;
    int spl = cfg->compile.static_per_layer;

    for (int g = 0; g < plan.n_groups; g++) {
        LayerGroup *lg = &plan.groups[g];
        lg->start_layer = g * group_size;
        lg->end_layer = lg->start_layer + group_size;
        if (lg->end_layer > cfg->dims.n_layers)
            lg->end_layer = cfg->dims.n_layers;
        lg->n_layers = lg->end_layer - lg->start_layer;
        lg->weight_kernels = lg->n_layers * kpl;
        lg->static_kernels = lg->n_layers * spl;
        lg->total_kernels = lg->weight_kernels + lg->static_kernels;
    }

    // Each compile batch needs one exec() restart (except possibly the last)
    // Forward: n_groups compiles. Backward: n_groups compiles.
    // Per training step: forward + backward = 2 * n_groups compile batches
    // Each batch may need exec() restart. Worst case:
    plan.total_exec_restarts = 2 * plan.n_groups;

    return plan;
}

static void pipeline_plan_free(PipelinePlan *plan) {
    free(plan->groups);
    plan->groups = NULL;
    plan->n_groups = 0;
}

// ===== Pretty-print plan =====

static void pipeline_plan_print(const ModelConfig *cfg, const PipelinePlan *plan) {
    printf("=== Pipeline Plan: %s ===\n", cfg->name);
    printf("  %d layers | dim=%d hidden=%d heads=%d seq=%d vocab=%d\n",
           cfg->dims.n_layers, cfg->dims.dim, cfg->dims.hidden_dim,
           cfg->dims.n_heads, cfg->dims.seq_len, cfg->dims.vocab_size);
    printf("  Compile budget: %d | %d weight-kernels/layer + %d static/layer\n",
           cfg->compile.compile_budget, cfg->compile.kernels_per_layer,
           cfg->compile.static_per_layer);
    printf("  %d layer group(s):\n", plan->n_groups);
    for (int g = 0; g < plan->n_groups; g++) {
        const LayerGroup *lg = &plan->groups[g];
        printf("    Group %d: layers [%d..%d) — %d layers, %d kernels (%d weight + %d static)\n",
               g, lg->start_layer, lg->end_layer, lg->n_layers,
               lg->total_kernels, lg->weight_kernels, lg->static_kernels);
    }
    printf("  Est. exec() restarts per step: %d\n", plan->total_exec_restarts);
    printf("  Memory per layer: weights=%.1fMB adam=%.1fMB acts=%.1fMB grads=%.1fMB\n",
           layer_weight_bytes(&cfg->dims)/1e6, layer_adam_bytes(&cfg->dims)/1e6,
           layer_activation_bytes(&cfg->dims)/1e6, layer_gradient_bytes(&cfg->dims)/1e6);
    printf("  Total model state: %.1fMB\n", total_model_bytes(cfg)/1e6);
}

// ===== FLOP estimation per step =====

static inline double flops_per_step(const ModelConfig *cfg) {
    const ModelDims *d = &cfg->dims;
    int N = d->n_layers, D = d->dim, H = d->hidden_dim, S = d->seq_len;
    int HD = d->head_dim, NH = d->n_heads;
    // Forward: 4 linear projections (QKV+O) + 3 FFN projections per layer
    double fwd = N * (4.0*2*D*D*S + 2.0*2*D*H*S + 2.0*H*D*S);
    // Backward dx same flops as forward
    double bwd_dx = fwd;
    // Backward dW same flops as forward
    double bwd_dw = fwd;
    // SDPA backward (attention score computation)
    double sdpa = N * 2.0 * NH * 5 * S * S * HD;
    // Classifier (forward + backward)
    double cls = 3.0 * 2.0 * d->vocab_size * D * S;
    return fwd + bwd_dx + bwd_dw + sdpa + cls;
}

static inline double ane_flops_per_step(const ModelConfig *cfg) {
    const ModelDims *d = &cfg->dims;
    int N = d->n_layers, D = d->dim, H = d->hidden_dim, S = d->seq_len;
    int HD = d->head_dim, NH = d->n_heads;
    double fwd = N * (4.0*2*D*D*S + 2.0*2*D*H*S + 2.0*H*D*S);
    double bwd_dx = fwd;
    double sdpa = N * 2.0 * NH * 5 * S * S * HD;
    return fwd + bwd_dx + sdpa;  // dW is on CPU (cblas)
}

// ===== Model presets =====

static ModelConfig model_config_stories110m(void) {
    ModelConfig cfg = {0};
    cfg.name = "Stories110M";
    cfg.dims = (ModelDims){
        .dim = 768, .hidden_dim = 2048, .n_heads = 12,
        .n_kv_heads = 12, .n_layers = 12, .vocab_size = 32000, .seq_len = 256
    };
    cfg.compile = (CompileConfig){
        .compile_budget = 119, .kernels_per_layer = 5,
        .static_per_layer = 1, .accum_steps = 10, .headroom_pct = 0.10f
    };
    model_dims_init(&cfg.dims);
    return cfg;
}

static ModelConfig model_config_stories42m(void) {
    ModelConfig cfg = {0};
    cfg.name = "Stories42M";
    cfg.dims = (ModelDims){
        .dim = 512, .hidden_dim = 1376, .n_heads = 8,
        .n_kv_heads = 8, .n_layers = 8, .vocab_size = 32000, .seq_len = 256
    };
    cfg.compile = (CompileConfig){
        .compile_budget = 119, .kernels_per_layer = 5,
        .static_per_layer = 1, .accum_steps = 10, .headroom_pct = 0.10f
    };
    model_dims_init(&cfg.dims);
    return cfg;
}

static ModelConfig model_config_llama_1b(void) {
    ModelConfig cfg = {0};
    cfg.name = "LLaMA-1.1B";
    cfg.dims = (ModelDims){
        .dim = 2048, .hidden_dim = 5504, .n_heads = 16,
        .n_kv_heads = 16, .n_layers = 22, .vocab_size = 32000, .seq_len = 512
    };
    cfg.compile = (CompileConfig){
        .compile_budget = 119, .kernels_per_layer = 5,
        .static_per_layer = 1, .accum_steps = 4, .headroom_pct = 0.10f
    };
    model_dims_init(&cfg.dims);
    return cfg;
}

static ModelConfig model_config_llama_7b(void) {
    ModelConfig cfg = {0};
    cfg.name = "LLaMA-7B";
    cfg.dims = (ModelDims){
        .dim = 4096, .hidden_dim = 11008, .n_heads = 32,
        .n_kv_heads = 32, .n_layers = 32, .vocab_size = 32000, .seq_len = 512
    };
    cfg.compile = (CompileConfig){
        .compile_budget = 119, .kernels_per_layer = 5,
        .static_per_layer = 1, .accum_steps = 2, .headroom_pct = 0.10f
    };
    model_dims_init(&cfg.dims);
    return cfg;
}

// Parse config from command-line (returns preset, caller can override)
static ModelConfig model_config_from_args(int argc, char *argv[]) {
    ModelConfig cfg = model_config_stories110m(); // default
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) {
            const char *name = argv[++i];
            if (strcmp(name, "stories42m") == 0) cfg = model_config_stories42m();
            else if (strcmp(name, "stories110m") == 0) cfg = model_config_stories110m();
            else if (strcmp(name, "llama1b") == 0) cfg = model_config_llama_1b();
            else if (strcmp(name, "llama7b") == 0) cfg = model_config_llama_7b();
            else fprintf(stderr, "Unknown model: %s (using stories110m)\n", name);
        }
        else if (strcmp(argv[i], "--dim") == 0 && i+1 < argc) cfg.dims.dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i+1 < argc) cfg.dims.hidden_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i+1 < argc) cfg.dims.n_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--layers") == 0 && i+1 < argc) cfg.dims.n_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i+1 < argc) cfg.dims.seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vocab") == 0 && i+1 < argc) cfg.dims.vocab_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--budget") == 0 && i+1 < argc) cfg.compile.compile_budget = atoi(argv[++i]);
        else if (strcmp(argv[i], "--accum") == 0 && i+1 < argc) cfg.compile.accum_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--headroom") == 0 && i+1 < argc) cfg.compile.headroom_pct = atof(argv[++i]);
    }
    model_dims_init(&cfg.dims);
    return cfg;
}

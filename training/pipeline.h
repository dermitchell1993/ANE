// pipeline.h — Layer-group scheduling and mmap state for multi-group ANE training
// Manages compile budgets, exec() restarts, and cross-exec shared tensor state
#pragma once
#include "model_config.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

// ===== Compile budget tracker =====

typedef struct {
    int budget;         // max compilations allowed
    int used;           // compilations consumed so far
    int headroom;       // safety margin (budget * 0.1)
} CompileBudget;

static CompileBudget budget_init(const CompileConfig *cc) {
    CompileBudget b;
    b.budget = cc->compile_budget;
    b.used = 0;
    float pct = (cc->headroom_pct > 0.0f && cc->headroom_pct < 1.0f)
              ? cc->headroom_pct : 0.10f;
    b.headroom = (int)(cc->compile_budget * pct);
    return b;
}

static bool budget_can_fit(const CompileBudget *b, int n_kernels) {
    return (b->used + n_kernels) <= (b->budget - b->headroom);
}

static void budget_consume(CompileBudget *b, int n_kernels) {
    b->used += n_kernels;
}

static bool budget_needs_restart(const CompileBudget *b) {
    return b->used >= (b->budget - b->headroom);
}

static int budget_remaining(const CompileBudget *b) {
    int r = b->budget - b->headroom - b->used;
    return (r > 0) ? r : 0;
}

// ===== Pipeline execution phases =====

typedef enum {
    PHASE_INIT = 0,
    PHASE_FORWARD,          // running forward pass through layer groups
    PHASE_BACKWARD,         // running backward pass through layer groups (reverse)
    PHASE_WEIGHT_UPDATE,    // Adam step on accumulated gradients
    PHASE_DONE              // training step complete
} PipelinePhase;

typedef enum {
    ACTION_COMPILE_GROUP,       // compile kernels for current layer group
    ACTION_RUN_FORWARD_GROUP,   // execute forward pass for compiled group
    ACTION_RUN_BACKWARD_GROUP,  // execute backward pass for compiled group
    ACTION_EXEC_RESTART,        // save state and exec() to reset compile budget
    ACTION_WEIGHT_UPDATE,       // run optimizer on all layers
    ACTION_STEP_DONE,           // training step complete
    ACTION_ERROR                // something went wrong
} PipelineAction;

// ===== Scheduler state =====

typedef struct {
    ModelConfig config;
    PipelinePlan plan;
    CompileBudget budget;

    PipelinePhase phase;
    int current_group;      // index into plan.groups
    int current_step;       // training step number
    int accum_step;         // gradient accumulation step within batch
    int total_steps;        // total training steps requested
    float learning_rate;
    float last_loss;

    // Flags
    bool group_compiled;    // whether current group's kernels are compiled
    bool needs_restart;     // whether we need exec() before next group
} PipelineScheduler;

static PipelineScheduler pipeline_scheduler_init(ModelConfig config, int total_steps, float lr) {
    PipelineScheduler s = {0};
    s.config = config;
    s.plan = compute_pipeline_plan(&config);
    s.budget = budget_init(&config.compile);
    s.phase = PHASE_FORWARD;
    s.current_group = 0;
    s.current_step = 0;
    s.accum_step = 0;
    s.total_steps = total_steps;
    s.learning_rate = lr;
    s.last_loss = 999.0f;
    s.group_compiled = false;
    s.needs_restart = false;
    return s;
}

// Get the next action the training loop should take
static PipelineAction pipeline_next_action(PipelineScheduler *s) {
    if (s->current_step >= s->total_steps)
        return ACTION_STEP_DONE;

    switch (s->phase) {
    case PHASE_FORWARD:
        if (s->current_group >= s->plan.n_groups) {
            // Forward pass complete for all groups — start backward
            s->phase = PHASE_BACKWARD;
            s->current_group = s->plan.n_groups - 1;
            s->group_compiled = false;
            return pipeline_next_action(s);
        }
        if (!s->group_compiled) {
            // Check if we have compile budget for this group
            LayerGroup *lg = &s->plan.groups[s->current_group];
            if (!budget_can_fit(&s->budget, lg->total_kernels)) {
                s->needs_restart = true;
                return ACTION_EXEC_RESTART;
            }
            return ACTION_COMPILE_GROUP;
        }
        return ACTION_RUN_FORWARD_GROUP;

    case PHASE_BACKWARD:
        if (s->current_group < 0) {
            // Backward complete — weight update
            s->phase = PHASE_WEIGHT_UPDATE;
            return ACTION_WEIGHT_UPDATE;
        }
        if (!s->group_compiled) {
            LayerGroup *lg = &s->plan.groups[s->current_group];
            if (!budget_can_fit(&s->budget, lg->total_kernels)) {
                s->needs_restart = true;
                return ACTION_EXEC_RESTART;
            }
            return ACTION_COMPILE_GROUP;
        }
        return ACTION_RUN_BACKWARD_GROUP;

    case PHASE_WEIGHT_UPDATE:
        return ACTION_WEIGHT_UPDATE;

    case PHASE_DONE:
        return ACTION_STEP_DONE;

    default:
        return ACTION_ERROR;
    }
}

// Called after successfully compiling a layer group's kernels
static void pipeline_group_compiled(PipelineScheduler *s) {
    LayerGroup *lg = &s->plan.groups[s->current_group];
    budget_consume(&s->budget, lg->total_kernels);
    s->group_compiled = true;
}

// Called after successfully running forward for current group
static void pipeline_forward_group_done(PipelineScheduler *s) {
    s->current_group++;
    s->group_compiled = false;
}

// Called after successfully running backward for current group
static void pipeline_backward_group_done(PipelineScheduler *s) {
    s->current_group--;
    s->group_compiled = false;
}

// Called after weight update completes
static void pipeline_weight_update_done(PipelineScheduler *s) {
    s->accum_step++;
    if (s->accum_step >= s->config.compile.accum_steps) {
        s->accum_step = 0;
        s->current_step++;
    }
    // Reset for next forward pass
    s->phase = PHASE_FORWARD;
    s->current_group = 0;
    s->group_compiled = false;
}

// ===== mmap-based cross-exec state =====
//
// Layout: [Header][Layer 0 weights][Layer 0 adam][Layer 0 grads]...[Global state]
// All tensors stored as fp32. The mmap file persists across exec() restarts.

#define MMAP_SENTINEL 0x414E4550  // "ANEP" — file format identifier
#define MMAP_VERSION 1

typedef struct {
    int sentinel;       // MMAP_SENTINEL for file identification
    int version;
    int n_layers;
    int dim;
    int hidden_dim;
    int n_heads;
    int vocab_size;
    int seq_len;
    // Scheduler state (for exec restart)
    int phase;
    int current_group;
    int current_step;
    int accum_step;
    int total_steps;
    int compile_count;      // compiles used in current process
    int adam_t;             // Adam timestep
    float learning_rate;
    float last_loss;
    // Offsets into mmap (bytes from base)
    size_t layer_weights_offset;    // start of per-layer weight data
    size_t layer_adam_offset;       // start of per-layer adam state
    size_t layer_grads_offset;      // start of per-layer gradient accumulators
    size_t layer_acts_offset;       // start of per-layer activation checkpoints
    size_t global_offset;           // start of global state (rms_final, embed, etc.)
    size_t total_size;              // total mmap size
    int pad[4];                     // alignment
} MmapHeader;

typedef struct {
    int fd;
    void *base;
    size_t size;
    MmapHeader *header;
    const char *path;
} MmapState;

// Compute mmap layout for a given config
static size_t mmap_compute_size(const ModelConfig *cfg) {
    const ModelDims *d = &cfg->dims;
    size_t header = sizeof(MmapHeader);
    // Round up to page boundary
    header = (header + 4095) & ~(size_t)4095;

    size_t per_layer_weights = layer_weight_bytes(d);
    size_t per_layer_adam = layer_adam_bytes(d);
    size_t per_layer_grads = layer_gradient_bytes(d);
    size_t per_layer_acts = layer_activation_bytes(d);

    size_t all_layers = (size_t)d->n_layers * (per_layer_weights + per_layer_adam + per_layer_grads + per_layer_acts);

    // Global: rms_final + embed + their adam states + embed gradients
    size_t global = (size_t)d->dim * 4                          // rms_final
                  + (size_t)d->vocab_size * d->dim * 4          // embed
                  + (size_t)d->dim * 2 * 4                      // rms_final adam (m+v)
                  + (size_t)d->vocab_size * d->dim * 2 * 4      // embed adam
                  + (size_t)d->dim * 4                          // rms_final grad
                  + (size_t)d->vocab_size * d->dim * 4;         // embed grad

    return header + all_layers + global;
}

// Create a new mmap state file
static MmapState *mmap_state_create(const char *path, const ModelConfig *cfg) {
    size_t total = mmap_compute_size(cfg);
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("mmap_state_create: open"); return NULL; }
    if (ftruncate(fd, total) < 0) { perror("mmap_state_create: ftruncate"); close(fd); return NULL; }

    void *base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap_state_create: mmap"); close(fd); return NULL; }

    MmapState *ms = (MmapState *)calloc(1, sizeof(MmapState));
    if (!ms) { perror("mmap_state_create: calloc"); munmap(base, total); close(fd); return NULL; }
    ms->fd = fd;
    ms->base = base;
    ms->size = total;
    ms->path = path;
    ms->header = (MmapHeader *)base;

    // Initialize header
    MmapHeader *h = ms->header;
    h->sentinel = MMAP_SENTINEL;
    h->version = MMAP_VERSION;
    h->n_layers = cfg->dims.n_layers;
    h->dim = cfg->dims.dim;
    h->hidden_dim = cfg->dims.hidden_dim;
    h->n_heads = cfg->dims.n_heads;
    h->vocab_size = cfg->dims.vocab_size;
    h->seq_len = cfg->dims.seq_len;

    // Compute offsets
    size_t header_end = (sizeof(MmapHeader) + 4095) & ~(size_t)4095;
    const ModelDims *d = &cfg->dims;
    size_t pw = layer_weight_bytes(d);
    size_t pa = layer_adam_bytes(d);
    size_t pg = layer_gradient_bytes(d);
    size_t pact = layer_activation_bytes(d);

    h->layer_weights_offset = header_end;
    h->layer_adam_offset = h->layer_weights_offset + (size_t)d->n_layers * pw;
    h->layer_grads_offset = h->layer_adam_offset + (size_t)d->n_layers * pa;
    h->layer_acts_offset = h->layer_grads_offset + (size_t)d->n_layers * pg;
    h->global_offset = h->layer_acts_offset + (size_t)d->n_layers * pact;
    h->total_size = total;

    return ms;
}

// Reopen existing mmap state (after exec() restart)
static MmapState *mmap_state_open(const char *path) {
    int fd = open(path, O_RDWR);
    if (fd < 0) { perror("mmap_state_open: open"); return NULL; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("mmap_state_open: fstat"); close(fd); return NULL; }

    void *base = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap_state_open: mmap"); close(fd); return NULL; }

    if ((size_t)st.st_size < sizeof(MmapHeader)) {
        fprintf(stderr, "mmap_state_open: file too small (%lld bytes)\n", (long long)st.st_size);
        munmap(base, st.st_size);
        close(fd);
        return NULL;
    }

    MmapHeader *h = (MmapHeader *)base;
    if (h->sentinel != MMAP_SENTINEL || h->version != MMAP_VERSION) {
        fprintf(stderr, "mmap_state_open: invalid header (sentinel=0x%08x version=%d)\n",
                h->sentinel, h->version);
        munmap(base, st.st_size);
        close(fd);
        return NULL;
    }

    if (h->total_size != 0 && (size_t)st.st_size < h->total_size) {
        fprintf(stderr, "mmap_state_open: file truncated (expected %zu, got %lld)\n",
                h->total_size, (long long)st.st_size);
        munmap(base, st.st_size);
        close(fd);
        return NULL;
    }

    MmapState *ms = (MmapState *)calloc(1, sizeof(MmapState));
    if (!ms) { perror("mmap_state_open: calloc"); munmap(base, st.st_size); close(fd); return NULL; }
    ms->fd = fd;
    ms->base = base;
    ms->size = st.st_size;
    ms->path = path;
    ms->header = h;
    return ms;
}

// Close and unmap (does NOT delete the file)
static void mmap_state_close(MmapState *ms) {
    if (!ms) return;
    if (ms->base && ms->base != MAP_FAILED) {
        if (msync(ms->base, ms->size, MS_SYNC) < 0) perror("mmap_state_close: msync");
        if (munmap(ms->base, ms->size) < 0) perror("mmap_state_close: munmap");
    }
    if (ms->fd >= 0) close(ms->fd);
    free(ms);
}

// Delete the mmap file (call after training completes)
static void mmap_state_destroy(MmapState *ms) {
    if (!ms) return;
    const char *p = ms->path;
    mmap_state_close(ms);
    unlink(p);
}

// ===== Typed accessors into mmap regions =====

// Reconstruct ModelDims from mmap header (avoids repeating in each accessor)
static inline ModelDims mmap_dims(const MmapState *ms) {
    return (ModelDims){
        .dim = ms->header->dim, .hidden_dim = ms->header->hidden_dim,
        .n_heads = ms->header->n_heads, .vocab_size = ms->header->vocab_size,
        .seq_len = ms->header->seq_len
    };
}

// Get pointer to layer L's weights in mmap (NULL if out of bounds)
static float *mmap_layer_weights(MmapState *ms, int layer) {
    if (!ms || layer < 0 || layer >= ms->header->n_layers) return NULL;
    ModelDims d = mmap_dims(ms);
    return (float *)((char *)ms->base + ms->header->layer_weights_offset
                     + (size_t)layer * layer_weight_bytes(&d));
}

// Get pointer to layer L's adam state in mmap (NULL if out of bounds)
static float *mmap_layer_adam(MmapState *ms, int layer) {
    if (!ms || layer < 0 || layer >= ms->header->n_layers) return NULL;
    ModelDims d = mmap_dims(ms);
    return (float *)((char *)ms->base + ms->header->layer_adam_offset
                     + (size_t)layer * layer_adam_bytes(&d));
}

// Get pointer to layer L's gradient accumulators in mmap (NULL if out of bounds)
static float *mmap_layer_grads(MmapState *ms, int layer) {
    if (!ms || layer < 0 || layer >= ms->header->n_layers) return NULL;
    ModelDims d = mmap_dims(ms);
    return (float *)((char *)ms->base + ms->header->layer_grads_offset
                     + (size_t)layer * layer_gradient_bytes(&d));
}

// Get pointer to layer L's activation checkpoint in mmap (NULL if out of bounds)
static float *mmap_layer_acts(MmapState *ms, int layer) {
    if (!ms || layer < 0 || layer >= ms->header->n_layers) return NULL;
    ModelDims d = mmap_dims(ms);
    return (float *)((char *)ms->base + ms->header->layer_acts_offset
                     + (size_t)layer * layer_activation_bytes(&d));
}

// Get pointer to global state region (rms_final, embed, etc.)
static float *mmap_global(MmapState *ms) {
    return (float *)((char *)ms->base + ms->header->global_offset);
}

// ===== Save/restore scheduler state to/from mmap header =====

static void pipeline_save_to_mmap(const PipelineScheduler *s, MmapState *ms) {
    MmapHeader *h = ms->header;
    h->phase = (int)s->phase;
    h->current_group = s->current_group;
    h->current_step = s->current_step;
    h->accum_step = s->accum_step;
    h->total_steps = s->total_steps;
    h->learning_rate = s->learning_rate;
    h->last_loss = s->last_loss;
    msync(ms->base, sizeof(MmapHeader), MS_SYNC);
}

static void pipeline_restore_from_mmap(PipelineScheduler *s, const MmapState *ms) {
    const MmapHeader *h = ms->header;
    s->phase = (PipelinePhase)h->phase;
    s->current_group = h->current_group;
    s->current_step = h->current_step;
    s->accum_step = h->accum_step;
    s->total_steps = h->total_steps;
    s->learning_rate = h->learning_rate;
    s->last_loss = h->last_loss;
    // Reset compile budget (new process after exec)
    s->budget = budget_init(&s->config.compile);
    s->group_compiled = false;
    s->needs_restart = false;
}

// ===== exec() restart with mmap persistence =====

// Call this when ACTION_EXEC_RESTART is returned.
// Saves scheduler state to mmap, syncs, and exec()s.
// Does not return on success.
static void pipeline_exec_restart(PipelineScheduler *s, MmapState *ms, char *argv[]) {
    pipeline_save_to_mmap(s, ms);
    printf("[pipeline] exec() restart: step=%d phase=%d group=%d compiles=%d\n",
           s->current_step, s->phase, s->current_group, s->budget.used);
    fflush(stdout);

    // Sync all mmap data before exec
    msync(ms->base, ms->size, MS_SYNC);

    // exec with --pipeline-resume flag
    execl(argv[0], argv[0], "--pipeline-resume", ms->path, NULL);
    perror("pipeline_exec_restart: execl");
}

// Resume from exec() restart. Returns true if this is a resume.
static bool pipeline_check_resume(int argc, char *argv[], PipelineScheduler *s, MmapState **ms_out) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--pipeline-resume") == 0 && i+1 < argc) {
            const char *mmap_path = argv[i+1];
            MmapState *ms = mmap_state_open(mmap_path);
            if (!ms) {
                fprintf(stderr, "[pipeline] Failed to reopen mmap at %s\n", mmap_path);
                return false;
            }
            pipeline_restore_from_mmap(s, ms);
            *ms_out = ms;
            printf("[pipeline] Resumed: step=%d phase=%d group=%d\n",
                   s->current_step, s->phase, s->current_group);
            return true;
        }
    }
    return false;
}

// ===== Pipeline pretty-print helpers =====

static const char *phase_name(PipelinePhase p) {
    switch (p) {
        case PHASE_INIT: return "INIT";
        case PHASE_FORWARD: return "FORWARD";
        case PHASE_BACKWARD: return "BACKWARD";
        case PHASE_WEIGHT_UPDATE: return "WEIGHT_UPDATE";
        case PHASE_DONE: return "DONE";
        default: return "UNKNOWN";
    }
}

static const char *action_name(PipelineAction a) {
    switch (a) {
        case ACTION_COMPILE_GROUP: return "COMPILE_GROUP";
        case ACTION_RUN_FORWARD_GROUP: return "RUN_FORWARD_GROUP";
        case ACTION_RUN_BACKWARD_GROUP: return "RUN_BACKWARD_GROUP";
        case ACTION_EXEC_RESTART: return "EXEC_RESTART";
        case ACTION_WEIGHT_UPDATE: return "WEIGHT_UPDATE";
        case ACTION_STEP_DONE: return "STEP_DONE";
        case ACTION_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static void pipeline_print_status(const PipelineScheduler *s) {
    printf("[pipeline] step=%d/%d accum=%d/%d phase=%s group=%d/%d budget=%d/%d\n",
           s->current_step, s->total_steps,
           s->accum_step, s->config.compile.accum_steps,
           phase_name(s->phase), s->current_group, s->plan.n_groups,
           s->budget.used, s->budget.budget);
}

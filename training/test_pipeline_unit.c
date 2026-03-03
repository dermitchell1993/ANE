// test_pipeline_unit.c — Unit tests for pipeline scheduler + checkpoint manager
// Pure C, no ANE dependency. Validates state machine transitions and checkpoint logic.
// Build: cc -O2 -o test_pipeline_unit test_pipeline_unit.c -lm
// Run:   ./test_pipeline_unit
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// Stub out mmap/exec dependencies — we only test the pure logic
#define _PIPELINE_SKIP_MMAP 1

#include "model_config.h"
#include "gradient_checkpoint.h"

// ===== Test helpers =====

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s", name); \
} while(0)

#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { FAIL(msg); printf("    got %d, expected %d\n", (int)(a), (int)(b)); return; } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

// ===== model_config.h tests =====

static void test_dims_init(void) {
    TEST("model_dims_init computes derived fields");
    ModelDims d = {.dim = 768, .n_heads = 12, .n_kv_heads = 12, .seq_len = 256};
    model_dims_init(&d);
    ASSERT_EQ(d.head_dim, 64, "head_dim = dim / n_heads");
    ASSERT_EQ(d.kv_dim, 768, "kv_dim = head_dim * n_kv_heads");
    ASSERT_EQ(d.score_ch, 12 * 256, "score_ch = n_heads * seq_len");
    PASS();
}

static void test_stories110m_preset(void) {
    TEST("Stories110M preset");
    ModelConfig cfg = model_config_stories110m();
    ASSERT_EQ(cfg.dims.dim, 768, "dim");
    ASSERT_EQ(cfg.dims.n_layers, 12, "n_layers");
    ASSERT_EQ(cfg.dims.n_heads, 12, "n_heads");
    ASSERT_EQ(cfg.compile.compile_budget, 119, "compile_budget");
    ASSERT_TRUE(cfg.compile.headroom_pct > 0.0f, "headroom > 0");
    PASS();
}

static void test_llama7b_preset(void) {
    TEST("LLaMA-7B preset");
    ModelConfig cfg = model_config_llama_7b();
    ASSERT_EQ(cfg.dims.dim, 4096, "dim");
    ASSERT_EQ(cfg.dims.n_layers, 32, "n_layers");
    ASSERT_EQ(cfg.dims.hidden_dim, 11008, "hidden_dim");
    PASS();
}

static void test_layer_memory_nonzero(void) {
    TEST("Per-layer memory sizes are nonzero");
    ModelConfig cfg = model_config_stories110m();
    ASSERT_TRUE(layer_weight_bytes(&cfg.dims) > 0, "weight bytes");
    ASSERT_TRUE(layer_adam_bytes(&cfg.dims) > 0, "adam bytes");
    ASSERT_TRUE(layer_activation_bytes(&cfg.dims) > 0, "activation bytes");
    ASSERT_TRUE(layer_gradient_bytes(&cfg.dims) > 0, "gradient bytes");
    ASSERT_TRUE(total_model_bytes(&cfg) > 0, "total model bytes");
    PASS();
}

static void test_adam_is_2x_weights(void) {
    TEST("Adam state = 2x weight size");
    ModelConfig cfg = model_config_stories110m();
    ASSERT_EQ(layer_adam_bytes(&cfg.dims), 2 * layer_weight_bytes(&cfg.dims), "adam = 2 * weights");
    PASS();
}

// ===== Pipeline planning tests =====

static void test_max_layers_per_compile(void) {
    TEST("max_layers_per_compile respects budget");
    CompileConfig cc = {.compile_budget = 119, .kernels_per_layer = 5,
                        .static_per_layer = 1, .headroom_pct = 0.10f};
    int max = max_layers_per_compile(&cc);
    // usable = floor(119 * 0.9) = 107, per_layer = 6, max = 107/6 = 17
    ASSERT_EQ(max, 17, "max layers = 17 for budget=119, 6 kernels/layer, 10% headroom");
    PASS();
}

static void test_configurable_headroom(void) {
    TEST("Configurable headroom changes max layers");
    CompileConfig cc5 = {.compile_budget = 119, .kernels_per_layer = 5,
                         .static_per_layer = 1, .headroom_pct = 0.05f};
    CompileConfig cc20 = {.compile_budget = 119, .kernels_per_layer = 5,
                          .static_per_layer = 1, .headroom_pct = 0.20f};
    int max5 = max_layers_per_compile(&cc5);    // floor(119*0.95/6) = 18
    int max20 = max_layers_per_compile(&cc20);   // floor(119*0.80/6) = 15
    ASSERT_TRUE(max5 > max20, "5% headroom fits more layers than 20%");
    ASSERT_EQ(max5, 18, "5% headroom: 18 layers");
    ASSERT_EQ(max20, 15, "20% headroom: 15 layers");
    PASS();
}

static void test_invalid_headroom_defaults(void) {
    TEST("Invalid headroom falls back to 10%");
    CompileConfig cc_neg = {.compile_budget = 119, .kernels_per_layer = 5,
                            .static_per_layer = 1, .headroom_pct = -0.5f};
    CompileConfig cc_over = {.compile_budget = 119, .kernels_per_layer = 5,
                             .static_per_layer = 1, .headroom_pct = 1.5f};
    CompileConfig cc_def = {.compile_budget = 119, .kernels_per_layer = 5,
                            .static_per_layer = 1, .headroom_pct = 0.10f};
    ASSERT_EQ(max_layers_per_compile(&cc_neg), max_layers_per_compile(&cc_def),
              "negative headroom -> default");
    ASSERT_EQ(max_layers_per_compile(&cc_over), max_layers_per_compile(&cc_def),
              "headroom > 1.0 -> default");
    PASS();
}

static void test_plan_stories110m(void) {
    TEST("Stories110M fits in 1 group");
    ModelConfig cfg = model_config_stories110m();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    ASSERT_EQ(plan.n_groups, 1, "1 group");
    ASSERT_EQ(plan.groups[0].start_layer, 0, "starts at 0");
    ASSERT_EQ(plan.groups[0].end_layer, 12, "ends at 12");
    ASSERT_EQ(plan.groups[0].n_layers, 12, "12 layers");
    ASSERT_EQ(plan.groups[0].total_kernels, 72, "72 total kernels");
    pipeline_plan_free(&plan);
    PASS();
}

static void test_plan_llama7b_multiple_groups(void) {
    TEST("LLaMA-7B needs multiple groups");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    ASSERT_TRUE(plan.n_groups >= 2, "at least 2 groups for 32 layers");
    // Verify all layers covered
    int total_layers = 0;
    for (int g = 0; g < plan.n_groups; g++) {
        total_layers += plan.groups[g].n_layers;
        ASSERT_TRUE(plan.groups[g].n_layers > 0, "no empty groups");
    }
    ASSERT_EQ(total_layers, 32, "all 32 layers covered");
    // Verify contiguous
    for (int g = 1; g < plan.n_groups; g++) {
        ASSERT_EQ(plan.groups[g].start_layer, plan.groups[g-1].end_layer, "contiguous groups");
    }
    pipeline_plan_free(&plan);
    PASS();
}

static void test_plan_kernel_budget(void) {
    TEST("No group exceeds compile budget");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    int usable = (int)(cfg.compile.compile_budget * (1.0f - cfg.compile.headroom_pct));
    for (int g = 0; g < plan.n_groups; g++) {
        ASSERT_TRUE(plan.groups[g].total_kernels <= usable,
                    "group kernel count <= usable budget");
    }
    pipeline_plan_free(&plan);
    PASS();
}

// ===== Gradient checkpoint tests =====

static void test_ckpt_all_saves_everything(void) {
    TEST("CKPT_ALL saves all layers");
    ModelConfig cfg = model_config_stories110m();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointManager cm = checkpoint_init(CKPT_ALL, &cfg, &plan, 0);
    ASSERT_EQ(cm.n_checkpointed, 12, "12 layers saved");
    for (int i = 0; i < 12; i++) {
        ASSERT_TRUE(checkpoint_should_save(&cm, i), "every layer saved");
        ASSERT_TRUE(!checkpoint_needs_recompute(&cm, i), "no recompute needed");
    }
    ASSERT_TRUE(checkpoint_recompute_overhead(&cm) < 0.001, "zero overhead");
    checkpoint_free(&cm);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_none_saves_minimum(void) {
    TEST("CKPT_NONE saves only layer 0");
    ModelConfig cfg = model_config_stories110m();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointManager cm = checkpoint_init(CKPT_NONE, &cfg, &plan, 0);
    ASSERT_EQ(cm.n_checkpointed, 1, "only 1 layer saved");
    ASSERT_TRUE(checkpoint_should_save(&cm, 0), "layer 0 saved");
    ASSERT_TRUE(checkpoint_needs_recompute(&cm, 5), "layer 5 needs recompute");
    checkpoint_free(&cm);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_sqrt_interval(void) {
    TEST("CKPT_SQRT uses sqrt(N) interval");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointManager cm = checkpoint_init(CKPT_SQRT, &cfg, &plan, 0);
    int expected_interval = (int)sqrtf(32.0f);  // 5
    ASSERT_EQ(cm.interval, expected_interval, "interval = sqrt(32) = 5");
    // Layer 0 always saved, then 5, 10, 15, 20, 25, 30, 31
    ASSERT_TRUE(checkpoint_should_save(&cm, 0), "layer 0 saved");
    ASSERT_TRUE(checkpoint_should_save(&cm, 5), "layer 5 saved");
    ASSERT_TRUE(!checkpoint_should_save(&cm, 3), "layer 3 not saved");
    ASSERT_TRUE(checkpoint_should_save(&cm, 31), "last layer saved");
    checkpoint_free(&cm);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_boundary(void) {
    TEST("CKPT_BOUNDARY saves group edges");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointManager cm = checkpoint_init(CKPT_BOUNDARY, &cfg, &plan, 0);
    // First layer of each group + last layer overall
    for (int g = 0; g < plan.n_groups; g++) {
        ASSERT_TRUE(checkpoint_should_save(&cm, plan.groups[g].start_layer),
                    "group start layer saved");
    }
    ASSERT_TRUE(checkpoint_should_save(&cm, 31), "last layer saved");
    // Middle of first group should not be saved
    if (plan.groups[0].n_layers > 2) {
        int mid = plan.groups[0].start_layer + plan.groups[0].n_layers / 2;
        ASSERT_TRUE(checkpoint_needs_recompute(&cm, mid), "mid-group needs recompute");
    }
    checkpoint_free(&cm);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_memory_savings(void) {
    TEST("Checkpoint memory savings are positive for non-ALL policies");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);

    CheckpointManager cm_all = checkpoint_init(CKPT_ALL, &cfg, &plan, 0);
    CheckpointManager cm_sqrt = checkpoint_init(CKPT_SQRT, &cfg, &plan, 0);
    CheckpointManager cm_none = checkpoint_init(CKPT_NONE, &cfg, &plan, 0);

    size_t saved_sqrt = checkpoint_memory_saved(&cm_sqrt, &cfg.dims);
    size_t saved_none = checkpoint_memory_saved(&cm_none, &cfg.dims);

    ASSERT_TRUE(saved_sqrt > 0, "SQRT saves memory");
    ASSERT_TRUE(saved_none > saved_sqrt, "NONE saves more than SQRT");
    ASSERT_EQ(checkpoint_memory_saved(&cm_all, &cfg.dims), 0, "ALL saves nothing");

    checkpoint_free(&cm_all);
    checkpoint_free(&cm_sqrt);
    checkpoint_free(&cm_none);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_recompute_depth(void) {
    TEST("Recompute depth counts layers from nearest checkpoint");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointManager cm = checkpoint_init(CKPT_SQRT, &cfg, &plan, 0);
    // With interval=5: checkpoints at 0, 5, 10, 15, 20, 25, 30, 31
    // Layer 3: nearest saved before = 0, depth = 3
    ASSERT_EQ(checkpoint_recompute_depth(&cm, 3), 3, "depth from layer 0 to 3");
    // Layer 7: nearest saved before = 5, depth = 2
    ASSERT_EQ(checkpoint_recompute_depth(&cm, 7), 2, "depth from layer 5 to 7");
    // Layer 5: nearest saved = 5, depth = 0
    ASSERT_EQ(checkpoint_recompute_depth(&cm, 5), 0, "checkpointed layer = 0 depth");
    checkpoint_free(&cm);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_out_of_bounds(void) {
    TEST("Checkpoint queries handle out-of-bounds gracefully");
    ModelConfig cfg = model_config_stories110m();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointManager cm = checkpoint_init(CKPT_ALL, &cfg, &plan, 0);
    ASSERT_TRUE(!checkpoint_should_save(&cm, -1), "negative index returns false");
    ASSERT_TRUE(!checkpoint_should_save(&cm, 100), "over-max index returns false");
    checkpoint_free(&cm);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_every_n_custom_interval(void) {
    TEST("CKPT_EVERY_N respects custom_interval parameter");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointManager cm3 = checkpoint_init(CKPT_EVERY_N, &cfg, &plan, 3);
    CheckpointManager cm8 = checkpoint_init(CKPT_EVERY_N, &cfg, &plan, 8);
    ASSERT_EQ(cm3.interval, 3, "interval=3 when custom_interval=3");
    ASSERT_EQ(cm8.interval, 8, "interval=8 when custom_interval=8");
    ASSERT_TRUE(cm3.n_checkpointed > cm8.n_checkpointed,
                "shorter interval = more checkpoints");
    // Verify layer 0 and last layer always saved
    ASSERT_TRUE(checkpoint_should_save(&cm3, 0), "layer 0 saved (interval=3)");
    ASSERT_TRUE(checkpoint_should_save(&cm3, 31), "last layer saved (interval=3)");
    ASSERT_TRUE(checkpoint_should_save(&cm8, 0), "layer 0 saved (interval=8)");
    ASSERT_TRUE(checkpoint_should_save(&cm8, 31), "last layer saved (interval=8)");
    checkpoint_free(&cm3);
    checkpoint_free(&cm8);
    pipeline_plan_free(&plan);
    PASS();
}

static void test_ckpt_n_checkpointed_accuracy(void) {
    TEST("n_checkpointed matches actual is_saved bit count");
    ModelConfig cfg = model_config_llama_7b();
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    CheckpointPolicy policies[] = {CKPT_ALL, CKPT_BOUNDARY, CKPT_SQRT, CKPT_EVERY_N, CKPT_NONE};
    for (int p = 0; p < 5; p++) {
        CheckpointManager cm = checkpoint_init(policies[p], &cfg, &plan, 0);
        int actual = 0;
        for (int i = 0; i < cm.n_layers; i++) {
            if (cm.is_saved[i]) actual++;
        }
        ASSERT_EQ(cm.n_checkpointed, actual, "n_checkpointed matches is_saved count");
        checkpoint_free(&cm);
    }
    pipeline_plan_free(&plan);
    PASS();
}

static void test_dims_init_zero_heads(void) {
    TEST("model_dims_init guards divide-by-zero on n_heads=0");
    ModelDims d = {.dim = 768, .n_heads = 0, .n_kv_heads = 0, .seq_len = 256};
    model_dims_init(&d);
    ASSERT_EQ(d.head_dim, 0, "head_dim=0 when n_heads=0");
    ASSERT_EQ(d.kv_dim, 0, "kv_dim=0 when n_heads=0");
    ASSERT_EQ(d.score_ch, 0, "score_ch=0 when n_heads=0");
    PASS();
}

// ===== FLOP estimation tests =====

static void test_flops_nonzero(void) {
    TEST("FLOP estimates are nonzero and ANE < total");
    ModelConfig cfg = model_config_stories110m();
    double total = flops_per_step(&cfg);
    double ane = ane_flops_per_step(&cfg);
    ASSERT_TRUE(total > 0, "total FLOPs > 0");
    ASSERT_TRUE(ane > 0, "ANE FLOPs > 0");
    ASSERT_TRUE(ane < total, "ANE FLOPs < total (dW is on CPU)");
    PASS();
}

static void test_flops_scale_with_layers(void) {
    TEST("FLOPs scale roughly linearly with layer count");
    ModelConfig cfg12 = model_config_stories110m();
    ModelConfig cfg8 = model_config_stories42m();
    double f12 = flops_per_step(&cfg12);
    double f8 = flops_per_step(&cfg8);
    // Not exact linear due to different dims, but 12-layer should be >8-layer
    ASSERT_TRUE(f12 > f8, "12 layers > 8 layers");
    PASS();
}

// ===== Pipeline plan edge cases =====

static void test_plan_single_layer(void) {
    TEST("Single-layer model = 1 group");
    ModelConfig cfg = model_config_stories110m();
    cfg.dims.n_layers = 1;
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    ASSERT_EQ(plan.n_groups, 1, "1 group");
    ASSERT_EQ(plan.groups[0].n_layers, 1, "1 layer in group");
    pipeline_plan_free(&plan);
    PASS();
}

static void test_plan_exact_budget_fit(void) {
    TEST("Layers that exactly fill budget = 1 group");
    ModelConfig cfg = model_config_stories110m();
    // 17 layers * 6 kernels = 102 <= 107 usable (10% headroom on 119)
    cfg.dims.n_layers = 17;
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    ASSERT_EQ(plan.n_groups, 1, "17 layers fit in 1 group");
    pipeline_plan_free(&plan);
    PASS();
}

static void test_plan_one_over_budget(void) {
    TEST("One layer over budget = 2 groups");
    ModelConfig cfg = model_config_stories110m();
    // 18 layers * 6 kernels = 108 > 107 usable -> 2 groups
    cfg.dims.n_layers = 18;
    PipelinePlan plan = compute_pipeline_plan(&cfg);
    ASSERT_EQ(plan.n_groups, 2, "18 layers = 2 groups");
    int total = plan.groups[0].n_layers + plan.groups[1].n_layers;
    ASSERT_EQ(total, 18, "all layers covered");
    pipeline_plan_free(&plan);
    PASS();
}

// ===== Main =====

int main(void) {
    printf("=== Pipeline Unit Tests ===\n\n");

    printf("[model_config.h]\n");
    test_dims_init();
    test_stories110m_preset();
    test_llama7b_preset();
    test_layer_memory_nonzero();
    test_adam_is_2x_weights();

    printf("\n[pipeline planning]\n");
    test_max_layers_per_compile();
    test_configurable_headroom();
    test_invalid_headroom_defaults();
    test_plan_stories110m();
    test_plan_llama7b_multiple_groups();
    test_plan_kernel_budget();
    test_plan_single_layer();
    test_plan_exact_budget_fit();
    test_plan_one_over_budget();

    printf("\n[gradient_checkpoint.h]\n");
    test_ckpt_all_saves_everything();
    test_ckpt_none_saves_minimum();
    test_ckpt_sqrt_interval();
    test_ckpt_boundary();
    test_ckpt_memory_savings();
    test_ckpt_recompute_depth();
    test_ckpt_out_of_bounds();
    test_ckpt_every_n_custom_interval();
    test_ckpt_n_checkpointed_accuracy();
    test_dims_init_zero_heads();

    printf("\n[FLOP estimation]\n");
    test_flops_nonzero();
    test_flops_scale_with_layers();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

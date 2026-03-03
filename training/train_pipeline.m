// train_pipeline.m — Pipeline-scheduled multi-group ANE training
//
// Entry point that uses the pipeline scaffolding to train models
// beyond the single-compile-batch limit.
//
// Architecture:
//   ModelConfig  → what the model looks like
//   PipelinePlan → which layers go in which compile groups
//   PipelineScheduler → state machine driving forward/backward/restart
//   MmapState    → cross-exec() shared memory for all tensor state
//   CheckpointManager → activation save/recompute policy
//
// Usage:
//   ./train_pipeline --model stories110m --steps 100 --lr 3e-4
//   ./train_pipeline --model llama1b --steps 50 --lr 1e-4 --checkpoint sqrt
//   ./train_pipeline --pipeline-resume /tmp/ane_pipeline.mmap   (auto after exec restart)
//
// Build:
//   make train_pipeline
//
// Currently runs in planning/dry-run mode — prints the full execution
// plan and simulates the scheduler state machine without compiling
// actual MIL programs. Enable ANE_LIVE for real kernels.

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <mach/mach_time.h>

#include "model_config.h"
#include "pipeline.h"
#include "gradient_checkpoint.h"

#define MMAP_PATH "/tmp/ane_pipeline.mmap"

// ===== Forward declarations for ANE kernel operations =====
// These would call into stories_io.h / stories_mil.h for real execution.
// Stubbed here for planning mode.

#ifdef ANE_LIVE
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops.h"
// Real ANE kernel compilation and execution would go here
#endif

// ===== Dry-run simulation =====

static void simulate_compile_group(const PipelineScheduler *s, const LayerGroup *lg) {
    printf("    [compile] Layers [%d..%d): %d weight-bearing + %d static kernels\n",
           lg->start_layer, lg->end_layer, lg->weight_kernels, lg->static_kernels);
    printf("             Budget: %d/%d used → %d/%d after\n",
           s->budget.used, s->budget.budget,
           s->budget.used + lg->total_kernels, s->budget.budget);
}

static void simulate_forward_group(const PipelineScheduler *s, const LayerGroup *lg,
                                    const CheckpointManager *cm) {
    printf("    [forward] Layers [%d..%d)\n", lg->start_layer, lg->end_layer);
    for (int L = lg->start_layer; L < lg->end_layer; L++) {
        bool save = checkpoint_should_save(cm, L);
        printf("      L%02d: fwdAttn → residual → fwdFFN → residual %s\n",
               L, save ? "[SAVE acts]" : "[skip acts]");
    }
}

static void simulate_backward_group(const PipelineScheduler *s, const LayerGroup *lg,
                                      const CheckpointManager *cm) {
    printf("    [backward] Layers [%d..%d) (reverse)\n", lg->start_layer, lg->end_layer);
    for (int L = lg->end_layer - 1; L >= lg->start_layer; L--) {
        bool recompute = checkpoint_needs_recompute(cm, L);
        if (recompute) {
            int from = checkpoint_nearest_saved_before(cm, L);
            printf("      L%02d: [RECOMPUTE from L%02d] → ffnBwd → rmsnorm2_bwd → sdpaBwd1 → sdpaBwd2 → qkvBwd → rmsnorm1_bwd\n",
                   L, from);
        } else {
            printf("      L%02d: ffnBwd → rmsnorm2_bwd → sdpaBwd1 → sdpaBwd2 → qkvBwd → rmsnorm1_bwd\n", L);
        }
    }
}

// ===== Main =====

int main(int argc, char *argv[]) {
    @autoreleasepool {
        // Parse model config from command line
        ModelConfig cfg = model_config_from_args(argc, argv);

        // Parse additional training args
        int total_steps = 100;
        float lr = 3e-4f;
        bool dry_run = true;
        CheckpointPolicy ckpt_policy = CKPT_ALL;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--live") == 0) dry_run = false;
            else if (strcmp(argv[i], "--checkpoint") == 0 && i+1 < argc) {
                const char *p = argv[++i];
                if (strcmp(p, "all") == 0) ckpt_policy = CKPT_ALL;
                else if (strcmp(p, "boundary") == 0) ckpt_policy = CKPT_BOUNDARY;
                else if (strcmp(p, "sqrt") == 0) ckpt_policy = CKPT_SQRT;
                else if (strcmp(p, "none") == 0) ckpt_policy = CKPT_NONE;
                else fprintf(stderr, "Unknown checkpoint policy: %s\n", p);
            }
        }

        // Check for exec() resume
        PipelineScheduler sched = pipeline_scheduler_init(cfg, total_steps, lr);
        MmapState *ms = NULL;

        if (pipeline_check_resume(argc, argv, &sched, &ms)) {
            printf("[pipeline] Resumed from exec() restart\n");
        } else {
            // Fresh start
            printf("=== ANE Pipeline Training ===\n");
            if (dry_run) printf("  ** DRY RUN MODE — no ANE kernels compiled **\n\n");

            // Print model config and pipeline plan
            PipelinePlan plan = compute_pipeline_plan(&cfg);
            pipeline_plan_print(&cfg, &plan);
            printf("\n");

            // Print checkpoint policy
            CheckpointManager cm = checkpoint_init(ckpt_policy, &cfg, &plan);
            checkpoint_print(&cm, &cfg.dims);
            printf("\n");

            // Print FLOP estimates
            double total_flops = flops_per_step(&cfg);
            double ane_flops = ane_flops_per_step(&cfg);
            printf("=== Compute Estimate ===\n");
            printf("  FLOPs/step: %.0fM total, %.0fM ANE (%.0f%% on-engine)\n",
                   total_flops/1e6, ane_flops/1e6, 100.0*ane_flops/total_flops);
            printf("  At 15.8 TFLOPS ANE: %.1f ms/step theoretical minimum\n",
                   ane_flops / 15.8e9);
            printf("  Training: %d steps × %d accum = %d optimizer updates\n",
                   total_steps, cfg.compile.accum_steps, total_steps / cfg.compile.accum_steps);
            printf("\n");

            // Print mmap state size
            size_t mmap_sz = mmap_compute_size(&cfg);
            printf("=== State Management ===\n");
            printf("  mmap file: %s (%.1fMB)\n", MMAP_PATH, mmap_sz/1e6);
            printf("  Per-layer: weights=%.1fMB adam=%.1fMB grads=%.1fMB acts=%.1fMB\n",
                   layer_weight_bytes(&cfg.dims)/1e6, layer_adam_bytes(&cfg.dims)/1e6,
                   layer_gradient_bytes(&cfg.dims)/1e6, layer_activation_bytes(&cfg.dims)/1e6);
            printf("\n");

            // Create mmap state
            ms = mmap_state_create(MMAP_PATH, &cfg);
            if (!ms) {
                fprintf(stderr, "Failed to create mmap state\n");
                checkpoint_free(&cm);
                pipeline_plan_free(&plan);
                return 1;
            }

            if (dry_run) {
                // ===== Simulate the full scheduler state machine =====
                printf("=== Execution Trace (1 training step) ===\n");
                int max_actions = 200;  // safety limit
                int action_count = 0;

                while (action_count < max_actions) {
                    PipelineAction action = pipeline_next_action(&sched);
                    action_count++;

                    printf("\n  [%d] %s (phase=%s group=%d)\n",
                           action_count, action_name(action),
                           phase_name(sched.phase), sched.current_group);

                    switch (action) {
                    case ACTION_COMPILE_GROUP: {
                        LayerGroup *lg = &sched.plan.groups[sched.current_group];
                        simulate_compile_group(&sched, lg);
                        pipeline_group_compiled(&sched);
                        break;
                    }
                    case ACTION_RUN_FORWARD_GROUP: {
                        LayerGroup *lg = &sched.plan.groups[sched.current_group];
                        simulate_forward_group(&sched, lg, &cm);
                        pipeline_forward_group_done(&sched);
                        break;
                    }
                    case ACTION_RUN_BACKWARD_GROUP: {
                        LayerGroup *lg = &sched.plan.groups[sched.current_group];
                        simulate_backward_group(&sched, lg, &cm);
                        pipeline_backward_group_done(&sched);
                        break;
                    }
                    case ACTION_EXEC_RESTART:
                        printf("    [exec] Would restart process to reset compile budget\n");
                        printf("           Saving scheduler state to mmap, calling exec()\n");
                        // In dry-run, just reset the budget and continue
                        sched.budget = budget_init(cfg.compile.compile_budget);
                        sched.needs_restart = false;
                        break;

                    case ACTION_WEIGHT_UPDATE:
                        printf("    [adam] Optimizer step on all %d layers + global params\n",
                               cfg.dims.n_layers);
                        printf("           LR=%.1e adam_t=%d\n", sched.learning_rate, sched.current_step+1);
                        pipeline_weight_update_done(&sched);
                        break;

                    case ACTION_STEP_DONE:
                        printf("\n=== Training step complete ===\n");
                        goto done_trace;

                    case ACTION_ERROR:
                        printf("    ERROR in scheduler\n");
                        goto done_trace;
                    }
                }
                done_trace:

                printf("\nTotal actions simulated: %d\n", action_count);
                printf("Compile budget consumed: %d/%d\n", sched.budget.used, sched.budget.budget);

                // Summary for multi-group models
                if (plan.n_groups > 1) {
                    printf("\n=== Multi-Group Pipeline Summary ===\n");
                    printf("  This model requires %d layer groups per training step\n", plan.n_groups);
                    printf("  Forward pass: %d compile batches (left to right)\n", plan.n_groups);
                    printf("  Backward pass: %d compile batches (right to left)\n", plan.n_groups);
                    printf("  Each compile batch may need exec() restart\n");
                    printf("  All tensor state survives restarts via mmap (%s)\n", MMAP_PATH);
                    printf("  Checkpoint policy '%s' saves %d/%d layer activations (%.0f%% memory reduction)\n",
                           checkpoint_policy_name(ckpt_policy), cm.n_checkpointed, cm.n_layers,
                           100.0 * checkpoint_memory_saved(&cm, &cfg.dims) /
                           ((double)cm.n_layers * layer_activation_bytes(&cfg.dims)));
                }

                checkpoint_free(&cm);
            } else {
                // ===== Live training mode =====
#ifdef ANE_LIVE
                printf("Live training not yet implemented — use train_large.m for Stories110M\n");
                printf("This entry point will be wired up once the scaffolding is validated.\n");
#else
                printf("Compiled without ANE_LIVE — use --live with ANE_LIVE defined.\n");
                printf("Build with: xcrun clang -DANE_LIVE -O2 ... train_pipeline.m\n");
#endif
                checkpoint_free(&cm);
            }

            pipeline_plan_free(&plan);
        }

        // Cleanup
        if (ms) mmap_state_destroy(ms);
    }
    return 0;
}


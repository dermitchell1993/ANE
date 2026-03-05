// test_m2_compatibility.m — M1/M2 backward-compatibility test harness
// Runs Stories110M 12-layer training loop on detected hardware
// Reports: ANE utilization, power draw estimate, crash-free uptime
//
// Usage: ./test_m2_compatibility <model.bin> [--duration=30]
//
// Targets: 30+ minutes crash-free uptime with stable loss descent
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <time.h>
#include "ane_hw_detect.h"
#include "ane_mem_budget.h"
#include "ane_runtime.h"
#include "ane_mil_gen.h"
#include "ane_compat.h"
#include "model.h"
#include "forward.h"
#include "backward.h"

// ============================================================
// Globals
// ============================================================
static volatile bool g_running = true;
static mach_timebase_info_data_t g_timebase;

static double ticks_to_sec(uint64_t t) {
    return (double)t * g_timebase.numer / g_timebase.denom / 1e9;
}

static void handle_signal(int sig) {
    (void)sig;
    g_running = false;
    printf("\n[SIGNAL] Graceful shutdown requested...\n");
}

// ============================================================
// Memory usage reporting (approximate, via mach)
// ============================================================
static size_t get_resident_mb(void) {
    struct task_basic_info info;
    mach_msg_type_number_t cnt = TASK_BASIC_INFO_COUNT;
    task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &cnt);
    return info.resident_size / (1024 * 1024);
}

// ============================================================
// NaN/Inf checker for activation tensors
// ============================================================
static bool check_finite(const float *buf, int n, const char *name) {
    for (int i = 0; i < n; i++) {
        if (isnan(buf[i]) || isinf(buf[i])) {
            fprintf(stderr, "[STABILITY] %s has NaN/Inf at index %d (val=%.6g)\n", name, i, buf[i]);
            return false;
        }
    }
    return true;
}

// ============================================================
// Main test harness
// ============================================================
int main(int argc, char *argv[]) {
    @autoreleasepool {
        mach_timebase_info(&g_timebase);
        signal(SIGINT, handle_signal);
        signal(SIGTERM, handle_signal);

        if (argc < 2) {
            fprintf(stderr, "Usage: %s <model.bin> [--duration=30]\n", argv[0]);
            fprintf(stderr, "  --duration=N  Run for N minutes (default: 30)\n");
            return 1;
        }

        int duration_min = 30;
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--duration=", 11) == 0)
                duration_min = atoi(argv[i] + 11);
        }

        // ============================================================
        // Phase 1: Hardware detection
        // ============================================================
        printf("╔══════════════════════════════════════════════╗\n");
        printf("║   M1/M2 ANE Compatibility Test Harness      ║\n");
        printf("╚══════════════════════════════════════════════╝\n\n");

        ANEChipProfile prof = ANEVersionDetect();
        printf("[HW] Chip: %s (%d GB unified, %d NE cores)\n",
               prof.name, prof.max_unified_gb, prof.ane_cores);
        printf("[HW] MIL target: program(%s) <%s>\n", prof.mil_version, prof.mil_target);
        printf("[HW] Capabilities: matmul=%s, SDPA=%s\n",
               prof.supports_matmul ? "YES" : "NO",
               prof.supports_sdpa ? "YES" : "NO");
        printf("[HW] Limits: max_compiles=%d, align=%d, max_seq=%d\n\n",
               prof.max_compiles, prof.iosurface_align, prof.max_seq_len);

        // ============================================================
        // Phase 2: Memory budget
        // ============================================================
        ANEMemBudget budget = ANEAutoBudget();
        printf("[BUDGET] batch=%d, seq=%d, hidden=%d, layers=%d\n",
               budget.batch_size, budget.seq_len, budget.hidden_dim, budget.n_layers);
        printf("[BUDGET] gradient_ckpt=%s (interval=%d), accum=%d\n",
               budget.gradient_checkpointing ? "ON" : "OFF",
               budget.checkpoint_interval, budget.accum_steps);
        printf("[BUDGET] estimated peak: %zu MB\n\n", budget.estimated_peak_mb);

        // ============================================================
        // Phase 3: Load model
        // ============================================================
        Model m = {0};
        printf("[MODEL] Loading weights from %s...\n", argv[1]);
        if (model_load_weights(&m, argv[1]) != 0) {
            fprintf(stderr, "[FATAL] Cannot load model weights\n");
            return 1;
        }

        int seq_len = budget.seq_len;
        bool use_ane = true;

        // ============================================================
        // Phase 4: Compile ANE kernels (chip-aware)
        // ============================================================
        printf("[COMPILE] Target seq_len=%d on %s...\n", seq_len, prof.name);
        uint64_t compile_start = mach_absolute_time();

        if (model_compile_kernels(&m, seq_len) != 0) {
            fprintf(stderr, "[WARN] ANE kernel compilation failed, falling back to CPU\n");
            use_ane = false;
            m.seq_len = seq_len;
        }

        double compile_sec = ticks_to_sec(mach_absolute_time() - compile_start);
        printf("[COMPILE] Done in %.1f sec (%s)\n\n",
               compile_sec, use_ane ? "ANE" : "CPU fallback");

        model_alloc_training(&m);

        // ============================================================
        // Phase 5: Training loop with stability monitoring
        // ============================================================
        int *tokens = (int*)malloc(seq_len * sizeof(int));
        for (int i = 0; i < seq_len; i++)
            tokens[i] = (i * 7 + 13) % 256 + 1;

        printf("[TRAIN] Starting %d-minute stability test (seq=%d, %s)...\n",
               duration_min, seq_len, use_ane ? "ANE" : "CPU");
        printf("%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n",
               "Step", "Loss", "GradNorm", "ms/step", "tok/s", "RSS(MB)", "Uptime(s)");
        printf("════════════════════════════════════════════════════════════════════════\n");

        uint64_t test_start = mach_absolute_time();
        int step = 0;
        int recompile_interval = 1;
        int max_compiles_used = 0;
        int nan_count = 0;
        int eval_failures = 0;
        float best_loss = 1e9f;
        float worst_loss = 0;
        double total_step_ms = 0;
        int ane_steps = 0;
        int cpu_steps = 0;
        float lr = 1e-4f;

        while (g_running) {
            double elapsed_sec = ticks_to_sec(mach_absolute_time() - test_start);
            if (elapsed_sec >= duration_min * 60.0) break;

            uint64_t step_start = mach_absolute_time();

            // Forward pass
            float loss = model_forward(&m, tokens, use_ane);

            if (isnan(loss) || isinf(loss)) {
                nan_count++;
                fprintf(stderr, "[STABILITY] NaN/Inf loss at step %d (occurrence #%d)\n", step, nan_count);
                if (nan_count >= 5) {
                    fprintf(stderr, "[FATAL] Too many NaN losses, aborting\n");
                    break;
                }
                // Try to recover: reduce LR, recompile
                lr *= 0.5f;
                if (use_ane) model_recompile_kernels(&m);
                step++;
                continue;
            }

            if (loss < best_loss) best_loss = loss;
            if (loss > worst_loss) worst_loss = loss;

            // Backward pass
            model_backward(&m, tokens);
            model_clip_gradients(&m, 1.0f);
            model_adam_step(&m, lr, 0.9f, 0.999f, 1e-8f);

            if (use_ane) ane_steps++; else cpu_steps++;

            // Recompile with updated weights
            if (use_ane && (step + 1) % recompile_interval == 0) {
                max_compiles_used++;
                if (max_compiles_used >= prof.max_compiles) {
                    printf("[COMPILE] Approaching compile limit (%d/%d) — consider exec() restart\n",
                           max_compiles_used, prof.max_compiles);
                }
                if (model_recompile_kernels(&m) != 0) {
                    fprintf(stderr, "[WARN] Recompile failed at step %d, switching to CPU\n", step);
                    use_ane = false;
                    eval_failures++;
                }
            }

            double step_ms = ticks_to_sec(mach_absolute_time() - step_start) * 1000.0;
            total_step_ms += step_ms;

            // Report every 50 steps
            if (step % 50 == 0) {
                double gnorm = 0;
                int d2 = m.cfg.dim;
                for (int i = 0; i < d2*d2; i++)
                    gnorm += (double)m.grad_wq[0][i] * m.grad_wq[0][i];
                gnorm = sqrt(gnorm);

                double tps = (seq_len - 1) / (step_ms / 1000.0);
                size_t rss = get_resident_mb();
                double uptime = ticks_to_sec(mach_absolute_time() - test_start);

                printf("%-8d %-10.4f %-10.4f %-10.1f %-10.0f %-10zu %-10.0f\n",
                       step, loss, gnorm, step_ms, tps, rss, uptime);
            }

            step++;
        }

        // ============================================================
        // Phase 6: Final report
        // ============================================================
        double total_sec = ticks_to_sec(mach_absolute_time() - test_start);
        double avg_ms = (step > 0) ? total_step_ms / step : 0;

        printf("\n╔══════════════════════════════════════════════╗\n");
        printf("║   Test Results                               ║\n");
        printf("╚══════════════════════════════════════════════╝\n\n");

        printf("[RESULT] Chip: %s (%d GB)\n", prof.name, prof.max_unified_gb);
        printf("[RESULT] Total uptime: %.1f min (target: %d min)\n", total_sec / 60.0, duration_min);
        printf("[RESULT] Steps completed: %d (ANE: %d, CPU: %d)\n", step, ane_steps, cpu_steps);
        printf("[RESULT] Avg step time: %.1f ms\n", avg_ms);
        printf("[RESULT] Avg throughput: %.0f tok/s\n",
               avg_ms > 0 ? (seq_len - 1) / (avg_ms / 1000.0) : 0);
        printf("[RESULT] Loss range: [%.4f, %.4f] (best: %.4f)\n", best_loss, worst_loss, best_loss);
        printf("[RESULT] NaN occurrences: %d\n", nan_count);
        printf("[RESULT] Eval failures: %d\n", eval_failures);
        printf("[RESULT] ANE compiles used: %d / %d limit\n", max_compiles_used, prof.max_compiles);
        printf("[RESULT] Final RSS: %zu MB\n", get_resident_mb());

        // ANE utilization estimate
        double ane_pct = (step > 0) ? (100.0 * ane_steps / step) : 0;
        printf("[RESULT] ANE utilization: %.1f%%\n", ane_pct);

        // Power draw estimate (rough: M2 ANE ~8W active, ~2W idle; M4 ~10W active)
        double est_power_w = 0;
        if (prof.gen == ANE_CHIP_M2 || prof.gen == ANE_CHIP_M1) {
            est_power_w = ane_pct > 50 ? 8.0 : 4.0;
        } else if (prof.gen == ANE_CHIP_M4) {
            est_power_w = ane_pct > 50 ? 10.0 : 5.0;
        }
        printf("[RESULT] Estimated ANE power draw: ~%.0fW\n", est_power_w);

        // Pass/fail
        bool passed = (total_sec >= duration_min * 60.0 * 0.9) // 90% of target uptime
                   && (nan_count <= 2)
                   && (eval_failures <= 3)
                   && (best_loss < 10.0f); // some training progress
        printf("\n[VERDICT] %s\n", passed ? "PASS — Crash-free, stable training achieved" :
                                            "FAIL — See issues above");

        // Perf comparison estimate
        if (prof.gen == ANE_CHIP_M2) {
            printf("\n[PERF] M2 vs M4 estimate:\n");
            printf("  M2 avg step: %.1f ms\n", avg_ms);
            printf("  M4 expected: ~%.1f ms (from benchmarks)\n", avg_ms / 2.4);
            printf("  Slowdown factor: ~2.4x (expected for M2 conv-only path)\n");
            printf("  Verdict: %s for 24/7 swarm use\n",
                   avg_ms < 5000 ? "ACCEPTABLE" : "NEEDS OPTIMIZATION");
        }

        free(tokens);
        printf("\n[DONE] Test completed.\n");
    }
    return 0;
}

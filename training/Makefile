CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

# Universal binary flags: M1 (arm64) through M4
# -arch arm64 covers all Apple Silicon generations
ARCH_FLAGS = -arch arm64
UNIVERSAL_CFLAGS = $(CFLAGS) $(ARCH_FLAGS)

# Header dependency groups
HEADERS_CORE = ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
HEADERS_COMPAT = ane_hw_detect.h ane_compat.h ane_mem_budget.h
HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h
HEADERS_ANE = $(HEADERS_LARGE) ane_rmsnorm_bwd.h ane_classifier.h

# === Primary targets ===

train: train.m $(HEADERS_CORE)
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ train.m $(LDFLAGS)

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

train_large_ane: train_large_ane.m $(HEADERS_ANE)
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ train_large_ane.m $(LDFLAGS) -framework Accelerate

# === M1/M2/M3/M4 compatibility test ===

test_m2_compatibility: test_m2_compatibility.m $(HEADERS_CORE) $(HEADERS_COMPAT)
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

# === Existing probes & tests ===

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced

test_rmsnorm_bwd: test_rmsnorm_bwd.m $(HEADERS_ANE)
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_classifier: test_classifier.m $(HEADERS_ANE)
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_weight_reload: test_weight_reload.m
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ $< $(LDFLAGS)

test_perf_stats: test_perf_stats.m
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ $< $(LDFLAGS)

test_qos_sweep: test_qos_sweep.m
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_advanced: test_ane_advanced.m
	$(CC) $(UNIVERSAL_CFLAGS) -o $@ $< $(LDFLAGS)

probes: $(PROBES)

# === Convenience targets ===

# Build everything for universal M1/M2/M3/M4 compatibility
all: train train_large train_large_ane test_m2_compatibility

# Quick compatibility check (compile only, no run)
compat-check: test_m2_compatibility
	@echo "Compatibility test binary built OK for all Apple Silicon generations"

tokenize:
	python3 tokenize.py

clean:
	rm -f train train_large train_large_ane test_m2_compatibility \
	      $(PROBES) test_rmsnorm_bwd test_classifier

.PHONY: clean tokenize probes all compat-check


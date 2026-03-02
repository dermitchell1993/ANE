CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h
HEADERS_PIPELINE = model_config.h pipeline.h gradient_checkpoint.h

train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) -o $@ train.m $(LDFLAGS)

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

train_pipeline: train_pipeline.m $(HEADERS_PIPELINE)
	$(CC) $(CFLAGS) -o $@ train_pipeline.m $(LDFLAGS) -framework Accelerate

train_pipeline_live: train_pipeline.m $(HEADERS_PIPELINE) $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -DANE_LIVE -o train_pipeline train_pipeline.m $(LDFLAGS) -framework Accelerate

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced

test_weight_reload: test_weight_reload.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_perf_stats: test_perf_stats.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_qos_sweep: test_qos_sweep.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_advanced: test_ane_advanced.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

probes: $(PROBES)

tokenize:
	python3 tokenize.py

test_pipeline_unit: test_pipeline_unit.c $(HEADERS_PIPELINE)
	cc -O2 -Wall -o $@ $< -lm

clean:
	rm -f train train_large train_pipeline test_pipeline_unit $(PROBES)

.PHONY: clean tokenize probes

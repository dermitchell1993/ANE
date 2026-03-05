// quant_pack.h — Q4/Q8 weight packing + NEON dequant for ANE weight-swap pipeline
// Phase 2 deliverable: enables 7B/13B models to fit in 24GB unified memory
//
// Architecture: weights stored as Q4/Q8 in system memory, dequanted to fp16
// on CPU (NEON) during the weight reload step. ANE kernels remain pure fp16.
//
// Q4 format: group quantization with group_size (default 128)
//   Per group: 1 fp16 scale, 1 fp16 zero_point, group_size/2 packed uint8 nibbles
//   Dequant: fp16_val = (nibble - 8) * scale + zero
//
// Performance target: <1ms per 7B layer on M2 (55MB Q4 -> 110MB fp16)
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Quantization types
typedef enum {
    QUANT_FP16 = 0,
    QUANT_Q8   = 1,
    QUANT_Q4   = 2,
    QUANT_Q3   = 3,
} QuantType;

// Per-group quantization parameters
typedef struct {
    _Float16 scale;
    _Float16 zero;
} QuantGroup;

// Packed weight tensor header
typedef struct {
    uint32_t rows;
    uint32_t cols;
    QuantType quant;
    uint32_t group_size;     // typically 128
    uint32_t n_groups;       // ceil(rows * cols / group_size)
    size_t data_offset;      // offset to packed nibble data
    size_t groups_offset;    // offset to QuantGroup array
    size_t total_bytes;      // total packed size
} PackedWeightHeader;

// ============================================================
// Q4 Packing: float32 -> packed nibbles + scales
// ============================================================

static PackedWeightHeader q4_pack(const float *src, int rows, int cols,
                                   int group_size, uint8_t **out_data,
                                   QuantGroup **out_groups) {
    int total = rows * cols;
    int n_groups = (total + group_size - 1) / group_size;
    int packed_bytes = (total + 1) / 2;  // 2 nibbles per byte

    *out_groups = (QuantGroup *)malloc(n_groups * sizeof(QuantGroup));
    *out_data = (uint8_t *)calloc(packed_bytes, 1);

    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = start + group_size;
        if (end > total) end = total;

        // Find min/max for this group
        float mn = src[start], mx = src[start];
        for (int i = start + 1; i < end; i++) {
            if (src[i] < mn) mn = src[i];
            if (src[i] > mx) mx = src[i];
        }

        // Compute scale and zero for 4-bit range [0, 15]
        float range = mx - mn;
        float scale = range / 15.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        float zero = mn;

        (*out_groups)[g].scale = (_Float16)scale;
        (*out_groups)[g].zero = (_Float16)zero;

        // Quantize and pack
        for (int i = start; i < end; i++) {
            int q = (int)((src[i] - zero) / scale + 0.5f);
            if (q < 0) q = 0;
            if (q > 15) q = 15;

            int byte_idx = i / 2;
            if (i % 2 == 0) {
                (*out_data)[byte_idx] |= (uint8_t)(q & 0x0F);
            } else {
                (*out_data)[byte_idx] |= (uint8_t)((q & 0x0F) << 4);
            }
        }
    }

    PackedWeightHeader hdr = {
        .rows = rows, .cols = cols,
        .quant = QUANT_Q4, .group_size = group_size,
        .n_groups = n_groups,
        .data_offset = 0,
        .groups_offset = packed_bytes,
        .total_bytes = packed_bytes + n_groups * sizeof(QuantGroup)
    };
    return hdr;
}

// ============================================================
// Q4 Dequant: packed nibbles + scales -> fp16 (NEON optimized)
// ============================================================

#ifdef __ARM_NEON
// NEON-optimized Q4 dequant for one group (128 elements)
// Processes 32 elements (16 bytes of packed data) per iteration
static inline void dequant_q4_group_neon(const uint8_t *packed, _Float16 scale,
                                          _Float16 zero, _Float16 *dst, int count) {
    float32x4_t v_scale = vdupq_n_f32((float)scale);
    float32x4_t v_zero = vdupq_n_f32((float)zero);

    int i = 0;
    for (; i + 7 < count; i += 8) {
        // Load 4 bytes = 8 nibbles
        uint8_t b0 = packed[i/2], b1 = packed[i/2 + 1];
        uint8_t b2 = packed[i/2 + 2], b3 = packed[i/2 + 3];

        // Extract nibbles
        float vals[8] = {
            (float)(b0 & 0x0F), (float)((b0 >> 4) & 0x0F),
            (float)(b1 & 0x0F), (float)((b1 >> 4) & 0x0F),
            (float)(b2 & 0x0F), (float)((b2 >> 4) & 0x0F),
            (float)(b3 & 0x0F), (float)((b3 >> 4) & 0x0F),
        };

        // Dequant: val * scale + zero
        float32x4_t v0 = vld1q_f32(vals);
        float32x4_t v1 = vld1q_f32(vals + 4);
        v0 = vmlaq_f32(v_zero, v0, v_scale);  // v0 * scale + zero
        v1 = vmlaq_f32(v_zero, v1, v_scale);

        // Convert to fp16 and store
        float16x4_t h0 = vcvt_f16_f32(v0);
        float16x4_t h1 = vcvt_f16_f32(v1);
        vst1_f16((__fp16*)(dst + i), h0);
        vst1_f16((__fp16*)(dst + i + 4), h1);
    }

    // Scalar tail
    for (; i < count; i++) {
        int byte_idx = i / 2;
        int nibble = (i % 2 == 0)
            ? (packed[byte_idx] & 0x0F)
            : ((packed[byte_idx] >> 4) & 0x0F);
        dst[i] = (_Float16)((float)nibble * (float)scale + (float)zero);
    }
}
#endif

// Dequant entire packed weight tensor to fp16 buffer
// Returns allocated fp16 buffer (caller must free)
static _Float16 *q4_dequant_full(const uint8_t *packed_data,
                                  const QuantGroup *groups,
                                  int rows, int cols, int group_size) {
    int total = rows * cols;
    _Float16 *out = (_Float16 *)malloc(total * sizeof(_Float16));

#ifdef __ARM_NEON
    for (int g = 0; g < (total + group_size - 1) / group_size; g++) {
        int start = g * group_size;
        int count = group_size;
        if (start + count > total) count = total - start;

        dequant_q4_group_neon(
            packed_data + start / 2,
            groups[g].scale, groups[g].zero,
            out + start, count
        );
    }
#else
    // Scalar fallback
    for (int i = 0; i < total; i++) {
        int g = i / group_size;
        int byte_idx = i / 2;
        int nibble = (i % 2 == 0)
            ? (packed_data[byte_idx] & 0x0F)
            : ((packed_data[byte_idx] >> 4) & 0x0F);
        out[i] = (_Float16)((float)nibble * (float)groups[g].scale
                            + (float)groups[g].zero);
    }
#endif
    return out;
}

// ============================================================
// ANE blob builder from dequanted fp16 weights
// Wraps dequanted data in the standard 128-byte header format
// ============================================================

static uint8_t *q4_dequant_to_ane_blob(const uint8_t *packed_data,
                                         const QuantGroup *groups,
                                         int rows, int cols, int group_size,
                                         size_t *out_len) {
    _Float16 *fp16 = q4_dequant_full(packed_data, groups, rows, cols, group_size);
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // ANE blob header
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t *)(buf + 72) = wsize;
    *(uint32_t *)(buf + 80) = 128;

    memcpy(buf + 128, fp16, wsize);
    free(fp16);

    *out_len = total;
    return buf;
}

// ============================================================
// .anepak file format — serialized quantized model
// ============================================================

#define ANEPAK_MAGIC 0x504B454E  // "ANEP" little-endian

typedef struct {
    uint32_t magic;         // ANEPAK_MAGIC
    uint32_t version;       // 1
    uint32_t n_layers;
    uint32_t dim;
    uint32_t hidden_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t vocab_size;
    uint32_t max_seq;
    QuantType quant_type;
    uint32_t group_size;
    uint64_t embed_offset;     // offset to embedding table
    uint64_t embed_size;
    uint64_t rms_final_offset;
    uint64_t layer_offsets[128]; // offset to each layer's packed data (max 128 layers)
    uint64_t layer_sizes[128];
} AnepakHeader;


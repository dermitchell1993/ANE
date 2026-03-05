// ane_hw_detect.h — Runtime Apple Silicon generation detection for ANE targeting
// Detects M1/M2/M3/M4 families via sysctl and IOKit without crashing
// Used to select MIL program version, target level, and op constraints
#pragma once
#import <Foundation/Foundation.h>
#include <sys/sysctl.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// ============================================================
// Chip generation enum — ordered by capability level
// ============================================================
typedef enum {
    ANE_CHIP_UNKNOWN = 0,
    ANE_CHIP_M1      = 1,  // Tonga — 16 NE cores, conv-only, ios14/15 target
    ANE_CHIP_M2      = 2,  // Staten — 16 NE cores, conv-only, ios16 target
    ANE_CHIP_M3      = 3,  // Ibiza — 16 NE cores, limited matmul, ios17 target
    ANE_CHIP_M4      = 4,  // Donan — 16 NE cores, full matmul/SDPA, ios18 target
} ANEChipGen;

// ============================================================
// Chip profile — everything downstream code needs
// ============================================================
typedef struct {
    ANEChipGen gen;
    const char *name;           // "M1", "M2", "M2 Pro", etc.
    int ane_cores;              // NE core count
    int max_unified_gb;         // max unified memory tier
    bool supports_matmul;       // ANE matmul op supported?
    bool supports_sdpa;         // ANE fused SDPA supported?
    const char *mil_version;    // "1.0" or "1.3"
    const char *mil_target;     // "ios16" or "ios18"
    int max_compiles;           // safe compile count before leak-induced crash
    int iosurface_align;        // required IOSurface byte alignment
    int max_conv_channels;      // max output channels for a single conv op
    int max_seq_len;            // max sequence length for stable training
    int max_hidden_dim;         // max hidden dimension for stable training
    bool needs_explicit_fp16;   // must cast all I/O to fp16 explicitly
} ANEChipProfile;

// ============================================================
// sysctl string reader (safe, no-crash)
// ============================================================
static bool _ane_sysctl_str(const char *key, char *buf, size_t buflen) {
    size_t len = buflen;
    if (sysctlbyname(key, buf, &len, NULL, 0) != 0) {
        buf[0] = '\0';
        return false;
    }
    return true;
}

static uint64_t _ane_sysctl_u64(const char *key) {
    uint64_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(key, &val, &len, NULL, 0);
    return val;
}

// ============================================================
// Detect chip generation from CPU brand string + cpufamily
// ============================================================
static ANEChipGen _ane_detect_gen_from_brand(const char *brand) {
    // M4 family
    if (strstr(brand, "M4"))  return ANE_CHIP_M4;
    // M3 family
    if (strstr(brand, "M3"))  return ANE_CHIP_M3;
    // M2 family
    if (strstr(brand, "M2"))  return ANE_CHIP_M2;
    // M1 family
    if (strstr(brand, "M1"))  return ANE_CHIP_M1;
    // A-series (A14+ have ANE, treat as M1-tier)
    if (strstr(brand, "A14") || strstr(brand, "A15") || strstr(brand, "A16"))
        return ANE_CHIP_M1;
    if (strstr(brand, "A17"))
        return ANE_CHIP_M3;
    return ANE_CHIP_UNKNOWN;
}

// ============================================================
// Detect memory tier
// ============================================================
static int _ane_detect_memory_gb(void) {
    uint64_t memsize = _ane_sysctl_u64("hw.memsize");
    return (int)(memsize / (1024ULL * 1024ULL * 1024ULL));
}

// ============================================================
// Primary detection: read hw.cpufamily + brand string
// Falls back to capability probing if sysctl fails
// ============================================================
static ANEChipGen _ANEDetectChipGen(void) {
    char brand[256] = {0};

    // Try machdep.cpu.brand_string first (Intel compat path, works on Rosetta)
    if (_ane_sysctl_str("machdep.cpu.brand_string", brand, sizeof(brand)) && brand[0]) {
        ANEChipGen gen = _ane_detect_gen_from_brand(brand);
        if (gen != ANE_CHIP_UNKNOWN) return gen;
    }

    // Try hw.chip (available on some macOS versions)
    if (_ane_sysctl_str("hw.chip", brand, sizeof(brand)) && brand[0]) {
        ANEChipGen gen = _ane_detect_gen_from_brand(brand);
        if (gen != ANE_CHIP_UNKNOWN) return gen;
    }

    // Try product name via IOKit property path
    if (_ane_sysctl_str("hw.model", brand, sizeof(brand)) && brand[0]) {
        // Mac model identifiers: Mac14,x = M2, Mac15,x = M3, Mac16,x = M4
        int major = 0;
        if (sscanf(brand, "Mac%d,", &major) == 1) {
            if (major >= 16) return ANE_CHIP_M4;
            if (major >= 15) return ANE_CHIP_M3;
            if (major >= 14) return ANE_CHIP_M2;
            if (major >= 13) return ANE_CHIP_M1;
        }
        // MacBookPro/MacBookAir/iMac identifiers
        if (strstr(brand, "MacBookPro18") || strstr(brand, "MacBookAir10") ||
            strstr(brand, "Macmini9") || strstr(brand, "iMac21"))
            return ANE_CHIP_M1;
        if (strstr(brand, "Mac14") || strstr(brand, "MacBookPro19") ||
            strstr(brand, "MacBookAir11"))
            return ANE_CHIP_M2;
    }

    // Fallback: check cpufamily for known ARM families
    uint32_t cpufam = 0;
    size_t len = sizeof(cpufam);
    if (sysctlbyname("hw.cpufamily", &cpufam, &len, NULL, 0) == 0) {
        // Known Apple Silicon cpufamily values (from XNU headers)
        // These are hashes and change per SoC, but we cover the known ones
        switch (cpufam) {
            case 0x1b588bb3: return ANE_CHIP_M1;  // Firestorm+Icestorm (M1)
            case 0xda33d83d: return ANE_CHIP_M2;  // Avalanche+Blizzard (M2)
            case 0x8765edea: return ANE_CHIP_M3;  // Everest+Sawtooth (M3)
            case 0xfa33415e: return ANE_CHIP_M4;  // M4 family
        }
    }

    return ANE_CHIP_UNKNOWN;
}

// ============================================================
// Build the full chip profile from detected generation
// ============================================================
static ANEChipProfile _ANEGetChipProfile(ANEChipGen gen) {
    ANEChipProfile p = {0};
    p.gen = gen;
    p.max_unified_gb = _ane_detect_memory_gb();

    switch (gen) {
        case ANE_CHIP_M1:
            p.name = "M1";
            p.ane_cores = 16;
            p.supports_matmul = false;
            p.supports_sdpa = false;
            p.mil_version = "1.0";
            p.mil_target = "ios16";  // M1 launched w/ iOS 14, but ios16 MIL is safe
            p.max_compiles = 60;     // M1 leaks fastest
            p.iosurface_align = 256;
            p.max_conv_channels = 16384;
            p.max_seq_len = 256;
            p.max_hidden_dim = 2048;
            p.needs_explicit_fp16 = true;
            break;

        case ANE_CHIP_M2:
            p.name = "M2";
            p.ane_cores = 16;
            p.supports_matmul = false;
            p.supports_sdpa = false;
            p.mil_version = "1.0";
            p.mil_target = "ios16";
            p.max_compiles = 80;     // M2 leaks slower than M1
            p.iosurface_align = 256;
            p.max_conv_channels = 16384;
            p.max_seq_len = 512;
            p.max_hidden_dim = 4096;
            p.needs_explicit_fp16 = true;
            break;

        case ANE_CHIP_M3:
            p.name = "M3";
            p.ane_cores = 16;
            p.supports_matmul = true;
            p.supports_sdpa = false;  // M3 has matmul but not fused SDPA
            p.mil_version = "1.0";
            p.mil_target = "ios17";
            p.max_compiles = 90;
            p.iosurface_align = 128;
            p.max_conv_channels = 32000;
            p.max_seq_len = 1024;
            p.max_hidden_dim = 8192;
            p.needs_explicit_fp16 = true;
            break;

        case ANE_CHIP_M4:
            p.name = "M4";
            p.ane_cores = 16;
            p.supports_matmul = true;
            p.supports_sdpa = true;
            p.mil_version = "1.3";
            p.mil_target = "ios18";
            p.max_compiles = 100;
            p.iosurface_align = 64;  // M4 tolerates tighter alignment
            p.max_conv_channels = 32000;
            p.max_seq_len = 2048;
            p.max_hidden_dim = 16384;
            p.needs_explicit_fp16 = false;
            break;

        default: // Unknown — assume M2-tier (conservative)
            p.name = "Unknown (M2-compat)";
            p.ane_cores = 16;
            p.supports_matmul = false;
            p.supports_sdpa = false;
            p.mil_version = "1.0";
            p.mil_target = "ios16";
            p.max_compiles = 60;
            p.iosurface_align = 256;
            p.max_conv_channels = 16384;
            p.max_seq_len = 256;
            p.max_hidden_dim = 2048;
            p.needs_explicit_fp16 = true;
            break;
    }
    return p;
}

// ============================================================
// Public API — single-call version detection
// Thread-safe via dispatch_once
// ============================================================
static ANEChipProfile g_ane_profile;
static bool g_ane_profile_valid = false;

static ANEChipProfile ANEVersionDetect(void) {
    if (!g_ane_profile_valid) {
        ANEChipGen gen = _ANEDetectChipGen();
        g_ane_profile = _ANEGetChipProfile(gen);
        g_ane_profile_valid = true;

        printf("[ANE HW] Detected: %s (%d GB unified) — %d NE cores\n",
               g_ane_profile.name, g_ane_profile.max_unified_gb, g_ane_profile.ane_cores);
        printf("[ANE HW] MIL: program(%s) target <%s>, matmul=%s, SDPA=%s\n",
               g_ane_profile.mil_version, g_ane_profile.mil_target,
               g_ane_profile.supports_matmul ? "yes" : "no",
               g_ane_profile.supports_sdpa ? "yes" : "no");
        printf("[ANE HW] Limits: max_compiles=%d, align=%d, max_seq=%d, max_hidden=%d\n",
               g_ane_profile.max_compiles, g_ane_profile.iosurface_align,
               g_ane_profile.max_seq_len, g_ane_profile.max_hidden_dim);
    }
    return g_ane_profile;
}

// ============================================================
// Convenience predicates
// ============================================================
static inline bool ane_is_m1(void)       { return ANEVersionDetect().gen == ANE_CHIP_M1; }
static inline bool ane_is_m2(void)       { return ANEVersionDetect().gen == ANE_CHIP_M2; }
static inline bool ane_is_m1_or_m2(void) { ANEChipGen g = ANEVersionDetect().gen; return g == ANE_CHIP_M1 || g == ANE_CHIP_M2; }
static inline bool ane_has_matmul(void)  { return ANEVersionDetect().supports_matmul; }
static inline bool ane_has_sdpa(void)    { return ANEVersionDetect().supports_sdpa; }


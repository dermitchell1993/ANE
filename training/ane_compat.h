// ane_compat.h — Runtime platform detection for Apple Silicon ANE compatibility
// Detects chip family, macOS version, ANE peak TFLOPS, and appropriate MIL target
#pragma once
#import <Foundation/Foundation.h>
#include <sys/sysctl.h>
#include <string.h>
#include <stdio.h>

// Chip family enumeration
typedef enum {
    ANE_CHIP_UNKNOWN = 0,
    ANE_CHIP_M1,
    ANE_CHIP_M1_PRO,
    ANE_CHIP_M1_MAX,
    ANE_CHIP_M1_ULTRA,
    ANE_CHIP_M2,
    ANE_CHIP_M2_PRO,
    ANE_CHIP_M2_MAX,
    ANE_CHIP_M2_ULTRA,
    ANE_CHIP_M3,
    ANE_CHIP_M3_PRO,
    ANE_CHIP_M3_MAX,
    ANE_CHIP_M3_ULTRA,
    ANE_CHIP_M4,
    ANE_CHIP_M4_PRO,
    ANE_CHIP_M4_MAX,
    ANE_CHIP_M4_ULTRA,
    ANE_CHIP_M5,
    ANE_CHIP_M5_PRO,
    ANE_CHIP_M5_MAX,
    ANE_CHIP_M5_ULTRA,
} ANEChipFamily;

// Platform info resolved at runtime
typedef struct {
    ANEChipFamily chip;
    char chip_name[64];       // e.g. "Apple M4"
    int macos_major;          // e.g. 14, 15
    int macos_minor;          // e.g. 0, 1
    double ane_peak_tflops;   // Estimated FP16 peak TFLOPS
    const char *mil_target;   // "ios16", "ios17", or "ios18"
    const char *mil_program;  // "1.0" for ios16/17, "1.3" for ios18
    bool api_available;       // Whether _ANEInMemoryModel is available
} ANEPlatform;

// Global platform info (set once by ane_detect_platform)
static ANEPlatform g_ane_platform = {0};
static bool g_ane_platform_detected = false;

// ---- Internal helpers ----

static ANEChipFamily _ane_identify_chip(const char *brand) {
    // Match chip family from sysctl brand string (e.g. "Apple M4", "Apple M2 Pro")
    if (strstr(brand, "M5 Ultra"))  return ANE_CHIP_M5_ULTRA;
    if (strstr(brand, "M5 Max"))    return ANE_CHIP_M5_MAX;
    if (strstr(brand, "M5 Pro"))    return ANE_CHIP_M5_PRO;
    if (strstr(brand, "M5"))        return ANE_CHIP_M5;
    if (strstr(brand, "M4 Ultra"))  return ANE_CHIP_M4_ULTRA;
    if (strstr(brand, "M4 Max"))    return ANE_CHIP_M4_MAX;
    if (strstr(brand, "M4 Pro"))    return ANE_CHIP_M4_PRO;
    if (strstr(brand, "M4"))        return ANE_CHIP_M4;
    if (strstr(brand, "M3 Ultra"))  return ANE_CHIP_M3_ULTRA;
    if (strstr(brand, "M3 Max"))    return ANE_CHIP_M3_MAX;
    if (strstr(brand, "M3 Pro"))    return ANE_CHIP_M3_PRO;
    if (strstr(brand, "M3"))        return ANE_CHIP_M3;
    if (strstr(brand, "M2 Ultra"))  return ANE_CHIP_M2_ULTRA;
    if (strstr(brand, "M2 Max"))    return ANE_CHIP_M2_MAX;
    if (strstr(brand, "M2 Pro"))    return ANE_CHIP_M2_PRO;
    if (strstr(brand, "M2"))        return ANE_CHIP_M2;
    if (strstr(brand, "M1 Ultra"))  return ANE_CHIP_M1_ULTRA;
    if (strstr(brand, "M1 Max"))    return ANE_CHIP_M1_MAX;
    if (strstr(brand, "M1 Pro"))    return ANE_CHIP_M1_PRO;
    if (strstr(brand, "M1"))        return ANE_CHIP_M1;
    return ANE_CHIP_UNKNOWN;
}

// Estimated FP16 ANE peak TFLOPS per chip.
// Apple publishes INT8 TOPS; FP16 throughput is roughly half.
// Values are best-effort estimates from known hardware specs.
// Ultra variants double the base die's ANE (2x neural engines).
static double _ane_peak_tflops(ANEChipFamily chip) {
    switch (chip) {
        case ANE_CHIP_M1:       return 5.5;
        case ANE_CHIP_M1_PRO:   return 5.5;
        case ANE_CHIP_M1_MAX:   return 5.5;
        case ANE_CHIP_M1_ULTRA: return 11.0;
        case ANE_CHIP_M2:       return 7.9;   // 15.8 TOPS / 2
        case ANE_CHIP_M2_PRO:   return 7.9;
        case ANE_CHIP_M2_MAX:   return 7.9;
        case ANE_CHIP_M2_ULTRA: return 15.8;
        case ANE_CHIP_M3:       return 9.0;   // 18 TOPS / 2
        case ANE_CHIP_M3_PRO:   return 9.0;
        case ANE_CHIP_M3_MAX:   return 9.0;
        case ANE_CHIP_M3_ULTRA: return 18.0;
        case ANE_CHIP_M4:       return 15.8;  // Empirically measured in this project
        case ANE_CHIP_M4_PRO:   return 15.8;
        case ANE_CHIP_M4_MAX:   return 15.8;
        case ANE_CHIP_M4_ULTRA: return 31.6;
        case ANE_CHIP_M5:       return 19.0;  // 38 TOPS / 2 (estimate)
        case ANE_CHIP_M5_PRO:   return 19.0;
        case ANE_CHIP_M5_MAX:   return 19.0;
        case ANE_CHIP_M5_ULTRA: return 38.0;
        default:                return 15.8;  // Fallback: assume M4-class
    }
}

static const char *_ane_chip_name_str(ANEChipFamily chip) {
    switch (chip) {
        case ANE_CHIP_M1:       return "M1";
        case ANE_CHIP_M1_PRO:   return "M1 Pro";
        case ANE_CHIP_M1_MAX:   return "M1 Max";
        case ANE_CHIP_M1_ULTRA: return "M1 Ultra";
        case ANE_CHIP_M2:       return "M2";
        case ANE_CHIP_M2_PRO:   return "M2 Pro";
        case ANE_CHIP_M2_MAX:   return "M2 Max";
        case ANE_CHIP_M2_ULTRA: return "M2 Ultra";
        case ANE_CHIP_M3:       return "M3";
        case ANE_CHIP_M3_PRO:   return "M3 Pro";
        case ANE_CHIP_M3_MAX:   return "M3 Max";
        case ANE_CHIP_M3_ULTRA: return "M3 Ultra";
        case ANE_CHIP_M4:       return "M4";
        case ANE_CHIP_M4_PRO:   return "M4 Pro";
        case ANE_CHIP_M4_MAX:   return "M4 Max";
        case ANE_CHIP_M4_ULTRA: return "M4 Ultra";
        case ANE_CHIP_M5:       return "M5";
        case ANE_CHIP_M5_PRO:   return "M5 Pro";
        case ANE_CHIP_M5_MAX:   return "M5 Max";
        case ANE_CHIP_M5_ULTRA: return "M5 Ultra";
        default:                return "Unknown";
    }
}

// ---- Public API ----

// Detect the current platform. Call once at startup.
// Returns the populated ANEPlatform struct (also stored in g_ane_platform).
static ANEPlatform ane_detect_platform(void) {
    if (g_ane_platform_detected) return g_ane_platform;

    ANEPlatform p = {0};

    // 1. Detect chip via sysctl
    char brand[128] = {0};
    size_t len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, NULL, 0) != 0) {
        // Fallback: try hw.machine or hw.model
        len = sizeof(brand);
        sysctlbyname("hw.model", brand, &len, NULL, 0);
    }
    strncpy(p.chip_name, brand, sizeof(p.chip_name) - 1);
    p.chip = _ane_identify_chip(brand);

    // 2. Detect macOS version
    NSOperatingSystemVersion ver = [[NSProcessInfo processInfo] operatingSystemVersion];
    p.macos_major = (int)ver.majorVersion;
    p.macos_minor = (int)ver.minorVersion;

    // 3. Set ANE peak TFLOPS
    p.ane_peak_tflops = _ane_peak_tflops(p.chip);

    // 4. Select MIL target based on macOS version
    //    - macOS 15+ (Sequoia)  → ios18 + program(1.3)
    //    - macOS 14  (Sonoma)   → ios17 + program(1.0)
    //    - macOS 13  (Ventura)  → ios16 + program(1.0)
    //    - older                → unsupported
    if (p.macos_major >= 15) {
        p.mil_target = "ios18";
        p.mil_program = "1.3";
    } else if (p.macos_major == 14) {
        p.mil_target = "ios17";
        p.mil_program = "1.0";
    } else if (p.macos_major == 13) {
        p.mil_target = "ios16";
        p.mil_program = "1.0";
    } else {
        p.mil_target = "ios16";
        p.mil_program = "1.0";
    }

    // 5. Check API availability
    p.api_available = (NSClassFromString(@"_ANEInMemoryModelDescriptor") != nil &&
                       NSClassFromString(@"_ANEInMemoryModel") != nil);

    g_ane_platform = p;
    g_ane_platform_detected = true;
    return p;
}

// Print detected platform info (call after ane_detect_platform)
static void ane_print_platform(void) {
    if (!g_ane_platform_detected) ane_detect_platform();
    const ANEPlatform *p = &g_ane_platform;
    printf("=== ANE Platform ===\n");
    printf("  Chip:       %s (%s)\n", _ane_chip_name_str(p->chip), p->chip_name);
    printf("  macOS:      %d.%d\n", p->macos_major, p->macos_minor);
    printf("  ANE peak:   %.1f TFLOPS (FP16 est.)\n", p->ane_peak_tflops);
    printf("  MIL target: %s (program %s)\n", p->mil_target, p->mil_program);
    printf("  API ready:  %s\n", p->api_available ? "YES" : "NO");
    printf("====================\n");
}

// Generate the MIL header string with correct program version and build info.
// Returns an autoreleased NSString.
static NSString *ane_mil_header(void) {
    if (!g_ane_platform_detected) ane_detect_platform();
    return [NSString stringWithFormat:
        @"program(%s)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"\"}, "
        "{\"coremlc-version\", \"\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"\"}})]\n{\n",
        g_ane_platform.mil_program];
}

// Get the MIL function target annotation (e.g. "ios17" or "ios18")
static const char *ane_mil_target(void) {
    if (!g_ane_platform_detected) ane_detect_platform();
    return g_ane_platform.mil_target;
}

// Get the ANE peak TFLOPS for utilization calculations
static double ane_peak_tflops(void) {
    if (!g_ane_platform_detected) ane_detect_platform();
    return g_ane_platform.ane_peak_tflops;
}

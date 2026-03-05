// ane_compat.h — M1/M2 backward-compatible MIL generators
// Conv-only paths for pre-M4 ANE hardware (no matmul, no SDPA)
// Uses program(1.0) with ios16 target and verbose tensor<fp16,...> syntax
//
// Architecture: Each existing MIL generator in ane_mil_gen.h, stories_mil.h,
// and ane_classifier.h has a parallel _m2() variant here that produces
// equivalent computation using only conv1d operations.
//
// The calling code checks ane_has_matmul() / chip profile and dispatches
// to the appropriate generator.
#pragma once
#import <Foundation/Foundation.h>
#include "ane_hw_detect.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// MIL header for M1/M2: program(1.0), ios16 target
// ============================================================
#define MIL_HDR_M2 \
    @"program(1.0)\n" \
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3010.1.1\"}, " \
    "{\"coremlc-version\", \"3005.2.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"7.0\"}})]\n{\n"

#define CONV_CONST_M2 \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// ============================================================
// M2-safe IOSurface creation with 256-byte alignment
// ============================================================
static IOSurfaceRef make_surface_m2(size_t bytes) {
    // Round up to 256-byte alignment for M1/M2 ANE DMA constraints
    size_t aligned = (bytes + 255) & ~((size_t)255);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0
    });
}

// ============================================================
// Chip-aware IOSurface factory
// ============================================================
static IOSurfaceRef make_surface_compat(size_t bytes) {
    ANEChipProfile p = ANEVersionDetect();
    if (p.iosurface_align >= 256) {
        return make_surface_m2(bytes);
    }
    // M4 path — original alignment
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

// ============================================================
// M2-compatible conv MIL: single conv with baked weights
// Input:  tensor<fp16, [1, in_ch, 1, S]>
// Weight: tensor<fp16, [out_ch, in_ch, 1, 1]> baked
// Output: tensor<fp16, [1, out_ch, 1, S]>
//
// This is the workhorse — every linear layer becomes a 1x1 conv.
// Explicit fp16 I/O throughout (M2 ANE doesn't auto-cast fp32).
// ============================================================
static NSString *mil_gen_conv_m2(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3010.1.1\"}, "
        "{\"coremlc-version\", \"3005.2.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"7.0\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x)[name = string(\"conv\")];\n"
        "    } -> (out);\n"
        "}\n",
        in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, spatial];
}

// ============================================================
// M2-compatible fused QKV: 3 parallel convs from same input
// All fp16 I/O, explicit tensor types, conv-only
// Input:  tensor<fp16, [1, dim, 1, S]>
// Output: Q, K, V each tensor<fp16, [1, dim, 1, S]>
// ============================================================
static NSString *mil_gen_qkv_m2(int dim, int spatial) {
    NSUInteger cs = 64 + (NSUInteger)dim * dim * 2;
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3010.1.1\"}, "
        "{\"coremlc-version\", \"3005.2.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"7.0\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wq = const()[name = string(\"Wq\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = string(\"Wk\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = string(\"Wv\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> q = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x)[name = string(\"conv_q\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> k = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x)[name = string(\"conv_k\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> v = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x)[name = string(\"conv_v\")];\n"
        "    } -> (q, k, v);\n"
        "}\n",
        dim, spatial,
        dim, dim, dim, dim,
        dim, dim, dim, dim, (unsigned long)(64 + cs),
        dim, dim, dim, dim, (unsigned long)(64 + 2*cs),
        dim, spatial, dim, spatial, dim, spatial];
}

// ============================================================
// M2-compatible FFN up: w1 + w3 parallel convs (no matmul)
// Input:  tensor<fp16, [1, dim, 1, S]>
// Output: h1, h3 each tensor<fp16, [1, hidden_dim, 1, S]>
// ============================================================
static NSString *mil_gen_ffn_up_m2(int dim, int hidden_dim, int spatial) {
    NSUInteger cs = 64 + (NSUInteger)hidden_dim * dim * 2;
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3010.1.1\"}, "
        "{\"coremlc-version\", \"3005.2.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"7.0\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W1 = const()[name = string(\"W1\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W3 = const()[name = string(\"W3\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> h1 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x)[name = string(\"conv_w1\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> h3 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x)[name = string(\"conv_w3\")];\n"
        "    } -> (h1, h3);\n"
        "}\n",
        dim, spatial,
        hidden_dim, dim, hidden_dim, dim,
        hidden_dim, dim, hidden_dim, dim, (unsigned long)(64 + cs),
        hidden_dim, spatial, hidden_dim, spatial];
}

// ============================================================
// M2-compatible SDPA forward: conv-only attention
// Since M2 ANE has no matmul, attention Q*K^T and attn*V are computed
// on CPU. The ANE handles only the linear projections (QKV + Wo).
//
// This is the "SDPA forward with taps" for the large pipeline.
// Input:  x [1, DIM, 1, SEQ] — fp16
// Baked:  Wq, Wk, Wv, Wo, rms1 weights
// Output: concat(o_out, Q, K, V, attn_out, xnorm) — [1, 6*DIM, 1, SEQ] fp16
//
// On M2, we split this into conv-only projections on ANE,
// then do the attention matmuls on CPU.
// ============================================================
static NSString *gen_sdpa_fwd_taps_m2(int dim, int heads, int hd, int seq) {
    float invd = 1.0f/(float)dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR_M2];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    // RMSNorm inline
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", dim, seq];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", seq];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", seq];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", dim, seq];
    // Conv projections only (no matmul for Q*K^T — that stays on CPU)
    [m appendString:@CONV_CONST_M2];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    // QKV projections via conv
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];\n", dim,seq];
    // Output Q, K, V, xnorm — attention will be done on CPU, then Wo conv applied separately
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(qf,kf,vf,xn))[name=string(\"cat\")];\n", 4*dim,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ============================================================
// M2-compatible FFN forward with taps
// Conv-only: rmsnorm + W1/W3 parallel convs
// Attention output comes in from CPU, SiLU/element-wise on CPU,
// then W2 conv applied as a separate kernel.
// ============================================================
static NSString *gen_ffn_fwd_taps_m2(int dim, int hidden, int seq) {
    float invd = 1.0f/(float)dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR_M2];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    // RMSNorm
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", dim, seq];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", seq];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", seq];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", dim, seq];
    // Conv projections
    [m appendString:@CONV_CONST_M2];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", hidden,dim,hidden,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", hidden,dim,hidden,dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", hidden,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", hidden,seq];
    // Concat h1, h3, xnorm for CPU SiLU + downstream
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(h1,h3,xn))[name=string(\"cat\")];\n", 2*hidden+dim,seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ============================================================
// M2-compatible RMSNorm backward (conv-only — same as M4, no matmul used)
// This is identical to gen_rmsnorm_bwd() but uses ios16 target
// ============================================================
static NSString *gen_rmsnorm_bwd_m2(int dim, int seq) {
    float invd = 1.0f / (float)dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR_M2];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> inp) {\n", 2*dim, seq];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dy = slice_by_size(x=inp,begin=b0,size=sz)[name=string(\"sdy\")];\n", dim, seq];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x = slice_by_size(x=inp,begin=b1,size=sz)[name=string(\"sx\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", dim, seq];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", seq];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", seq];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> w = const()[name=string(\"w\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_w.bin\"), offset=uint64(64)))];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dyw = mul(x=dy,y=w)[name=string(\"dyw\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dywx = mul(x=dyw,y=x)[name=string(\"dywx\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> dot_sum = reduce_sum(x=dywx,axes=rax,keep_dims=kd)[name=string(\"ds\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> dot_sc = mul(x=dot_sum,y=invd)[name=string(\"dsc\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms2 = mul(x=rrms,y=rrms)[name=string(\"rr2\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> coeff = mul(x=dot_sc,y=rrms2)[name=string(\"cof\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xc = mul(x=x,y=coeff)[name=string(\"xc\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> diff = sub(x=dyw,y=xc)[name=string(\"dif\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = mul(x=diff,y=rrms)[name=string(\"out\")];\n", dim, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ============================================================
// M2-compatible classifier forward: conv-only
// Uses conv instead of matmul for embed @ x_final
// Input:  tensor<fp16, [1, DIM, 1, SEQ]>
// Weight: tensor<fp16, [VOCAB, DIM, 1, 1]> baked
// Output: tensor<fp16, [1, VOCAB, 1, SEQ]>
// ============================================================
static NSString *gen_classifier_fwd_m2(int dim, int vocab, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR_M2];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendString:@CONV_CONST_M2];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> We = const()[name=string(\"We\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/embed.bin\"), offset=uint64(64)))];\n",
        vocab, dim, vocab, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=We,x=x)[name=string(\"cls\")];\n", vocab, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ============================================================
// M2-compatible classifier backward: conv-only (replaces matmul)
// On M4, this uses matmul for dx = embed^T @ dlogits.
// On M2, we use a conv with transposed weights: [DIM, VOCAB, 1, 1]
// This requires pre-transposing embed weights at weight-load time.
//
// Input:  dlogits [1, VOCAB, 1, SEQ] fp16
// Weight: embed_t [DIM, VOCAB, 1, 1] baked (transposed embed)
// Output: dx [1, DIM, 1, SEQ] fp16
// ============================================================
static NSString *gen_classifier_bwd_m2(int dim, int vocab, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR_M2];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> dl) {\n", vocab, seq];
    [m appendString:@CONV_CONST_M2];
    // Transposed embed as conv weight: [DIM, VOCAB, 1, 1]
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wet = const()[name=string(\"Wet\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/embed_t.bin\"), offset=uint64(64)))];\n",
        dim, vocab, dim, vocab];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wet,x=dl)[name=string(\"cls_bwd\")];\n", dim, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ============================================================
// M2-compatible final RMSNorm (ios16 target)
// Same math as M4 version, just different program header
// ============================================================
static NSString *gen_final_rmsnorm_m2(int dim, int seq) {
    float invd = 1.0f/(float)dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR_M2];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", dim, seq];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", seq];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", seq];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_w.bin\"), offset=uint64(64)))];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = mul(x=xr,y=rw)[name=string(\"out\")];\n", dim, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ============================================================
// M2-compatible softmax over VOCAB (same math, ios16 target)
// ============================================================
static NSString *gen_softmax_vocab_m2(int vocab, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR_M2];
    [m appendFormat:@"    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x) {\n", vocab, seq];
    [m appendString:@"        int32 ax = const()[name=string(\"ax\"), val=int32(1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = softmax(axis=ax,x=x)[name=string(\"sm\")];\n", vocab, seq];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ============================================================
// Chip-aware dispatch helpers
// Select M2 or M4 variant based on detected hardware
// ============================================================
static NSString *mil_gen_conv_compat(int in_ch, int out_ch, int spatial) {
    if (ane_is_m1_or_m2()) return mil_gen_conv_m2(in_ch, out_ch, spatial);
    // M4 path — use existing mil_gen_conv from ane_mil_gen.h
    return nil; // caller should use mil_gen_conv() for M4
}

static NSString *mil_gen_qkv_compat(int dim, int spatial) {
    if (ane_is_m1_or_m2()) return mil_gen_qkv_m2(dim, spatial);
    return nil; // caller should use mil_gen_qkv() for M4
}

static NSString *mil_gen_ffn_up_compat(int dim, int hidden_dim, int spatial) {
    if (ane_is_m1_or_m2()) return mil_gen_ffn_up_m2(dim, hidden_dim, spatial);
    return nil; // caller should use mil_gen_ffn_up() for M4
}


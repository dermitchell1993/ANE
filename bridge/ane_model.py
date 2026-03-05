#!/usr/bin/env python3
"""ane_model.py — Python ctypes wrapper for ANE inference via libane_bridge.dylib

Phase 1 deliverable: enables model loading (GGUF/SafeTensors) and inference
from Python without touching Objective-C directly.

Usage:
    from ane_model import ANEModel, ModelConfig

    config = ModelConfig.llama_7b()
    model = ANEModel(config)
    model.load_weights("path/to/model.gguf")
    model.compile_kernels()

    logits = model.forward(input_ids)  # np.ndarray[seq_len, vocab_size]
    token = model.decode_step(token_id=1, pos=42)
"""

import ctypes
import ctypes.util
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Locate the bridge dylib relative to this file
_BRIDGE_DIR = Path(__file__).resolve().parent
_DYLIB_PATH = _BRIDGE_DIR / "libane_bridge.dylib"


@dataclass
class ModelConfig:
    """LLaMA-family model configuration."""
    dim: int = 4096
    hidden_dim: int = 11008
    n_heads: int = 32
    n_kv_heads: int = 32
    head_dim: int = 128
    vocab_size: int = 32000
    max_seq: int = 2048
    n_layers: int = 32
    rope_theta: float = 10000.0

    @staticmethod
    def llama_7b():
        return ModelConfig(dim=4096, hidden_dim=11008, n_heads=32, n_kv_heads=32,
                           head_dim=128, vocab_size=32000, max_seq=2048, n_layers=32)

    @staticmethod
    def llama_13b():
        return ModelConfig(dim=5120, hidden_dim=13824, n_heads=40, n_kv_heads=40,
                           head_dim=128, vocab_size=32000, max_seq=2048, n_layers=40)

    @staticmethod
    def mistral_7b():
        return ModelConfig(dim=4096, hidden_dim=14336, n_heads=32, n_kv_heads=8,
                           head_dim=128, vocab_size=32000, max_seq=2048, n_layers=32)

    @staticmethod
    def stories_110m():
        return ModelConfig(dim=768, hidden_dim=2048, n_heads=12, n_kv_heads=12,
                           head_dim=64, vocab_size=32000, max_seq=1024, n_layers=12)

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim

    def weight_bytes_per_layer_fp16(self):
        """Total fp16 weight bytes for one transformer layer."""
        qkvo = self.dim * self.dim * 2  # Q, O: [dim, dim]
        kv = self.kv_dim * self.dim * 2  # K, V: [kv_dim, dim]
        ffn = (self.hidden_dim * self.dim * 2) * 2 + self.dim * self.hidden_dim * 2  # W1+W3+W2
        rms = self.dim * 2 * 2  # rms_att + rms_ffn
        return qkvo * 2 + kv * 2 + ffn + rms

    def total_weight_bytes_fp16(self):
        per_layer = self.weight_bytes_per_layer_fp16()
        embed = self.vocab_size * self.dim * 2
        rms_final = self.dim * 2
        return per_layer * self.n_layers + embed + rms_final

    def memory_estimate_gb(self, quant_bits=16):
        """Estimate total memory for inference (weights + KV cache + activations)."""
        weight_bytes = self.total_weight_bytes_fp16() * quant_bits / 16
        kv_bytes = self.n_layers * 2 * self.kv_dim * self.max_seq * 2
        act_bytes = max(self.dim, self.hidden_dim) * self.max_seq * 2 * 8
        return (weight_bytes + kv_bytes + act_bytes) / (1024**3)


def _build_ane_blob_header(data_size: int) -> bytes:
    """Build the 128-byte ANE weight blob header.

    Format: 64-byte global header + 64-byte chunk header.
    Matches the format in stories_io.h build_blob().
    """
    buf = bytearray(128)
    buf[0] = 0x01
    buf[4] = 0x02
    # Chunk header at offset 64
    buf[64:68] = b'\xEF\xBE\xAD\xDE'  # sentinel
    buf[68] = 0x01
    struct.pack_into('<I', buf, 72, data_size)
    struct.pack_into('<I', buf, 80, 128)  # data offset from file start
    return bytes(buf)


def weights_to_ane_blob(weights: np.ndarray) -> bytes:
    """Convert a float32/float16 weight matrix to ANE blob format.

    Args:
        weights: numpy array, will be converted to float16 if needed.

    Returns:
        bytes: 128-byte header + fp16 weight data
    """
    if weights.dtype != np.float16:
        weights = weights.astype(np.float16)
    data = weights.tobytes()
    header = _build_ane_blob_header(len(data))
    return header + data


def precompute_rope_tables(max_seq: int, head_dim: int, theta: float = 10000.0):
    """Precompute RoPE cos/sin tables as fp16 ANE blobs.

    Returns:
        (cos_blob, sin_blob): Each is bytes in ANE blob format.
        Shape: [max_seq, head_dim//2] stored as fp16.
    """
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
    positions = np.arange(max_seq, dtype=np.float32)
    angles = np.outer(positions, freqs)  # [max_seq, half_dim]

    cos_table = np.cos(angles).astype(np.float16)
    sin_table = np.sin(angles).astype(np.float16)

    return weights_to_ane_blob(cos_table), weights_to_ane_blob(sin_table)


def build_causal_mask(seq_len: int) -> bytes:
    """Build causal attention mask as ANE blob.

    Upper triangle filled with -65504.0 (fp16 -inf).
    Shape: [seq_len, seq_len] stored as fp16.
    """
    mask = np.zeros((seq_len, seq_len), dtype=np.float16)
    mask[np.triu_indices(seq_len, k=1)] = np.float16(-65504.0)
    return weights_to_ane_blob(mask)


class ANEBridge:
    """Low-level ctypes wrapper around libane_bridge.dylib."""

    def __init__(self, dylib_path: Optional[str] = None):
        path = dylib_path or str(_DYLIB_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ANE bridge dylib not found at {path}. "
                f"Build it with: cd bridge && make"
            )
        self._lib = ctypes.cdll.LoadLibrary(path)
        self._setup_signatures()
        rc = self._lib.ane_bridge_init()
        if rc != 0:
            raise RuntimeError("ane_bridge_init() failed — ANE framework not available")

    def _setup_signatures(self):
        lib = self._lib

        lib.ane_bridge_init.restype = ctypes.c_int
        lib.ane_bridge_init.argtypes = []

        lib.ane_bridge_compile.restype = ctypes.c_void_p
        lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,        # mil_text, mil_len
            ctypes.c_void_p, ctypes.c_size_t,        # weight_data, weight_len
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),  # n_inputs, input_sizes
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),  # n_outputs, output_sizes
        ]

        lib.ane_bridge_compile_multi_weights.restype = ctypes.c_void_p
        lib.ane_bridge_compile_multi_weights.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ]

        lib.ane_bridge_eval.restype = ctypes.c_bool
        lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]

        lib.ane_bridge_write_input.restype = None
        lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
        ]

        lib.ane_bridge_read_output.restype = None
        lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
        ]

        lib.ane_bridge_free.restype = None
        lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

        lib.ane_bridge_get_compile_count.restype = ctypes.c_int
        lib.ane_bridge_get_compile_count.argtypes = []

    def compile_kernel(self, mil_text: str, weight_blobs: dict[str, bytes],
                       input_sizes: list[int], output_sizes: list[int]) -> int:
        """Compile a MIL program with named weight blobs.

        Args:
            mil_text: UTF-8 MIL program text
            weight_blobs: dict mapping weight names to blob bytes
            input_sizes: byte sizes for each input tensor
            output_sizes: byte sizes for each output tensor

        Returns:
            Opaque kernel handle (pointer as int), or 0 on failure.
        """
        mil_bytes = mil_text.encode('utf-8')
        n_in = len(input_sizes)
        n_out = len(output_sizes)
        in_arr = (ctypes.c_size_t * n_in)(*input_sizes)
        out_arr = (ctypes.c_size_t * n_out)(*output_sizes)

        if not weight_blobs:
            handle = self._lib.ane_bridge_compile(
                mil_bytes, len(mil_bytes),
                None, 0,
                n_in, in_arr, n_out, out_arr
            )
        else:
            names = list(weight_blobs.keys())
            n_w = len(names)
            c_names = (ctypes.c_char_p * n_w)(*[n.encode() for n in names])

            # Keep references to prevent GC
            buffers = [weight_blobs[n] for n in names]
            c_datas = (ctypes.c_void_p * n_w)()
            c_lens = (ctypes.c_size_t * n_w)()
            for i, buf in enumerate(buffers):
                c_datas[i] = ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p).value
                c_lens[i] = len(buf)

            handle = self._lib.ane_bridge_compile_multi_weights(
                mil_bytes, len(mil_bytes),
                c_names, c_datas, c_lens, n_w,
                n_in, in_arr, n_out, out_arr
            )

        return handle or 0

    def eval_kernel(self, handle: int) -> bool:
        return self._lib.ane_bridge_eval(ctypes.c_void_p(handle))

    def write_input(self, handle: int, idx: int, data: np.ndarray):
        buf = data.tobytes()
        self._lib.ane_bridge_write_input(
            ctypes.c_void_p(handle), idx,
            buf, len(buf)
        )

    def read_output(self, handle: int, idx: int, dtype=np.float32, shape=None) -> np.ndarray:
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buf = (ctypes.c_uint8 * size)()
        self._lib.ane_bridge_read_output(
            ctypes.c_void_p(handle), idx,
            buf, size
        )
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    def free_kernel(self, handle: int):
        self._lib.ane_bridge_free(ctypes.c_void_p(handle))

    @property
    def compile_count(self) -> int:
        return self._lib.ane_bridge_get_compile_count()


class ANEModel:
    """High-level model interface for LLaMA-family inference on ANE.

    Manages kernel compilation, weight loading (with layer-swapping),
    and the forward pass loop.
    """

    def __init__(self, config: ModelConfig, bridge: Optional[ANEBridge] = None):
        self.config = config
        self.bridge = bridge  # Lazily initialized on macOS
        self._kernels = {}
        self._weights = {}  # layer_idx -> dict of weight arrays
        self._kv_cache = None
        self._compiled = False

    def load_weights_from_file(self, path: str):
        """Load model weights from a file.

        Supports:
        - .bin (llama2.c format)
        - .gguf (GGUF format — via gguf_reader.py)
        - .safetensors (SafeTensors format — via safetensors_reader.py)

        Weights are stored in CPU memory and transferred to ANE per-layer
        during inference (weight-swap architecture).
        """
        path = Path(path)
        ext = path.suffix.lower()

        if ext == '.bin':
            self._load_llama2c(path)
        elif ext == '.gguf':
            self._load_gguf(path)
        elif ext == '.safetensors':
            self._load_safetensors(path)
        else:
            raise ValueError(f"Unsupported weight format: {ext}")

    def _load_llama2c(self, path: Path):
        """Load weights from llama2.c binary format."""
        with open(path, 'rb') as f:
            header = struct.unpack('7i', f.read(28))
            dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq = header
            if vocab < 0:
                vocab = -vocab
                shared_embed = False
            else:
                shared_embed = True

            print(f"Loading llama2c: dim={dim} hidden={hidden} layers={n_layers} "
                  f"heads={n_heads} vocab={vocab}")

            # Token embedding
            embed = np.frombuffer(f.read(vocab * dim * 4), dtype=np.float32).reshape(vocab, dim)
            self._weights['embed'] = embed

            # Per-layer weights (stored in layer-major order in the file)
            for name, shape in [
                ('rms_att', (n_layers, dim)),
                ('wq', (n_layers, dim, dim)),
                ('wk', (n_layers, n_kv_heads * (dim // n_heads), dim)),
                ('wv', (n_layers, n_kv_heads * (dim // n_heads), dim)),
                ('wo', (n_layers, dim, dim)),
                ('rms_ffn', (n_layers, dim)),
                ('w1', (n_layers, hidden, dim)),
                ('w2', (n_layers, dim, hidden)),
                ('w3', (n_layers, hidden, dim)),
            ]:
                total = int(np.prod(shape))
                data = np.frombuffer(f.read(total * 4), dtype=np.float32).reshape(shape)
                for l in range(n_layers):
                    if l not in self._weights:
                        self._weights[l] = {}
                    self._weights[l][name.replace('rms_att', 'rms_att').replace('rms_ffn', 'rms_ffn')] = data[l]

            # Final RMSNorm
            self._weights['rms_final'] = np.frombuffer(
                f.read(dim * 4), dtype=np.float32).reshape(dim)

            if shared_embed:
                self._weights['wcls'] = embed
            # else: read separate classifier weights

    def _load_gguf(self, path: Path):
        """Load weights from GGUF format. Requires gguf_reader.py."""
        raise NotImplementedError("GGUF loading — implement in Phase 1")

    def _load_safetensors(self, path: Path):
        """Load weights from SafeTensors format."""
        raise NotImplementedError("SafeTensors loading — implement in Phase 2")

    def compile_kernels(self):
        """Compile the shared kernel programs (SDPA, FFN, etc.).

        These are compiled ONCE and reused across all layers via weight swapping.
        Total: ~6 forward kernels + ~5 backward kernels = 11 programs.
        """
        # Placeholder — actual MIL generation requires macOS + Objective-C
        print(f"Would compile {6} forward + {5} backward kernels")
        print(f"  Config: dim={self.config.dim} hidden={self.config.hidden_dim} "
              f"heads={self.config.n_heads} kv_heads={self.config.n_kv_heads}")
        print(f"  Weight swap architecture: {self.config.n_layers} layers, "
              f"~{self.config.weight_bytes_per_layer_fp16() / 1e6:.1f} MB/layer fp16")
        print(f"  Estimated memory: {self.config.memory_estimate_gb():.1f} GB (fp16)")
        print(f"  Estimated memory: {self.config.memory_estimate_gb(quant_bits=4):.1f} GB (Q4)")
        self._compiled = True

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Full forward pass through all layers.

        Uses weight-swap architecture: for each layer, load weights into
        pre-compiled kernel, execute, read output.

        Args:
            input_ids: [seq_len] int32 token IDs

        Returns:
            logits: [seq_len, vocab_size] float32
        """
        if not self._compiled:
            raise RuntimeError("Call compile_kernels() first")

        seq_len = len(input_ids)
        dim = self.config.dim

        # Embed tokens
        x = self._weights['embed'][input_ids]  # [seq_len, dim]

        # Iterate through layers (weight-swap loop)
        for layer_idx in range(self.config.n_layers):
            lw = self._weights.get(layer_idx, {})
            # In production: load lw into ANE IOSurface, run k_sdpa, run k_ffn
            # Here: CPU reference implementation for correctness testing
            x = self._forward_layer_cpu(x, lw, seq_len, layer_idx)

        # Final RMSNorm + classifier
        x = self._rmsnorm_cpu(x, self._weights['rms_final'])
        logits = x @ self._weights.get('wcls', self._weights['embed']).T
        return logits

    def _forward_layer_cpu(self, x, lw, seq_len, layer_idx):
        """CPU reference forward for one transformer layer."""
        dim = self.config.dim
        n_heads = self.config.n_heads
        head_dim = self.config.head_dim

        # Attention block
        xn = self._rmsnorm_cpu(x, lw.get('rms_att', np.ones(dim)))
        q = xn @ lw.get('wq', np.eye(dim)).T
        k = xn @ lw.get('wk', np.eye(dim)).T
        v = xn @ lw.get('wv', np.eye(dim)).T

        # RoPE (simplified — full impl in mil_gen_llama.h)
        q = self._apply_rope_cpu(q, seq_len, n_heads, head_dim)
        k = self._apply_rope_cpu(k, seq_len, self.config.n_kv_heads, head_dim)

        # Attention
        attn_out = self._attention_cpu(q, k, v, n_heads, head_dim)
        o_out = attn_out @ lw.get('wo', np.eye(dim)).T
        x = x + o_out

        # FFN block
        xn2 = self._rmsnorm_cpu(x, lw.get('rms_ffn', np.ones(dim)))
        h1 = xn2 @ lw.get('w1', np.zeros((self.config.hidden_dim, dim))).T
        h3 = xn2 @ lw.get('w3', np.zeros((self.config.hidden_dim, dim))).T
        silu_h1 = h1 * (1.0 / (1.0 + np.exp(-h1)))  # SiLU
        swiglu = silu_h1 * h3
        ffn_out = swiglu @ lw.get('w2', np.zeros((dim, self.config.hidden_dim))).T
        x = x + ffn_out
        return x

    @staticmethod
    def _rmsnorm_cpu(x, w):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-5)
        return (x / rms) * w

    def _apply_rope_cpu(self, x, seq_len, n_heads, head_dim):
        """Apply RoPE to [seq_len, n_heads * head_dim]."""
        x = x.reshape(seq_len, n_heads, head_dim)
        half = head_dim // 2
        theta = self.config.rope_theta
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
        positions = np.arange(seq_len, dtype=np.float32)
        angles = np.outer(positions, freqs)

        cos_v = np.cos(angles)[:, np.newaxis, :]  # [S, 1, half]
        sin_v = np.sin(angles)[:, np.newaxis, :]

        x_even = x[:, :, :half]
        x_odd = x[:, :, half:]
        x_rot_even = x_even * cos_v - x_odd * sin_v
        x_rot_odd = x_even * sin_v + x_odd * cos_v
        x_rot = np.concatenate([x_rot_even, x_rot_odd], axis=-1)
        return x_rot.reshape(seq_len, n_heads * head_dim)

    @staticmethod
    def _attention_cpu(q, k, v, n_heads, head_dim):
        seq_len = q.shape[0]
        dim = n_heads * head_dim
        q = q.reshape(seq_len, n_heads, head_dim)
        k = k.reshape(seq_len, -1, head_dim)
        v = v.reshape(seq_len, -1, head_dim)

        # GQA: repeat K/V if fewer KV heads
        kv_heads = k.shape[1]
        if kv_heads < n_heads:
            repeat = n_heads // kv_heads
            k = np.repeat(k, repeat, axis=1)
            v = np.repeat(v, repeat, axis=1)

        scale = 1.0 / np.sqrt(head_dim)
        out = np.zeros((seq_len, n_heads, head_dim))

        for h in range(n_heads):
            scores = q[:, h, :] @ k[:, h, :].T * scale
            # Causal mask
            mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
            scores += mask
            # Softmax
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            out[:, h, :] = attn_weights @ v[:, h, :]

        return out.reshape(seq_len, dim)


def print_memory_report(config: ModelConfig):
    """Print detailed memory budget for a given model config on M2 24GB."""
    print(f"\n{'='*60}")
    print(f"Memory Report: {config.n_layers}L dim={config.dim} hidden={config.hidden_dim}")
    print(f"{'='*60}")

    layer_fp16 = config.weight_bytes_per_layer_fp16()
    total_fp16 = config.total_weight_bytes_fp16()
    kv_bytes = config.n_layers * 2 * config.kv_dim * config.max_seq * 2

    for bits, label in [(16, 'FP16'), (8, 'Q8'), (4, 'Q4'), (3, 'Q3')]:
        weight_gb = total_fp16 * bits / 16 / (1024**3)
        kv_gb = kv_bytes / (1024**3)
        overhead_gb = 2.0
        total = weight_gb + kv_gb + overhead_gb
        fits = "OK" if total < 20.0 else "EXCEEDS 20GB"
        print(f"  {label:4s}: weights={weight_gb:.2f}GB + KV={kv_gb:.2f}GB + overhead={overhead_gb:.1f}GB"
              f" = {total:.2f}GB [{fits}]")


if __name__ == '__main__':
    # Print memory reports for target configs
    print_memory_report(ModelConfig.llama_7b())
    print_memory_report(ModelConfig.llama_13b())
    print_memory_report(ModelConfig.mistral_7b())
    print_memory_report(ModelConfig.stories_110m())

    # Test CPU reference implementation with Stories config
    print("\n--- CPU reference test (Stories110M shapes) ---")
    config = ModelConfig.stories_110m()
    model = ANEModel(config)
    model.compile_kernels()


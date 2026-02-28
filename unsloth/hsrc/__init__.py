# Copyright 2026 - Holographic Inference Engine
# HSRC: Holographic Spectral Residual Compression for KV Caches

"""
HSRC (Holographic Spectral Residual Compression) accelerates LLM inference
by compressing the KV cache using a three-layer encoding:

1. Holographic Boundary: Exact tokens at block edges + linear interpolation
2. Spectral SVD: Low-rank approximation of residual (INT8 coefficients)  
3. Sparse Residual: Exact corrections for 'needle' tokens

Key insight: Pre-RoPE keys have ~3x lower effective rank than post-RoPE,
enabling much better compression. Keys are stored pre-RoPE and RoPE is
reapplied during attention (free compute in bandwidth-bound decode).

Target: 4.7-6.5x compression with >0.98 attention fidelity.
"""

from .config import HSRCConfig
from .block import CompressedBlock, compress_block, reconstruct_block_keys, reconstruct_block_values
from .cache import HSRCCache, HSRCLayerCache

__all__ = [
    "HSRCConfig",
    "CompressedBlock",
    "compress_block",
    "reconstruct_block_keys",
    "reconstruct_block_values",
    "HSRCCache",
    "HSRCLayerCache",
]

__version__ = "0.1.0"

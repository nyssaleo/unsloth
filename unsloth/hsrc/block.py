# Copyright 2026 - Holographic Inference Engine
# HSRC Block: Three-layer compression unit

"""
HSRCBlock implements the three-layer encoding from the spec:

Layer 1 (Boundary): Store first/last b tokens exactly in FP16.
    Construct linear interpolation of interior from boundary anchors.
    
Layer 2 (Spectral): SVD of (interior - interpolation) residual.
    Coefficients quantized to INT8. Basis stored in FP16.
    
Layer 3 (Sparse): Exact FP16 corrections for 'needle' tokens
    whose post-SVD residual norm exceeds threshold.

Memory layout per block (for keys, block_size=256, b=8, k=12):
    Boundaries:   2 * 8 * D * 2 bytes = 4096 bytes
    Coefficients: 240 * 12 * 1 byte   = 2880 bytes  (INT8!)
    Scale:        4 bytes
    Basis:        12 * D * 2 bytes     = 3072 bytes
    Sparse:       ~4 * D * 2 + 4*4    = ~1040 bytes
    Total:        ~11,092 bytes vs 65,536 standard = ~5.9x compression (keys only)
"""

import torch
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass(frozen=True)
class CompressedBlock:
    """Immutable compressed representation of a KV cache block.
    
    All tensors are on the same device. Coefficient tensors are
    genuinely torch.int8, not simulated quantization on float tensors.
    """
    # Metadata
    start_pos: int          # Position of first token in this block
    block_len: int          # Number of tokens in this block
    boundary_size: int      # b: tokens at each edge stored exactly
    device: torch.device
    
    # Layer 1: Boundary tokens [b, D] in FP16
    K_left: torch.Tensor    # [b, D] float16
    K_right: torch.Tensor   # [b, D] float16  
    V_left: torch.Tensor    # [b, D] float16
    V_right: torch.Tensor   # [b, D] float16
    
    # Layer 2: Spectral SVD of interior residual
    # Keys
    C_K: torch.Tensor       # [interior_len, k_K] int8 (REAL int8!)
    scale_K: torch.Tensor   # [1] float32 — symmetric quantization scale
    B_K: torch.Tensor       # [k_K, D] float16 — right singular vectors
    # Values
    C_V: torch.Tensor       # [interior_len, k_V] int8
    scale_V: torch.Tensor   # [1] float32
    B_V: torch.Tensor       # [k_V, D] float16
    
    # Layer 3: Sparse corrections for needle tokens
    sparse_indices: torch.Tensor   # [n_sparse] int32 — positions relative to interior start
    sparse_K: torch.Tensor         # [n_sparse, D] float16
    sparse_V: torch.Tensor         # [n_sparse, D] float16
    
    @property
    def interior_len(self) -> int:
        return self.block_len - 2 * self.boundary_size
    
    @property
    def n_sparse(self) -> int:
        return self.sparse_indices.shape[0]
    
    @property
    def key_rank(self) -> int:
        return self.B_K.shape[0]
    
    @property
    def value_rank(self) -> int:
        return self.B_V.shape[0]
    
    @property  
    def head_dim(self) -> int:
        return self.B_K.shape[1]
    
    def memory_bytes(self) -> int:
        """Actual GPU memory consumed by this block."""
        mem = 0
        # Boundaries
        mem += self.K_left.nelement() * self.K_left.element_size()
        mem += self.K_right.nelement() * self.K_right.element_size()
        mem += self.V_left.nelement() * self.V_left.element_size()
        mem += self.V_right.nelement() * self.V_right.element_size()
        # Spectral
        mem += self.C_K.nelement() * self.C_K.element_size()  # 1 byte per element!
        mem += self.scale_K.nelement() * self.scale_K.element_size()
        mem += self.B_K.nelement() * self.B_K.element_size()
        mem += self.C_V.nelement() * self.C_V.element_size()
        mem += self.scale_V.nelement() * self.scale_V.element_size()
        mem += self.B_V.nelement() * self.B_V.element_size()
        # Sparse
        mem += self.sparse_indices.nelement() * self.sparse_indices.element_size()
        mem += self.sparse_K.nelement() * self.sparse_K.element_size()
        mem += self.sparse_V.nelement() * self.sparse_V.element_size()
        return mem


def compress_block(
    K_block: torch.Tensor,   # [T_block, D] pre-RoPE keys in float16/32
    V_block: torch.Tensor,   # [T_block, D] values in float16/32
    start_pos: int,
    boundary_size: int = 8,
    key_rank: int = 12,
    value_rank: int = 24,
    max_sparse: int = 8,
    sparse_threshold_mult: float = 3.0,
    use_int8: bool = True,
) -> CompressedBlock:
    """
    Compress a block of KV cache tokens using three-layer HSRC encoding.
    
    This function does the actual work: boundary extraction, interpolation,
    SVD, quantization, and sparse residual detection.
    
    Args:
        K_block: Pre-RoPE key cache for this block [T_block, D]
        V_block: Value cache for this block [T_block, D]
        start_pos: Position of first token in the full sequence
        boundary_size: Number of tokens at each edge to store exactly
        key_rank: SVD rank for key compression
        value_rank: SVD rank for value compression
        max_sparse: Maximum sparse corrections per block
        sparse_threshold_mult: Threshold multiplier for needle detection
        use_int8: Whether to quantize coefficients to INT8
        
    Returns:
        CompressedBlock with all three layers encoded
    """
    T_block, D = K_block.shape
    device = K_block.device
    assert T_block > 2 * boundary_size, \
        f"Block too small ({T_block}) for boundary_size={boundary_size}"
    
    # Work in float32 for SVD numerical stability, keep outputs in float16
    K_f32 = K_block.float()
    V_f32 = V_block.float()
    
    # === Layer 1: Boundary Encoding ===
    K_left = K_block[:boundary_size].to(torch.float16).contiguous()
    K_right = K_block[-boundary_size:].to(torch.float16).contiguous()
    V_left = V_block[:boundary_size].to(torch.float16).contiguous()
    V_right = V_block[-boundary_size:].to(torch.float16).contiguous()
    
    # Interpolate interior from boundary anchors
    i_start = boundary_size
    i_end = T_block - boundary_size
    i_len = i_end - i_start
    
    # Linear interpolation: lerp from last left boundary to first right boundary
    t = torch.linspace(0, 1, i_len, device=device, dtype=torch.float32).unsqueeze(1)
    left_anchor = K_f32[i_start - 1:i_start]   # [1, D] — last boundary token
    right_anchor = K_f32[i_end:i_end + 1]       # [1, D] — first right boundary token
    K_interp = left_anchor * (1 - t) + right_anchor * t  # [i_len, D]
    
    left_anchor_V = V_f32[i_start - 1:i_start]
    right_anchor_V = V_f32[i_end:i_end + 1]
    V_interp = left_anchor_V * (1 - t) + right_anchor_V * t
    
    # Residual from interpolation
    K_residual = K_f32[i_start:i_end] - K_interp  # [i_len, D]
    V_residual = V_f32[i_start:i_end] - V_interp
    
    # === Layer 2: Spectral SVD ===
    C_K_f32, scale_K, B_K, C_K_int8 = _svd_compress(
        K_residual, key_rank, use_int8
    )
    C_V_f32, scale_V, B_V, C_V_int8 = _svd_compress(
        V_residual, value_rank, use_int8
    )
    
    # === Layer 3: Sparse Residual ===
    # CRITICAL: Detect needles from PRE-SVD residual, not post-SVD.
    # SVD captures high-energy outliers first, making them invisible
    # in the post-SVD residual. Pre-SVD detection catches them reliably.
    
    # Detect needles from interpolation residual (pre-SVD)
    pre_svd_norms = torch.norm(K_residual, dim=1)  # [i_len]
    median_norm = torch.median(pre_svd_norms)
    threshold = median_norm * sparse_threshold_mult
    
    above_threshold = torch.where(pre_svd_norms > threshold)[0]
    
    if above_threshold.numel() > max_sparse:
        _, top_idx = torch.topk(pre_svd_norms[above_threshold], max_sparse)
        sparse_positions = above_threshold[top_idx]
    else:
        sparse_positions = above_threshold
    
    if sparse_positions.numel() > 0:
        sparse_positions, _ = torch.sort(sparse_positions)
    
    # Now compute the ACTUAL sparse corrections as the full residual
    # after interpolation + SVD at the needle positions.
    # This accounts for whatever SVD did (or didn't) capture.
    if use_int8:
        K_after_svd = K_interp + (C_K_int8.float() * scale_K) @ B_K.float()
    else:
        K_after_svd = K_interp + C_K_f32 @ B_K.float()
    K_final_residual = K_f32[i_start:i_end] - K_after_svd
    
    if use_int8:
        V_after_svd = V_interp + (C_V_int8.float() * scale_V) @ B_V.float()
    else:
        V_after_svd = V_interp + C_V_f32 @ B_V.float()
    V_final_residual = V_f32[i_start:i_end] - V_after_svd
    
    if sparse_positions.numel() > 0:
        sparse_K_vals = K_final_residual[sparse_positions].to(torch.float16).contiguous()
        sparse_V_vals = V_final_residual[sparse_positions].to(torch.float16).contiguous()
        sparse_idx = sparse_positions.to(torch.int32).contiguous()
    else:
        sparse_K_vals = torch.zeros((0, D), dtype=torch.float16, device=device)
        sparse_V_vals = torch.zeros((0, D), dtype=torch.float16, device=device)
        sparse_idx = torch.zeros((0,), dtype=torch.int32, device=device)
    
    return CompressedBlock(
        start_pos=start_pos,
        block_len=T_block,
        boundary_size=boundary_size,
        device=device,
        K_left=K_left,
        K_right=K_right,
        V_left=V_left,
        V_right=V_right,
        C_K=C_K_int8 if use_int8 else C_K_f32.to(torch.float16),
        scale_K=scale_K,
        B_K=B_K,
        C_V=C_V_int8 if use_int8 else C_V_f32.to(torch.float16),
        scale_V=scale_V,
        B_V=B_V,
        sparse_indices=sparse_idx,
        sparse_K=sparse_K_vals,
        sparse_V=sparse_V_vals,
    )


def _svd_compress(
    residual: torch.Tensor,  # [T, D] float32
    rank: int,
    use_int8: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Truncated SVD + optional INT8 quantization.
    
    Returns:
        C_f32: float32 coefficients [T, k] (for sparse residual computation)
        scale: float32 scalar (INT8 scale, or 1.0 if not using INT8)
        B: float16 basis [k, D]
        C_int8: int8 quantized coefficients [T, k] (or C_f32 cast to float if not using INT8)
    """
    T, D = residual.shape
    k = min(rank, min(T, D))
    
    if k == 0 or T == 0:
        empty_C = torch.zeros((T, 0), dtype=torch.float32, device=residual.device)
        empty_B = torch.zeros((0, D), dtype=torch.float16, device=residual.device)
        scale = torch.ones(1, dtype=torch.float32, device=residual.device)
        return empty_C, scale, empty_B, empty_C.to(torch.int8) if use_int8 else empty_C
    
    # Truncated SVD via torch.linalg.svd
    # For large matrices, randomized SVD would be faster; for Phase 1 use exact
    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
    
    # Truncate to rank k
    U_k = U[:, :k]       # [T, k]
    S_k = S[:k]           # [k]
    Vh_k = Vh[:k, :]      # [k, D]
    
    # Coefficient matrix: C = U_k * S_k
    C_f32 = U_k * S_k.unsqueeze(0)  # [T, k]
    B = Vh_k.to(torch.float16)       # [k, D]
    
    if use_int8:
        # Symmetric INT8 quantization: per-head (per-block) scale
        abs_max = C_f32.abs().max()
        if abs_max > 0:
            scale = abs_max / 127.0
        else:
            scale = torch.ones(1, dtype=torch.float32, device=residual.device)
        
        C_int8 = torch.round(C_f32 / scale).clamp(-127, 127).to(torch.int8)
        scale = scale.reshape(1).to(torch.float32)
    else:
        C_int8 = C_f32  # Keep as float32, will be cast later
        scale = torch.ones(1, dtype=torch.float32, device=residual.device)
    
    return C_f32, scale, B, C_int8


def reconstruct_block_keys(
    block: CompressedBlock,
    cos: Optional[torch.Tensor] = None,   # [T_block, D/2] for RoPE
    sin: Optional[torch.Tensor] = None,   # [T_block, D/2] for RoPE
    apply_rope: bool = True,
) -> torch.Tensor:
    """
    Reconstruct full key cache from compressed block.
    
    If apply_rope=True and cos/sin provided, applies RoPE rotation
    to reconstruct post-RoPE keys ready for attention computation.
    
    Returns: [T_block, D] tensor in float16
    """
    T = block.block_len
    D = block.head_dim
    b = block.boundary_size
    device = block.device
    
    result = torch.zeros((T, D), dtype=torch.float32, device=device)
    
    # Boundaries
    result[:b] = block.K_left.float()
    result[-b:] = block.K_right.float()
    
    # Interior: interpolation + spectral + sparse
    i_len = T - 2 * b
    if i_len > 0:
        t = torch.linspace(0, 1, i_len, device=device, dtype=torch.float32).unsqueeze(1)
        left_anchor = block.K_left[-1:].float()
        right_anchor = block.K_right[:1].float()
        interior = left_anchor * (1 - t) + right_anchor * t  # [i_len, D]
        
        # Add spectral reconstruction
        if block.C_K.dtype == torch.int8:
            C_float = block.C_K.float() * block.scale_K
        else:
            C_float = block.C_K.float()
        interior = interior + C_float @ block.B_K.float()
        
        # Add sparse corrections
        if block.n_sparse > 0:
            for idx in range(block.n_sparse):
                pos = block.sparse_indices[idx].item()
                interior[pos] += block.sparse_K[idx].float()
        
        result[b:T - b] = interior
    
    # Apply RoPE if requested
    if apply_rope and cos is not None and sin is not None:
        result = _apply_rope_to_tensor(result, cos, sin)
    
    return result.to(torch.float16)


def reconstruct_block_values(block: CompressedBlock) -> torch.Tensor:
    """
    Reconstruct full value cache from compressed block.
    Values don't need RoPE — they're stored and used directly.
    
    Returns: [T_block, D] tensor in float16
    """
    T = block.block_len
    D = block.head_dim
    b = block.boundary_size
    device = block.device
    
    result = torch.zeros((T, D), dtype=torch.float32, device=device)
    
    # Boundaries
    result[:b] = block.V_left.float()
    result[-b:] = block.V_right.float()
    
    # Interior
    i_len = T - 2 * b
    if i_len > 0:
        t = torch.linspace(0, 1, i_len, device=device, dtype=torch.float32).unsqueeze(1)
        left_anchor = block.V_left[-1:].float()
        right_anchor = block.V_right[:1].float()
        interior = left_anchor * (1 - t) + right_anchor * t
        
        if block.C_V.dtype == torch.int8:
            C_float = block.C_V.float() * block.scale_V
        else:
            C_float = block.C_V.float()
        interior = interior + C_float @ block.B_V.float()
        
        if block.n_sparse > 0:
            for idx in range(block.n_sparse):
                pos = block.sparse_indices[idx].item()
                interior[pos] += block.sparse_V[idx].float()
        
        result[b:T - b] = interior
    
    return result.to(torch.float16)


def _apply_rope_to_tensor(
    x: torch.Tensor,       # [T, D] float32
    cos: torch.Tensor,     # [T, D/2] 
    sin: torch.Tensor,     # [T, D/2]
) -> torch.Tensor:
    """
    Apply RoPE using the Unsloth/Llama half-split convention:
        out[..., :h] = x[..., :h] * cos - x[..., h:] * sin
        out[..., h:] = x[..., h:] * cos + x[..., :h] * sin
    
    This matches the convention in unsloth/kernels/rope_embedding.py
    and the inline code in LlamaAttention_fast_forward_inference.
    """
    T, D = x.shape
    h = D // 2
    
    # Ensure cos/sin match the sequence length
    cos = cos[:T]  # [T, h]
    sin = sin[:T]  # [T, h]
    
    x0 = x[:, :h]
    x1 = x[:, h:]
    
    out = torch.empty_like(x)
    out[:, :h] = x0 * cos - x1 * sin
    out[:, h:] = x1 * cos + x0 * sin
    
    return out
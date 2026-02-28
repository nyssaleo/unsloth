# Copyright 2026 - Holographic Inference Engine
# HSRC Configuration

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HSRCConfig:
    """
    Configuration for Holographic Spectral Residual Compression.
    
    All parameters have validated defaults from empirical testing
    (see hsrc_final.py and HIE_Technical_Specification_v1.docx).
    """
    
    # === Block structure ===
    block_size: int = 256
    """Number of tokens per compressed block. Must be > 2 * boundary_size.
    256 is the sweet spot: large enough for SVD to work well,
    small enough that each block is coherent within a topic."""
    
    boundary_size: int = 8
    """Number of exact tokens stored at each edge of a block.
    These are the 'holographic boundary' — they anchor the
    linear interpolation of the interior."""
    
    # === SVD ranks ===
    key_rank: int = 12
    """Truncated SVD rank for key residuals (after boundary interpolation).
    12 is sufficient because the interpolation removes the smooth trend,
    leaving a lower-energy residual. Keys need less fidelity than values
    because attention is sparser than value mixing."""
    
    value_rank: int = 24
    """Truncated SVD rank for value residuals.
    Higher than key_rank because output quality depends directly on
    value reconstruction fidelity (output = attention_weights @ V)."""
    
    # === Sparse residual ===
    max_sparse_per_block: int = 8
    """Maximum number of sparse corrections ('needles') per block.
    These are tokens whose SVD residual norm exceeds the threshold,
    indicating they carry anomalous information (names, numbers, etc.)."""
    
    sparse_threshold_multiplier: float = 3.0
    """A token is flagged as sparse if its post-SVD residual L2 norm
    exceeds median_norm * this multiplier. 3.0 catches ~1-3% of tokens."""
    
    # === Quantization ===
    use_int8_coefficients: bool = True
    """Quantize SVD coefficient matrices (C_K, C_V) to INT8.
    Empirically adds <0.06% degradation — essentially free 2x compression
    on the coefficient storage."""
    
    # === Hot buffer ===
    hot_buffer_extra: int = 16
    """Extra tokens to keep uncompressed beyond the current block boundary.
    This ensures we always have enough context for the next block's
    boundary encoding. Total hot = block_size + hot_buffer_extra."""
    
    # === Integration mode ===
    store_pre_rope: bool = True
    """Store pre-RoPE keys and reconstruct with RoPE during attention.
    This is the core insight: pre-RoPE keys have ~3x lower effective rank
    than post-RoPE keys. Should always be True for RoPE models."""
    
    # === Compression scheduling ===
    min_seq_len_to_compress: int = 512
    """Don't compress until we have at least this many tokens.
    Avoids overhead for short sequences where compression isn't needed."""
    
    # === Debug / profiling ===
    verify_reconstruction: bool = False
    """If True, verify reconstruction quality during compression (slow).
    Use only for debugging."""
    
    def __post_init__(self):
        assert self.block_size > 2 * self.boundary_size, \
            f"block_size ({self.block_size}) must be > 2 * boundary_size ({2 * self.boundary_size})"
        assert self.key_rank > 0 and self.value_rank > 0
        assert self.max_sparse_per_block >= 0
        assert self.sparse_threshold_multiplier > 0
    
    @property
    def interior_size(self) -> int:
        """Number of interior tokens per block (total - 2 * boundary)."""
        return self.block_size - 2 * self.boundary_size
    
    @property
    def hot_buffer_size(self) -> int:
        """Total hot buffer capacity: enough for one full block plus overlap."""
        return self.block_size + self.hot_buffer_extra + self.boundary_size
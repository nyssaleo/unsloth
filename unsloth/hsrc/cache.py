# Copyright 2026 - Holographic Inference Engine
# HSRC Cache: DynamicCache subclass for compressed KV storage

"""
HSRCCache manages the full KV cache state across all layers and heads.

Architecture per layer:
    ┌─────────────────────────────────────────────────────────┐
    │ Cold blocks:  [CompressedBlock_0, CompressedBlock_1, ...] │
    │   - Each block: boundary + INT8 SVD + sparse            │
    │   - Pre-RoPE keys, raw values                           │
    │                                                         │
    │ Hot buffer:   [T_hot, n_kv_heads, D] in FP16            │
    │   - Uncompressed recent tokens                          │
    │   - Pre-RoPE keys, raw values                           │
    │   - When it fills up → compress oldest block_size       │
    └─────────────────────────────────────────────────────────┘

Integration with Unsloth:
    - past_key_values[layer_idx] returns (K_view, V_view) where
      K_view is the hot buffer's post-RoPE keys (for standard attention path)
    - The inference loop can call hsrc_attention() for the full 
      compressed + hot attention
"""

import torch
from typing import Optional, Tuple, List, Dict
from .config import HSRCConfig
from .block import CompressedBlock, compress_block, reconstruct_block_keys, reconstruct_block_values, _apply_rope_to_tensor


class HSRCLayerCache:
    """
    Per-layer KV cache with HSRC compression.
    
    Manages cold (compressed) blocks and a hot (uncompressed) buffer
    for a single attention layer across all KV heads.
    """
    
    def __init__(
        self,
        config: HSRCConfig,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Cold storage: list of compressed blocks per head
        # blocks[head_idx] = [CompressedBlock, ...]
        self.blocks: List[List[CompressedBlock]] = [[] for _ in range(n_kv_heads)]
        
        # Hot buffer: uncompressed recent tokens, stored pre-RoPE
        # Shape: [max_hot, n_kv_heads, head_dim]
        max_hot = config.hot_buffer_size
        self.K_hot = torch.zeros((max_hot, n_kv_heads, head_dim), dtype=dtype, device=device)
        self.V_hot = torch.zeros((max_hot, n_kv_heads, head_dim), dtype=dtype, device=device)
        self.hot_len = 0  # Current number of tokens in hot buffer
        
        # Track total compressed tokens
        self.compressed_len = 0  # Total tokens across all cold blocks
        
        # RoPE tables (set during first use from the model's rotary embedding)
        self._cos_cached: Optional[torch.Tensor] = None  # [max_seq, head_dim/2]
        self._sin_cached: Optional[torch.Tensor] = None
    
    @property
    def total_len(self) -> int:
        """Total sequence length (compressed + hot)."""
        return self.compressed_len + self.hot_len
    
    def set_rope_cache(self, cos: torch.Tensor, sin: torch.Tensor):
        """Set RoPE tables from the model's rotary embedding."""
        self._cos_cached = cos.to(self.device)
        self._sin_cached = sin.to(self.device)
    
    def append_token(self, K_pre_rope: torch.Tensor, V: torch.Tensor):
        """
        Append a single token's KV to the cache.
        
        Args:
            K_pre_rope: [1, n_kv_heads, head_dim] pre-RoPE key (or [n_kv_heads, head_dim])
            V: [1, n_kv_heads, head_dim] value (or [n_kv_heads, head_dim])
        """
        if K_pre_rope.dim() == 3:
            K_pre_rope = K_pre_rope.squeeze(0)  # [n_kv_heads, head_dim]
            V = V.squeeze(0)
        
        # Check if hot buffer needs to be expanded
        if self.hot_len >= self.K_hot.shape[0]:
            # Grow hot buffer
            new_size = self.K_hot.shape[0] + self.config.block_size
            self.K_hot = _resize_buffer(self.K_hot, new_size)
            self.V_hot = _resize_buffer(self.V_hot, new_size)
        
        self.K_hot[self.hot_len] = K_pre_rope.to(self.dtype)
        self.V_hot[self.hot_len] = V.to(self.dtype)
        self.hot_len += 1
        
        # Check if we should compress
        self._maybe_compress()
    
    def _maybe_compress(self):
        """Compress the oldest block_size tokens if hot buffer is full enough."""
        cfg = self.config
        
        # Need at least block_size + boundary_size tokens to compress
        # (keep boundary_size tokens for overlap with next block)
        compress_trigger = cfg.block_size + cfg.boundary_size
        
        # Don't compress until we have enough total tokens
        if self.total_len < cfg.min_seq_len_to_compress:
            return
        
        while self.hot_len >= compress_trigger:
            self._compress_one_block()
    
    def _compress_one_block(self):
        """Compress the oldest block_size tokens from the hot buffer."""
        cfg = self.config
        bs = cfg.block_size
        
        for head_idx in range(self.n_kv_heads):
            K_block = self.K_hot[:bs, head_idx, :].contiguous()  # [bs, D]
            V_block = self.V_hot[:bs, head_idx, :].contiguous()  # [bs, D]
            
            compressed = compress_block(
                K_block=K_block,
                V_block=V_block,
                start_pos=self.compressed_len,
                boundary_size=cfg.boundary_size,
                key_rank=cfg.key_rank,
                value_rank=cfg.value_rank,
                max_sparse=cfg.max_sparse_per_block,
                sparse_threshold_mult=cfg.sparse_threshold_multiplier,
                use_int8=cfg.use_int8_coefficients,
            )
            self.blocks[head_idx].append(compressed)
        
        # Shift hot buffer: remove the compressed tokens
        remaining = self.hot_len - bs
        if remaining > 0:
            self.K_hot[:remaining] = self.K_hot[bs:bs + remaining].clone()
            self.V_hot[:remaining] = self.V_hot[bs:bs + remaining].clone()
        self.hot_len = remaining
        self.compressed_len += bs
    
    def get_hot_kv_post_rope(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hot buffer keys/values with RoPE applied to keys.
        
        Returns:
            K_post: [n_kv_heads, hot_len, head_dim] post-RoPE keys
            V: [n_kv_heads, hot_len, head_dim] values
        """
        if self.hot_len == 0:
            empty = torch.zeros((self.n_kv_heads, 0, self.head_dim),
                              dtype=self.dtype, device=self.device)
            return empty, empty
        
        K_pre = self.K_hot[:self.hot_len]  # [hot_len, n_kv_heads, head_dim]
        V = self.V_hot[:self.hot_len]      # [hot_len, n_kv_heads, head_dim]
        
        # Apply RoPE to keys
        # cos/sin are indexed by absolute position
        start_pos = self.compressed_len
        end_pos = start_pos + self.hot_len
        
        cos_slice = cos[start_pos:end_pos]  # [hot_len, head_dim/2]
        sin_slice = sin[start_pos:end_pos]
        
        K_post = torch.zeros_like(K_pre)
        h = self.head_dim // 2
        
        # Apply RoPE per head (all heads use same position)
        for hi in range(self.n_kv_heads):
            K_h = K_pre[:, hi, :].float()  # [hot_len, D]
            K_post[:, hi, :] = _apply_rope_to_tensor(K_h, cos_slice, sin_slice).to(self.dtype)
        
        # Permute to [n_kv_heads, hot_len, head_dim]
        return K_post.permute(1, 0, 2), V.permute(1, 0, 2)
    
    def reconstruct_all_keys_post_rope(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct ALL keys (compressed + hot) with RoPE applied.
        
        Returns: [n_kv_heads, total_len, head_dim] post-RoPE keys
        
        This is used in Phase 1 for the simple (non-fused) attention path.
        In Phase 2, this will be replaced by the fused Triton kernel.
        """
        total = self.total_len
        if total == 0:
            return torch.zeros((self.n_kv_heads, 0, self.head_dim),
                             dtype=self.dtype, device=self.device)
        
        all_keys = torch.zeros((self.n_kv_heads, total, self.head_dim),
                              dtype=torch.float32, device=self.device)
        
        for hi in range(self.n_kv_heads):
            pos = 0
            # Reconstruct from compressed blocks
            for block in self.blocks[hi]:
                T_b = block.block_len
                # Get RoPE for this block's positions
                cos_b = cos[block.start_pos:block.start_pos + T_b]
                sin_b = sin[block.start_pos:block.start_pos + T_b]
                
                K_b = reconstruct_block_keys(block, cos_b, sin_b, apply_rope=True)
                all_keys[hi, pos:pos + T_b] = K_b.float()
                pos += T_b
            
            # Hot buffer with RoPE
            if self.hot_len > 0:
                start_pos = self.compressed_len
                cos_h = cos[start_pos:start_pos + self.hot_len]
                sin_h = sin[start_pos:start_pos + self.hot_len]
                K_hot_h = self.K_hot[:self.hot_len, hi, :].float()
                K_hot_roped = _apply_rope_to_tensor(K_hot_h, cos_h, sin_h)
                all_keys[hi, pos:pos + self.hot_len] = K_hot_roped
        
        return all_keys.to(self.dtype)
    
    def reconstruct_all_values(self) -> torch.Tensor:
        """
        Reconstruct ALL values (compressed + hot).
        
        Returns: [n_kv_heads, total_len, head_dim] values
        """
        total = self.total_len
        if total == 0:
            return torch.zeros((self.n_kv_heads, 0, self.head_dim),
                             dtype=self.dtype, device=self.device)
        
        all_values = torch.zeros((self.n_kv_heads, total, self.head_dim),
                                dtype=torch.float32, device=self.device)
        
        for hi in range(self.n_kv_heads):
            pos = 0
            for block in self.blocks[hi]:
                T_b = block.block_len
                V_b = reconstruct_block_values(block)
                all_values[hi, pos:pos + T_b] = V_b.float()
                pos += T_b
            
            if self.hot_len > 0:
                all_values[hi, pos:pos + self.hot_len] = \
                    self.V_hot[:self.hot_len, hi, :].float()
        
        return all_values.to(self.dtype)
    
    def memory_bytes(self) -> Dict[str, int]:
        """Report memory usage breakdown."""
        cold_bytes = sum(
            block.memory_bytes()
            for head_blocks in self.blocks
            for block in head_blocks
        )
        hot_bytes = self.hot_len * self.n_kv_heads * self.head_dim * 2 * 2  # K+V in FP16
        standard_bytes = self.total_len * self.n_kv_heads * self.head_dim * 2 * 2  # K+V FP16
        
        return {
            "cold_bytes": cold_bytes,
            "hot_bytes": hot_bytes,
            "total_bytes": cold_bytes + hot_bytes,
            "standard_bytes": standard_bytes,
            "compression_ratio": standard_bytes / max(cold_bytes + hot_bytes, 1),
            "n_compressed_blocks": sum(len(hb) for hb in self.blocks),
            "compressed_tokens": self.compressed_len,
            "hot_tokens": self.hot_len,
        }


class HSRCCache:
    """
    Full HSRC cache across all layers.
    
    This is the top-level object that replaces past_key_values in the
    inference loop. It manages HSRCLayerCache instances for each layer
    and provides the interface expected by Unsloth's attention functions.
    
    Usage with Unsloth:
        past_key_values[layer_idx] returns a tuple-like (K, V) that
        provides the initial hot buffer for the paged attention init.
        
        During decode, HSRC intercepts before RoPE and stores pre-RoPE
        keys, then provides reconstructed post-RoPE keys for attention.
    """
    
    def __init__(
        self,
        config: HSRCConfig,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.num_layers = num_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        self.layers: List[HSRCLayerCache] = [
            HSRCLayerCache(config, n_kv_heads, head_dim, device, dtype)
            for _ in range(num_layers)
        ]
    
    def __len__(self):
        return self.num_layers
    
    def __getitem__(self, layer_idx: int) -> HSRCLayerCache:
        return self.layers[layer_idx]
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total cached sequence length."""
        return self.layers[layer_idx].total_len
    
    def memory_report(self) -> Dict[str, int]:
        """Aggregate memory usage across all layers."""
        reports = [layer.memory_bytes() for layer in self.layers]
        total_cold = sum(r["cold_bytes"] for r in reports)
        total_hot = sum(r["hot_bytes"] for r in reports)
        total_standard = sum(r["standard_bytes"] for r in reports)
        
        return {
            "cold_bytes": total_cold,
            "hot_bytes": total_hot,
            "total_bytes": total_cold + total_hot,
            "standard_bytes": total_standard,
            "compression_ratio": total_standard / max(total_cold + total_hot, 1),
            "total_compressed_blocks": sum(r["n_compressed_blocks"] for r in reports),
            "total_compressed_tokens": sum(r["compressed_tokens"] for r in reports) // self.num_layers,
            "hot_tokens": reports[0]["hot_tokens"] if reports else 0,
        }
    
    @staticmethod
    def from_prefill_cache(
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        config: HSRCConfig,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> 'HSRCCache':
        """
        Create an HSRCCache from prefill-generated standard cache.
        
        After prefill, Unsloth gives us past_key_values as a list of
        (K, V) tuples with shape [bsz, n_kv_heads, seq_len, head_dim].
        These are POST-RoPE keys.
        
        For HSRC, we need PRE-RoPE keys. We undo the RoPE rotation
        using the inverse (cos, -sin), then store in the HSRC cache.
        
        Handles both RoPE formats:
          - [seq_len, head_dim/2]: mathematical convention
          - [seq_len, head_dim]:   Unsloth convention (cat((cos,cos), dim=-1))
        
        This is a one-time conversion at the start of decode.
        """
        num_layers = len(past_key_values)
        K0, V0 = past_key_values[0]
        bsz, n_kv_heads, seq_len, head_dim = K0.shape
        assert bsz == 1, "HSRC currently supports batch_size=1"
        
        device = K0.device
        dtype = K0.dtype
        h = head_dim // 2
        
        cache = HSRCCache(config, num_layers, n_kv_heads, head_dim, device, dtype)
        
        # Set RoPE tables
        for layer in cache.layers:
            layer.set_rope_cache(cos, sin)
        
        # Prepare RoPE tables for un-rotation
        cos_seq = cos[:seq_len].float()  # [seq_len, D] or [seq_len, D/2]
        sin_seq = sin[:seq_len].float()
        
        # Auto-detect Unsloth doubled format: [T, D] → take first half
        if cos_seq.shape[-1] == head_dim:
            cos_seq = cos_seq[:, :h]  # [seq_len, h]
            sin_seq = sin_seq[:, :h]
        
        # Broadcast for vectorized un-rotation: [1, seq_len, h]
        cos_b = cos_seq.unsqueeze(0)
        sin_b = sin_seq.unsqueeze(0)
        
        for layer_idx in range(num_layers):
            K_post, V = past_key_values[layer_idx]
            # K_post: [1, n_kv_heads, seq_len, head_dim] — post-RoPE
            # V: [1, n_kv_heads, seq_len, head_dim]
            
            K_post_f = K_post.squeeze(0).float()  # [n_kv_heads, seq_len, head_dim]
            V_sq = V.squeeze(0)                    # [n_kv_heads, seq_len, head_dim]
            
            # Vectorized inverse RoPE: undo rotation with (cos, -sin)
            # Forward:  post[:h] = pre[:h]*cos - pre[h:]*sin
            #           post[h:] = pre[h:]*cos + pre[:h]*sin
            # Inverse:  pre[:h] = post[:h]*cos + post[h:]*sin
            #           pre[h:] = post[h:]*cos - post[:h]*sin
            k0 = K_post_f[:, :, :h]  # [n_kv_heads, seq_len, h]
            k1 = K_post_f[:, :, h:]
            
            K_pre = torch.empty_like(K_post_f)
            K_pre[:, :, :h] = k0 * cos_b + k1 * sin_b
            K_pre[:, :, h:] = k1 * cos_b - k0 * sin_b
            
            # Ensure hot buffer is large enough
            layer_cache = cache.layers[layer_idx]
            if seq_len > layer_cache.K_hot.shape[0]:
                new_size = seq_len + config.block_size  # Extra room
                layer_cache.K_hot = _resize_buffer(layer_cache.K_hot, new_size)
                layer_cache.V_hot = _resize_buffer(layer_cache.V_hot, new_size)
            
            # Store in hot buffer: K_hot is [max_hot, n_kv_heads, head_dim]
            # K_pre is [n_kv_heads, seq_len, head_dim] → permute to [seq_len, n_kv_heads, head_dim]
            layer_cache.K_hot[:seq_len] = K_pre.permute(1, 0, 2).to(dtype)
            layer_cache.V_hot[:seq_len] = V_sq.permute(1, 0, 2).to(dtype)
            layer_cache.hot_len = seq_len
            
            # Trigger compression of full blocks
            layer_cache._maybe_compress()
        
        return cache


def _resize_buffer(buf: torch.Tensor, new_first_dim: int) -> torch.Tensor:
    """Resize the first dimension of a buffer, preserving existing data."""
    if new_first_dim <= buf.shape[0]:
        return buf
    new_buf = torch.zeros(
        (new_first_dim, *buf.shape[1:]),
        dtype=buf.dtype, device=buf.device
    )
    new_buf[:buf.shape[0]] = buf
    return new_buf
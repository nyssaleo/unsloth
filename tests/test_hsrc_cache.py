"""
Tests for HSRCLayerCache and HSRCCache.

Covers: hot buffer management, compression triggers, block creation,
attention computation, memory tracking, and the DynamicCache-like API.
"""
import pytest
import torch
import numpy as np
from hsrc.config import HSRCConfig
from hsrc.cache import HSRCLayerCache, HSRCCache, _resize_buffer
from hsrc.block import reconstruct_block_keys, reconstruct_block_values, _apply_rope_to_tensor

from tests.conftest import HEAD_DIM, N_KV_HEADS, BLOCK_SIZE, BOUNDARY_SIZE, _generate_kv_block


def _make_config(**overrides) -> HSRCConfig:
    """Create config with test-friendly defaults."""
    defaults = dict(
        block_size=BLOCK_SIZE,
        boundary_size=BOUNDARY_SIZE,
        key_rank=12,
        value_rank=24,
        max_sparse_per_block=8,
        sparse_threshold_multiplier=3.0,
        use_int8_coefficients=True,
        hot_buffer_extra=16,
        min_seq_len_to_compress=0,  # Compress immediately for tests
    )
    defaults.update(overrides)
    return HSRCConfig(**defaults)


def _make_layer_cache(config=None, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM):
    """Create a fresh HSRCLayerCache."""
    if config is None:
        config = _make_config()
    return HSRCLayerCache(config, n_kv_heads, head_dim, device=torch.device("cpu"))


def _append_n_tokens(cache: HSRCLayerCache, n: int, seed=42):
    """Append n synthetic tokens to a layer cache."""
    rng = np.random.RandomState(seed)
    for i in range(n):
        K = torch.from_numpy(rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32))
        V = torch.from_numpy(rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32))
        cache.append_token(K, V)


# ── HSRCLayerCache basic ────────────────────────────────────────────────────

class TestHSRCLayerCacheBasic:
    """Test basic hot buffer operations."""
    
    def test_initial_state(self):
        cache = _make_layer_cache()
        assert cache.hot_len == 0
        assert cache.compressed_len == 0
        assert cache.total_len == 0
        assert all(len(hb) == 0 for hb in cache.blocks)
    
    def test_append_one_token(self):
        cache = _make_layer_cache()
        K = torch.randn(N_KV_HEADS, HEAD_DIM)
        V = torch.randn(N_KV_HEADS, HEAD_DIM)
        cache.append_token(K, V)
        assert cache.hot_len == 1
        assert cache.total_len == 1
        assert cache.compressed_len == 0
    
    def test_append_with_batch_dim(self):
        """append_token should handle [1, n_kv_heads, D] input."""
        cache = _make_layer_cache()
        K = torch.randn(1, N_KV_HEADS, HEAD_DIM)
        V = torch.randn(1, N_KV_HEADS, HEAD_DIM)
        cache.append_token(K, V)
        assert cache.hot_len == 1
    
    def test_hot_buffer_grows(self):
        cache = _make_layer_cache()
        _append_n_tokens(cache, 50)
        assert cache.hot_len == 50
        assert cache.total_len == 50
    
    def test_total_len_equals_compressed_plus_hot(self):
        cache = _make_layer_cache()
        _append_n_tokens(cache, 100)
        assert cache.total_len == cache.compressed_len + cache.hot_len
    
    def test_stored_values_match_input(self):
        """Verify that appended tokens are actually stored in the buffer."""
        cache = _make_layer_cache()
        K = torch.randn(N_KV_HEADS, HEAD_DIM)
        V = torch.randn(N_KV_HEADS, HEAD_DIM)
        cache.append_token(K, V)
        stored_K = cache.K_hot[0]
        stored_V = cache.V_hot[0]
        assert torch.allclose(stored_K, K.to(torch.float16), atol=1e-3)
        assert torch.allclose(stored_V, V.to(torch.float16), atol=1e-3)


# ── Compression trigger ─────────────────────────────────────────────────────

class TestCompressionTrigger:
    """Test when compression triggers and block creation."""
    
    def test_no_compression_below_threshold(self):
        """Don't compress until hot buffer reaches block_size + boundary_size."""
        cache = _make_layer_cache()
        _append_n_tokens(cache, BLOCK_SIZE)
        # block_size=256, compress_trigger = block_size + boundary_size = 264
        # At exactly 256 tokens, no compression yet
        assert cache.compressed_len == 0
    
    def test_compression_at_trigger(self):
        """Compression should trigger at block_size + boundary_size tokens."""
        cache = _make_layer_cache()
        trigger = BLOCK_SIZE + BOUNDARY_SIZE  # 264
        _append_n_tokens(cache, trigger)
        # Should have compressed one block
        assert cache.compressed_len == BLOCK_SIZE
        assert cache.hot_len == trigger - BLOCK_SIZE  # 8
        assert cache.total_len == trigger
    
    def test_compressed_block_created_per_head(self):
        """Each head should get its own CompressedBlock."""
        cache = _make_layer_cache()
        _append_n_tokens(cache, BLOCK_SIZE + BOUNDARY_SIZE)
        for head_idx in range(N_KV_HEADS):
            assert len(cache.blocks[head_idx]) == 1
    
    def test_multiple_blocks(self):
        """Appending 3*block_size tokens should create 2+ blocks."""
        cache = _make_layer_cache()
        _append_n_tokens(cache, 3 * BLOCK_SIZE)
        # With trigger at bs + b = 264, after 768 tokens:
        # First compression at 264: compresses 256, leaves 8
        # Second at 264+256=520: wait... let me think.
        # Actually it's a while loop, so after 768 tokens are added:
        # hot_len can trigger compression multiple times per append_token
        # But tokens are added one at a time, so:
        # After 264 tokens: compress → compressed_len=256, hot_len=8
        # After 264+256=520 tokens: compress → compressed_len=512, hot_len=8
        # After 520+256=776 tokens: would need 776 but we only have 768.
        # So compressed_len=512, hot_len=256
        assert cache.compressed_len >= 2 * BLOCK_SIZE
        for head_idx in range(N_KV_HEADS):
            assert len(cache.blocks[head_idx]) >= 2
    
    def test_min_seq_len_prevents_early_compression(self):
        """With min_seq_len > 0, compression should wait."""
        config = _make_config(min_seq_len_to_compress=512)
        cache = _make_layer_cache(config=config)
        _append_n_tokens(cache, BLOCK_SIZE + BOUNDARY_SIZE)
        # total_len = 264 < 512, so no compression
        assert cache.compressed_len == 0
        assert cache.hot_len == BLOCK_SIZE + BOUNDARY_SIZE
    
    def test_min_seq_len_then_compress(self):
        """After exceeding min_seq_len, compression should catch up."""
        config = _make_config(min_seq_len_to_compress=300)
        cache = _make_layer_cache(config=config)
        _append_n_tokens(cache, 300)
        # At 300 tokens, total_len >= 300, hot_len=300 >= 264 → compress
        assert cache.compressed_len >= BLOCK_SIZE


# ── Hot buffer with RoPE ─────────────────────────────────────────────────────

class TestHotBufferRoPE:
    """Test get_hot_kv_post_rope method."""
    
    def test_empty_hot_buffer(self, rope_tables):
        cache = _make_layer_cache()
        cos, sin = rope_tables
        K, V = cache.get_hot_kv_post_rope(cos, sin)
        assert K.shape == (N_KV_HEADS, 0, HEAD_DIM)
        assert V.shape == (N_KV_HEADS, 0, HEAD_DIM)
    
    def test_hot_buffer_shape(self, rope_tables):
        cache = _make_layer_cache()
        _append_n_tokens(cache, 10)
        cos, sin = rope_tables
        K, V = cache.get_hot_kv_post_rope(cos, sin)
        assert K.shape == (N_KV_HEADS, 10, HEAD_DIM)
        assert V.shape == (N_KV_HEADS, 10, HEAD_DIM)
    
    def test_rope_applied_to_keys_not_values(self, rope_tables):
        """Keys should have RoPE applied; values should not."""
        cache = _make_layer_cache()
        K_in = torch.randn(N_KV_HEADS, HEAD_DIM)
        V_in = torch.randn(N_KV_HEADS, HEAD_DIM)
        cache.append_token(K_in, V_in)
        
        cos, sin = rope_tables
        K_post, V_out = cache.get_hot_kv_post_rope(cos, sin)
        
        # Values should match what was stored (in FP16)
        V_stored = V_in.to(torch.float16)
        assert torch.allclose(V_out[0, 0, :].to(torch.float16), 
                             V_stored[0, :].unsqueeze(0).squeeze(), atol=1e-2)


# ── Full reconstruction ─────────────────────────────────────────────────────

class TestFullReconstruction:
    """Test reconstruct_all_keys_post_rope and reconstruct_all_values."""
    
    def test_empty_cache_reconstruction(self, rope_tables):
        cache = _make_layer_cache()
        cos, sin = rope_tables
        K = cache.reconstruct_all_keys_post_rope(cos, sin)
        V = cache.reconstruct_all_values()
        assert K.shape == (N_KV_HEADS, 0, HEAD_DIM)
        assert V.shape == (N_KV_HEADS, 0, HEAD_DIM)
    
    def test_hot_only_reconstruction(self, rope_tables):
        """With only hot tokens, reconstruction = hot buffer with RoPE."""
        cache = _make_layer_cache()
        _append_n_tokens(cache, 20)
        cos, sin = rope_tables
        K_all = cache.reconstruct_all_keys_post_rope(cos, sin)
        V_all = cache.reconstruct_all_values()
        assert K_all.shape == (N_KV_HEADS, 20, HEAD_DIM)
        assert V_all.shape == (N_KV_HEADS, 20, HEAD_DIM)
    
    def test_mixed_reconstruction_shape(self, rope_tables):
        """After compression, reconstruction has compressed + hot tokens."""
        cache = _make_layer_cache()
        total_tokens = BLOCK_SIZE + BOUNDARY_SIZE + 20  # Force 1 block + 28 hot
        _append_n_tokens(cache, total_tokens)
        cos, sin = rope_tables
        K_all = cache.reconstruct_all_keys_post_rope(cos, sin)
        V_all = cache.reconstruct_all_values()
        assert K_all.shape[0] == N_KV_HEADS
        assert K_all.shape[1] == total_tokens
        assert V_all.shape[1] == total_tokens


# ── Memory tracking ──────────────────────────────────────────────────────────

class TestMemoryTracking:
    """Verify memory reporting is accurate and shows compression benefits."""
    
    def test_memory_report_keys_present(self):
        cache = _make_layer_cache()
        _append_n_tokens(cache, BLOCK_SIZE + BOUNDARY_SIZE)
        report = cache.memory_bytes()
        assert "cold_bytes" in report
        assert "hot_bytes" in report
        assert "total_bytes" in report
        assert "standard_bytes" in report
        assert "compression_ratio" in report
    
    def test_cold_bytes_positive_after_compression(self):
        cache = _make_layer_cache()
        _append_n_tokens(cache, BLOCK_SIZE + BOUNDARY_SIZE)
        report = cache.memory_bytes()
        assert report["cold_bytes"] > 0
    
    def test_compression_ratio_exceeds_1(self):
        """Compressed cache should use less memory than standard."""
        cache = _make_layer_cache()
        _append_n_tokens(cache, BLOCK_SIZE + BOUNDARY_SIZE)
        report = cache.memory_bytes()
        assert report["compression_ratio"] > 1.0, \
            f"Compression ratio {report['compression_ratio']:.2f} <= 1.0"
    
    def test_total_bytes_less_than_standard(self):
        cache = _make_layer_cache()
        _append_n_tokens(cache, 2 * BLOCK_SIZE + BOUNDARY_SIZE)
        report = cache.memory_bytes()
        assert report["total_bytes"] < report["standard_bytes"]


# ── HSRCCache (multi-layer) ─────────────────────────────────────────────────

class TestHSRCCache:
    """Test the full multi-layer cache wrapper."""
    
    def test_creation(self):
        config = _make_config()
        cache = HSRCCache(config, num_layers=4, n_kv_heads=N_KV_HEADS,
                         head_dim=HEAD_DIM, device=torch.device("cpu"))
        assert len(cache) == 4
        assert cache.num_layers == 4
    
    def test_getitem(self):
        config = _make_config()
        cache = HSRCCache(config, num_layers=4, n_kv_heads=N_KV_HEADS,
                         head_dim=HEAD_DIM, device=torch.device("cpu"))
        layer0 = cache[0]
        assert isinstance(layer0, HSRCLayerCache)
    
    def test_get_seq_length_empty(self):
        config = _make_config()
        cache = HSRCCache(config, num_layers=4, n_kv_heads=N_KV_HEADS,
                         head_dim=HEAD_DIM, device=torch.device("cpu"))
        assert cache.get_seq_length() == 0
    
    def test_get_seq_length_after_tokens(self):
        config = _make_config()
        cache = HSRCCache(config, num_layers=4, n_kv_heads=N_KV_HEADS,
                         head_dim=HEAD_DIM, device=torch.device("cpu"))
        # Add tokens to layer 0
        rng = np.random.RandomState(42)
        for _ in range(10):
            K = torch.from_numpy(rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32))
            V = torch.from_numpy(rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32))
            cache[0].append_token(K, V)
        assert cache.get_seq_length(0) == 10
    
    def test_memory_report_aggregates(self):
        config = _make_config()
        num_layers = 4
        cache = HSRCCache(config, num_layers=num_layers, n_kv_heads=N_KV_HEADS,
                         head_dim=HEAD_DIM, device=torch.device("cpu"))
        
        # Add tokens to all layers
        rng = np.random.RandomState(42)
        for layer_idx in range(num_layers):
            for _ in range(BLOCK_SIZE + BOUNDARY_SIZE):
                K = torch.from_numpy(rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32))
                V = torch.from_numpy(rng.randn(N_KV_HEADS, HEAD_DIM).astype(np.float32))
                cache[layer_idx].append_token(K, V)
        
        report = cache.memory_report()
        assert report["cold_bytes"] > 0
        assert report["total_bytes"] > 0
        assert report["compression_ratio"] > 1.0


# ── from_prefill_cache ──────────────────────────────────────────────────────

class TestFromPrefillCache:
    """Test creating HSRC cache from standard prefill output."""
    
    def test_basic_creation(self, rope_tables):
        """Create HSRC cache from synthetic prefill cache."""
        cos, sin = rope_tables
        num_layers = 2
        seq_len = 32  # Short for test speed
        
        # Create mock prefill output: list of (K, V) with post-RoPE keys
        past_kv = []
        for _ in range(num_layers):
            K_pre = torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM)
            # Apply RoPE to create post-RoPE keys
            K_post = torch.zeros_like(K_pre)
            h = HEAD_DIM // 2
            for t in range(seq_len):
                for hi in range(N_KV_HEADS):
                    k = K_pre[0, hi, t, :].float()
                    k0, k1 = k[:h], k[h:]
                    K_post[0, hi, t, :h] = k0 * cos[t] - k1 * sin[t]
                    K_post[0, hi, t, h:] = k1 * cos[t] + k0 * sin[t]
            V = torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM)
            past_kv.append((K_post, V))
        
        config = _make_config()
        cache = HSRCCache.from_prefill_cache(past_kv, config, cos, sin)
        
        assert len(cache) == num_layers
        assert cache.get_seq_length(0) == seq_len


# ── _resize_buffer ──────────────────────────────────────────────────────────

class TestResizeBuffer:
    """Test the buffer resize utility."""
    
    def test_grow(self):
        buf = torch.zeros(10, 4, 8)
        buf[0] = 1.0
        new_buf = _resize_buffer(buf, 20)
        assert new_buf.shape[0] == 20
        assert torch.equal(new_buf[0], buf[0])
    
    def test_no_shrink(self):
        buf = torch.zeros(20, 4, 8)
        result = _resize_buffer(buf, 10)
        assert result.shape[0] == 20  # Should not shrink
    
    def test_preserves_dtype(self):
        buf = torch.zeros(10, 4, dtype=torch.float16)
        new_buf = _resize_buffer(buf, 20)
        assert new_buf.dtype == torch.float16
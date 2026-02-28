"""
Tests for HSRCConfig: validation, defaults, derived properties, edge cases.
"""
import pytest
from hsrc.config import HSRCConfig


class TestHSRCConfigDefaults:
    """Verify all default values match spec."""
    
    def test_default_block_size(self):
        cfg = HSRCConfig()
        assert cfg.block_size == 256
    
    def test_default_boundary_size(self):
        cfg = HSRCConfig()
        assert cfg.boundary_size == 8
    
    def test_default_key_rank(self):
        cfg = HSRCConfig()
        assert cfg.key_rank == 12
    
    def test_default_value_rank(self):
        cfg = HSRCConfig()
        assert cfg.value_rank == 24
    
    def test_default_max_sparse(self):
        cfg = HSRCConfig()
        assert cfg.max_sparse_per_block == 8
    
    def test_default_sparse_threshold(self):
        cfg = HSRCConfig()
        assert cfg.sparse_threshold_multiplier == 3.0
    
    def test_default_int8(self):
        cfg = HSRCConfig()
        assert cfg.use_int8_coefficients is True
    
    def test_default_hot_buffer_extra(self):
        cfg = HSRCConfig()
        assert cfg.hot_buffer_extra == 16
    
    def test_default_store_pre_rope(self):
        cfg = HSRCConfig()
        assert cfg.store_pre_rope is True
    
    def test_default_min_seq_len(self):
        cfg = HSRCConfig()
        assert cfg.min_seq_len_to_compress == 512
    
    def test_default_verify_reconstruction(self):
        cfg = HSRCConfig()
        assert cfg.verify_reconstruction is False


class TestHSRCConfigDerivedProperties:
    """Verify derived properties compute correctly."""
    
    def test_interior_size_default(self):
        cfg = HSRCConfig()
        assert cfg.interior_size == 256 - 2 * 8  # = 240
    
    def test_interior_size_custom(self):
        cfg = HSRCConfig(block_size=128, boundary_size=4)
        assert cfg.interior_size == 120
    
    def test_hot_buffer_size_default(self):
        cfg = HSRCConfig()
        # block_size + hot_buffer_extra + boundary_size = 256 + 16 + 8 = 280
        assert cfg.hot_buffer_size == 280
    
    def test_hot_buffer_size_custom(self):
        cfg = HSRCConfig(block_size=512, hot_buffer_extra=32, boundary_size=16)
        assert cfg.hot_buffer_size == 512 + 32 + 16


class TestHSRCConfigValidation:
    """Verify __post_init__ catches invalid configurations."""
    
    def test_block_size_too_small_for_boundary(self):
        """block_size must be > 2 * boundary_size."""
        with pytest.raises(AssertionError):
            HSRCConfig(block_size=16, boundary_size=8)
    
    def test_block_size_exactly_double_boundary(self):
        """block_size == 2 * boundary_size is invalid (not strictly greater)."""
        with pytest.raises(AssertionError):
            HSRCConfig(block_size=16, boundary_size=8)
    
    def test_minimum_legal_block_size(self):
        """block_size = 2 * boundary_size + 1 should work."""
        cfg = HSRCConfig(block_size=17, boundary_size=8)
        assert cfg.interior_size == 1
    
    def test_key_rank_zero(self):
        with pytest.raises(AssertionError):
            HSRCConfig(key_rank=0)
    
    def test_value_rank_zero(self):
        with pytest.raises(AssertionError):
            HSRCConfig(value_rank=0)
    
    def test_negative_key_rank(self):
        with pytest.raises(AssertionError):
            HSRCConfig(key_rank=-1)
    
    def test_sparse_threshold_zero(self):
        with pytest.raises(AssertionError):
            HSRCConfig(sparse_threshold_multiplier=0.0)
    
    def test_sparse_threshold_negative(self):
        with pytest.raises(AssertionError):
            HSRCConfig(sparse_threshold_multiplier=-1.0)
    
    def test_max_sparse_zero_is_valid(self):
        """max_sparse_per_block=0 means no sparse corrections — valid."""
        cfg = HSRCConfig(max_sparse_per_block=0)
        assert cfg.max_sparse_per_block == 0
    
    def test_negative_max_sparse(self):
        with pytest.raises(AssertionError):
            HSRCConfig(max_sparse_per_block=-1)


class TestHSRCConfigCustom:
    """Custom configurations for different use cases."""
    
    def test_aggressive_compression(self):
        """Low rank, small blocks — maximum compression."""
        cfg = HSRCConfig(block_size=128, boundary_size=4, key_rank=4, value_rank=8)
        assert cfg.interior_size == 120
        assert cfg.block_size == 128
    
    def test_high_fidelity(self):
        """High rank, more sparse — maximum quality."""
        cfg = HSRCConfig(key_rank=32, value_rank=48, max_sparse_per_block=16)
        assert cfg.key_rank == 32
        assert cfg.value_rank == 48
    
    def test_int8_disabled(self):
        """No quantization — FP16 coefficients."""
        cfg = HSRCConfig(use_int8_coefficients=False)
        assert cfg.use_int8_coefficients is False
    
    def test_large_block_size(self):
        """512-token blocks."""
        cfg = HSRCConfig(block_size=512, boundary_size=16)
        assert cfg.interior_size == 480
        assert cfg.hot_buffer_size == 512 + 16 + 16
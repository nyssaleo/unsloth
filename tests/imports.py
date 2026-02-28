"""
Dual-path import helper for HSRC tests.

Tries to import from unsloth.hsrc (full package) first, which works on GPU systems
where unsloth can initialize. Falls back to direct hsrc imports on Mac/CPU where
unsloth.__init__ fails due to GPU requirements.

All test files should import from this module instead of directly from unsloth.hsrc
or hsrc to ensure compatibility across both environments.
"""

# Try full package path first (works on GPU/Colab)
try:
    from unsloth.hsrc.config import HSRCConfig
    from unsloth.hsrc.block import (
        CompressedBlock,
        compress_block,
        reconstruct_block_keys,
        reconstruct_block_values,
        _apply_rope_to_tensor,
    )
    from unsloth.hsrc.cache import (
        HSRCLayerCache,
        HSRCCache,
        _resize_buffer,
    )
    from unsloth.hsrc.integration import (
        hsrc_attention_forward_inference,
        create_hsrc_model_forward,
    )
    UNSLOTH_AVAILABLE = True
    
except (ImportError, RuntimeError, OSError):
    # Fallback for Mac/CPU where main unsloth package can't initialize
    from hsrc.config import HSRCConfig
    from hsrc.block import (
        CompressedBlock,
        compress_block,
        reconstruct_block_keys,
        reconstruct_block_values,
        _apply_rope_to_tensor,
    )
    from hsrc.cache import (
        HSRCLayerCache,
        HSRCCache,
        _resize_buffer,
    )
    from hsrc.integration import (
        hsrc_attention_forward_inference,
        create_hsrc_model_forward,
    )
    UNSLOTH_AVAILABLE = False

__all__ = [
    "HSRCConfig",
    "CompressedBlock",
    "compress_block",
    "reconstruct_block_keys",
    "reconstruct_block_values",
    "_apply_rope_to_tensor",
    "HSRCLayerCache",
    "HSRCCache",
    "_resize_buffer",
    "hsrc_attention_forward_inference",
    "create_hsrc_model_forward",
    "UNSLOTH_AVAILABLE",
]

"""
Shared fixtures for HSRC test suite.
"""
import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path so we can import unsloth.hsrc
repo_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, repo_root)

# For Mac/CPU testing: Add unsloth directory directly to path
# This allows importing hsrc modules without initializing the main unsloth package
# (which requires GPU via unsloth_zoo)
unsloth_dir = os.path.join(repo_root, "unsloth")
if unsloth_dir not in sys.path:
    sys.path.insert(0, unsloth_dir)


# ── Markers ──────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: long-running test")


# ── Constants ────────────────────────────────────────────────────────────────

HEAD_DIM = 128
N_KV_HEADS = 8
BLOCK_SIZE = 256
BOUNDARY_SIZE = 8
KEY_RANK = 12
VALUE_RANK = 24
INTERIOR_SIZE = BLOCK_SIZE - 2 * BOUNDARY_SIZE  # 240


# ── Synthetic data generators ────────────────────────────────────────────────

def _generate_kv_block(T, D, seed=42, smooth=True, needle_positions=None, needle_mag=8.0):
    """
    Generate synthetic KV data mimicking real LLM pre-RoPE keys.
    
    Uses temporally-correlated random data (low effective rank ~8-12)
    with optional needle tokens.
    """
    rng = np.random.RandomState(seed)
    
    if smooth:
        # Mimic real KV: temporally correlated, low intrinsic rank
        rank = 8
        temporal = rng.randn(T, rank).astype(np.float32)
        for i in range(1, T):
            temporal[i] = 0.8 * temporal[i - 1] + 0.2 * temporal[i]
        data = temporal @ (rng.randn(rank, D).astype(np.float32) * 1.5)
        data += rng.randn(T, D).astype(np.float32) * 0.3
    else:
        data = rng.randn(T, D).astype(np.float32)
    
    if needle_positions is not None:
        for p in needle_positions:
            if p < T:
                data[p] = rng.randn(D).astype(np.float32) * needle_mag
    
    return data


@pytest.fixture
def default_kv_block():
    """Default [256, 128] smooth KV block as torch tensors."""
    K_np = _generate_kv_block(BLOCK_SIZE, HEAD_DIM, seed=42, smooth=True)
    V_np = _generate_kv_block(BLOCK_SIZE, HEAD_DIM, seed=43, smooth=True)
    return torch.from_numpy(K_np), torch.from_numpy(V_np)


@pytest.fixture
def needle_kv_block():
    """KV block with needle tokens at known positions."""
    needles = [50, 120, 200]
    K_np = _generate_kv_block(BLOCK_SIZE, HEAD_DIM, seed=42, smooth=True,
                               needle_positions=needles, needle_mag=10.0)
    V_np = _generate_kv_block(BLOCK_SIZE, HEAD_DIM, seed=43, smooth=True,
                               needle_positions=needles, needle_mag=10.0)
    return torch.from_numpy(K_np), torch.from_numpy(V_np), needles


@pytest.fixture
def rope_tables():
    """RoPE cos/sin tables for 4096 positions, head_dim=128."""
    max_len = 4096
    h = HEAD_DIM // 2
    theta = 10000.0
    pos = torch.arange(max_len, dtype=torch.float64)
    freqs = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.outer(pos, freqs)
    cos = angles.cos().float()  # [max_len, h]
    sin = angles.sin().float()  # [max_len, h]
    return cos, sin
    
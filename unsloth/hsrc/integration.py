# Copyright 2026 - Holographic Inference Engine
# Integration layer between HSRC and Unsloth's inference path

"""
This module provides the modified attention forward function that integrates
HSRC into Unsloth's inference loop. It replaces 
LlamaAttention_fast_forward_inference with a version that:

1. Intercepts pre-RoPE keys (before RoPE application)
2. Stores them in the HSRC hot buffer (pre-RoPE)
3. Reconstructs compressed + hot keys with RoPE during attention
4. Falls back to standard attention when HSRC is not active

Integration is done by replacing the attention function in the model's
forward pass, NOT by monkey-patching. The replacement is clean: it handles
both HSRC and standard paths through the same function.

Usage:
    from unsloth.hsrc.integration import create_hsrc_inference_function
    
    # Get modified inference function that uses HSRC
    hsrc_inference = create_hsrc_inference_function(hsrc_cache)
    
    # Patch the model's forward
    model.forward = CausalLM_fast_forward(
        _LlamaModel_fast_forward_inference(
            attention_fast_forward_inference=hsrc_inference
        )
    )
"""

import torch
from typing import Optional, Tuple
from .cache import HSRCCache, HSRCLayerCache
from .block import _apply_rope_to_tensor


def hsrc_attention_forward_inference(
    self,
    hidden_states: torch.Tensor,
    past_key_value,
    position_ids,
    do_prefill: bool = False,
    attention_mask=None,
    hsrc_layer_cache: Optional[HSRCLayerCache] = None,
):
    """
    Modified LlamaAttention_fast_forward_inference that uses HSRC.
    
    This function follows the exact same structure as Unsloth's original,
    but intercepts the key projection BEFORE RoPE to store pre-RoPE keys
    in the HSRC cache, and uses HSRC for attention computation.
    
    When hsrc_layer_cache is None, falls back to standard behavior.
    
    Args:
        self: LlamaAttention module
        hidden_states: [bsz, 1, hidden_dim] input
        past_key_value: (K, V) tuple from prefill OR HSRCLayerCache
        position_ids: position indices
        do_prefill: True for first decode step (initializes buffers)
        attention_mask: optional attention mask for batched inference
        hsrc_layer_cache: HSRC layer cache (if active)
    """
    from math import sqrt as math_sqrt
    
    Xn = hidden_states
    bsz, _, hd = hidden_states.size()
    dtype = Xn.dtype
    device = hidden_states.device
    
    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    hidden_size = self.config.hidden_size
    attention_size = n_heads * head_dim
    
    # If no HSRC cache, delegate to standard path
    if hsrc_layer_cache is None:
        # Import the original function
        from unsloth.models.llama import LlamaAttention_fast_forward_inference
        return LlamaAttention_fast_forward_inference(
            self, hidden_states, past_key_value, position_ids,
            do_prefill=do_prefill, attention_mask=attention_mask
        )
    
    # ========== HSRC Path ==========
    
    # Get sequence lengths
    kv_seq_len = hsrc_layer_cache.total_len + 1  # +1 for the token we're about to add
    seq_len = hsrc_layer_cache.total_len  # current length before this token
    
    # Allocate temp buffers on first use (same as standard path)
    if do_prefill or not hasattr(self, 'temp_QA'):
        self.temp_QA = torch.empty(
            (2, bsz, 1, attention_size), dtype=dtype, device=device
        )
        self.temp_KV = torch.empty(
            (2, bsz, 1, n_kv_heads * head_dim), dtype=dtype, device=device
        )
        self.RH_Q = torch.empty(
            (bsz, n_heads, 1, head_dim), dtype=dtype, device=device
        )
        if attention_size != hidden_size:
            self.temp_O = torch.empty(
                (1, bsz, hidden_size), dtype=dtype, device=device
            )
        else:
            self.temp_O = self.temp_QA[1][:, :, :hidden_size]
        self.scalar = 1.0 / math_sqrt(head_dim)
        self.half_head_dim = head_dim // 2
    
    # --- Project Q, K, V ---
    # Use Unsloth's fast_linear_forward if available, otherwise standard
    try:
        from unsloth.models.llama import fast_linear_forward
        Qn = fast_linear_forward(self.q_proj, Xn, out=self.temp_QA[0])
        Kn = fast_linear_forward(self.k_proj, Xn, out=self.temp_KV[0])
        Vn = fast_linear_forward(self.v_proj, Xn, out=self.temp_KV[1])
    except (ImportError, RuntimeError, OSError):
        Qn = self.q_proj(Xn)
        Kn = self.k_proj(Xn)
        Vn = self.v_proj(Xn)
    
    Qn = Qn.view(bsz, 1, n_heads, head_dim).transpose(1, 2)
    Kn = Kn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)
    Vn = Vn.view(bsz, 1, n_kv_heads, head_dim).transpose(1, 2)
    # Shapes: Qn [bsz, n_heads, 1, head_dim], Kn/Vn [bsz, n_kv_heads, 1, head_dim]
    
    # *** HSRC INTERCEPTION POINT ***
    # Store pre-RoPE key and value into HSRC cache BEFORE applying RoPE
    K_pre_rope = Kn.squeeze(2).squeeze(0)  # [n_kv_heads, head_dim]
    V_token = Vn.squeeze(2).squeeze(0)     # [n_kv_heads, head_dim]
    hsrc_layer_cache.append_token(K_pre_rope, V_token)
    
    # --- Apply RoPE to Q only ---
    # (Keys are stored pre-RoPE; RoPE applied during reconstruction)
    self.rotary_emb.extend_rope_embedding(Vn, seq_len + 2)
    cos, sin = self.rotary_emb.get_cached(kv_seq_len, Qn.device.index)
    
    # Update HSRC's RoPE cache
    hsrc_layer_cache.set_rope_cache(cos, sin)
    
    cos_q = cos[position_ids].unsqueeze(1)
    sin_q = sin[position_ids].unsqueeze(1)
    h = self.half_head_dim
    
    # Apply RoPE to Q (in-place, matching Unsloth's convention)
    RH_Q = self.RH_Q
    RH_Q[:, :, :, :h] = Qn[:, :, :, h:]
    RH_Q[:, :, :, h:] = Qn[:, :, :, :h]
    RH_Q[:, :, :, :h].neg_()
    Qn *= cos_q
    Qn.addcmul_(RH_Q, sin_q)
    # Now Qn is post-RoPE: [bsz, n_heads, 1, head_dim]
    
    # --- Reconstruct all KV with RoPE and compute attention ---
    # Phase 1: Full reconstruction (no fused kernel yet)
    # In Phase 2, this will be a fused Triton kernel that reads compressed
    # data from HBM and reconstructs in SRAM.
    
    Kn_all = hsrc_layer_cache.reconstruct_all_keys_post_rope(cos, sin)
    # [n_kv_heads, total_len, head_dim]
    
    Vn_all = hsrc_layer_cache.reconstruct_all_values()
    # [n_kv_heads, total_len, head_dim]
    
    # Add batch dimension: [bsz, n_kv_heads, total_len, head_dim]
    Kn_all = Kn_all.unsqueeze(0).to(dtype)
    Vn_all = Vn_all.unsqueeze(0).to(dtype)
    
    # Handle grouped query attention
    cached_len = Kn_all.shape[2]
    if bsz == 1 and n_groups != 1:
        Knn = Kn_all[:, :, None, :, :].expand(
            bsz, n_kv_heads, n_groups, cached_len, head_dim
        ).reshape(bsz, n_heads, cached_len, head_dim)
        Vnn = Vn_all[:, :, None, :, :].expand(
            bsz, n_kv_heads, n_groups, cached_len, head_dim
        ).reshape(bsz, n_heads, cached_len, head_dim)
    else:
        Knn = Kn_all
        Vnn = Vn_all
    
    # --- Attention ---
    if bsz == 1:
        Qn_scaled = Qn * self.scalar
        # Need attention buffer
        if not hasattr(self, 'hsrc_attention') or self.hsrc_attention.shape[-1] < cached_len:
            self.hsrc_attention = torch.empty(
                (bsz, n_heads, 1, cached_len + 512),
                dtype=dtype, device=device
            )
        A = torch.matmul(Qn_scaled, Knn.transpose(2, 3),
                        out=self.hsrc_attention[:, :, :, :cached_len])
        A[:] = torch.nn.functional.softmax(A, dim=-1, dtype=torch.float32)
        A = torch.matmul(A, Vnn, out=Qn)
    else:
        A = torch.nn.functional.scaled_dot_product_attention(
            Qn, Knn, Vnn, attn_mask=attention_mask, is_causal=False
        )
    
    A = A.transpose(1, 2).reshape(bsz, 1, attention_size)
    try:
        from unsloth.models.llama import fast_linear_forward
        A = fast_linear_forward(self.o_proj, A, out=self.temp_O)
    except (ImportError, RuntimeError, OSError):
        A = self.o_proj(A)
    
    # Return format matching Unsloth: (output, (K_cache, V_cache))
    # For HSRC, we return dummy tensors for the cache tuple since
    # the real cache is in hsrc_layer_cache. The dummy must have the
    # right shape for the next iteration's seq_len calculation.
    dummy_shape = (bsz, n_kv_heads, hsrc_layer_cache.total_len, head_dim)
    # Use a view of zeros to avoid allocation
    if not hasattr(self, '_hsrc_dummy_K') or self._hsrc_dummy_K.shape != dummy_shape:
        self._hsrc_dummy_K = torch.zeros(dummy_shape, dtype=dtype, device=device)
        self._hsrc_dummy_V = torch.zeros(dummy_shape, dtype=dtype, device=device)
    
    return A, (self._hsrc_dummy_K, self._hsrc_dummy_V)


def create_hsrc_model_forward(
    hsrc_cache: HSRCCache,
    original_forward_fn=None,
):
    """
    Create a model forward function that threads HSRC cache through layers.
    
    This replaces the inner loop of LlamaModel_fast_forward_inference to
    pass the per-layer HSRC cache to each attention call.
    
    Args:
        hsrc_cache: The full HSRC cache object
        original_forward_fn: The original forward function (for non-HSRC fallback)
    
    Returns:
        A forward function compatible with CausalLM_fast_forward
    """
    
    def hsrc_model_forward_inference(
        self,
        input_ids,
        past_key_values,
        position_ids,
        attention_mask=None,
        **kwargs,
    ):
        """Modified LlamaModel forward that uses HSRC cache."""
        from unsloth.models.llama import (
            fast_rms_layernorm_inference,
            fast_swiglu_inference,
            fast_linear_forward,
            _get_dtype,
            dtype_from_config,
            BaseModelOutputWithPast,
            DEVICE_TYPE_TORCH,
            DEVICE_COUNT,
        )
        try:
            from unsloth.models.llama import move_to_device
        except (ImportError, RuntimeError, OSError):
            def move_to_device(idx, *args):
                return args
        
        input_ids = input_ids[:, :self.max_seq_length]
        bsz, q_len = input_ids.shape
        hd = self.config.hidden_size
        mlp_size = self.config.intermediate_size
        
        X = self.model.embed_tokens(input_ids)
        X = X.to(_get_dtype(dtype_from_config(self.config)))
        bsz, q_len, hd = X.shape
        assert q_len == 1, "HSRC decode requires q_len=1"
        
        device = X.device
        residual = torch.empty(
            (bsz, q_len, hd), dtype=torch.float32, device=device
        )
        _XX = torch.empty(
            (2, bsz, q_len, hd), dtype=torch.float32, device=device
        )
        XX, XX2 = _XX[0], _XX[1]
        variance = torch.empty(
            (bsz, q_len, 1), dtype=torch.float32, device=device
        )
        temp_mlp = torch.empty(
            (2, bsz, 1, mlp_size), dtype=X.dtype, device=device
        )
        temp_gates = (temp_mlp[0],)
        temp_ups = (temp_mlp[1],)
        
        if bsz != 1:
            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_causal_attention_mask_for_sdpa,
            )
            seq_len = hsrc_cache.get_seq_length()
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask, (bsz, q_len), X, seq_len,
                sliding_window=getattr(self.config, "sliding_window", None),
            )
        else:
            attention_mask = None
        
        next_decoder_cache = []
        
        for idx, decoder_layer in enumerate(self.model.layers):
            device_index = getattr(decoder_layer, "_per_layer_device_index", 0)
            X, residual, position_ids = move_to_device(
                device_index, X, residual, position_ids
            )
            residual.copy_(X)
            X = fast_rms_layernorm_inference(
                decoder_layer.input_layernorm, X,
                XX=XX, XX2=XX2, variance=variance,
            )
            
            # Use HSRC attention
            X, present_key_value = hsrc_attention_forward_inference(
                decoder_layer.self_attn,
                hidden_states=X,
                past_key_value=past_key_values[idx] if isinstance(past_key_values, list) else None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                do_prefill=not hasattr(decoder_layer.self_attn, 'temp_QA'),
                hsrc_layer_cache=hsrc_cache[idx],
            )
            X += residual
            
            residual.copy_(X)
            X = fast_rms_layernorm_inference(
                decoder_layer.post_attention_layernorm, X,
                XX=XX, XX2=XX2, variance=variance,
            )
            X = fast_swiglu_inference(
                decoder_layer.mlp, X,
                temp_gate=temp_gates[min(device_index, len(temp_gates)-1)],
                temp_up=temp_ups[min(device_index, len(temp_ups)-1)],
            )
            X += residual
            
            next_decoder_cache.append(present_key_value)
        
        X = fast_rms_layernorm_inference(
            self.model.norm, X,
            XX=XX, XX2=XX2, variance=variance,
        )
        
        return BaseModelOutputWithPast(
            last_hidden_state=X,
            past_key_values=next_decoder_cache,
            hidden_states=[],
            attentions=[],
        )
    
    return hsrc_model_forward_inference
"""
Custom Attention Layers for SpatialLM

Extends Qwen2Attention and LlamaAttention to support 3D RoPE for point cloud tokens.

Key modification:
- Apply 3D RoPE to Q/K AFTER head splitting (on head_dim, per-head)
- All attention heads share the same 3D spatial encoding pattern
- This works with GQA (Grouped Query Attention) architectures
- Standard 1D RoPE still applies to text tokens as usual
"""

import math
import os
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Global verbose flag for debug printouts
VERBOSE_3D_PE = os.environ.get("SPATIALLM_VERBOSE", "0").lower() in ("1", "true", "yes")

try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        apply_rotary_pos_emb,
        repeat_kv,
    )
    _QWEN2_AVAILABLE = True
except ImportError:
    _QWEN2_AVAILABLE = False

try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
        repeat_kv as llama_repeat_kv,
    )
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False

try:
    from spatiallm.model.volumetric_pe import apply_rotary_pos_emb_3d
    _3D_ROPE_AVAILABLE = True
except ImportError:
    _3D_ROPE_AVAILABLE = False


# ===============================================================================
# Qwen2 Custom Attention
# ===============================================================================

if _QWEN2_AVAILABLE:
    class SpatialQwen2Attention(Qwen2Attention):
        """
        Modified Qwen2Attention to support 3D RoPE for point cloud tokens.
        
        === SPATIAL-LM MODIFICATIONS ===
        1. Apply 3D RoPE to Q/K AFTER head splitting (per-head, all heads share pattern)
        2. Point token positions and 3D RoPE data passed via model attributes
        
        All other logic identical to parent Qwen2Attention.
        """
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """
            === MODIFICATIONS FROM PARENT ===
            Lines marked with "=== SPATIAL-LM MODIFICATION ===" are the ONLY changes.
            All other code is identical to Qwen2Attention.forward().
            """
            
            bsz, q_len, _ = hidden_states.size()
            
            # === UNCHANGED: Standard Q/K/V projection ===
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            # ============================================
            
            # === SPATIAL-LM MODIFICATION: Check if we need to apply 3D RoPE ===
            # 3D RoPE is only applied during prefill stage
            
            # Detect if this is prefill stage: either past_key_value is None,
            # or if it's a Cache object, check if it's empty for this layer
            is_prefill = (past_key_value is None)
            if past_key_value is not None and hasattr(past_key_value, 'get_seq_length'):
                is_prefill = (past_key_value.get_seq_length(self.layer_idx) == 0)
            
            has_3d_rope = (hasattr(self, '_spatial_3d_rope_data') and 
                          self._spatial_3d_rope_data is not None and
                          is_prefill)  # Only during prefill
            # ==================================================================
            
            # === UNCHANGED: Reshape to multi-head format ===
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # ===============================================
            
            # === SPATIAL-LM MODIFICATION: Apply 3D RoPE to point tokens (AFTER head splitting) ===
            point_token_positions = []
            if has_3d_rope:
                if VERBOSE_3D_PE:
                    print("\n=== [Qwen2 3D_RoPE Sanity Check] ===")
                    print(f"[CHECKPOINT 1] After Q/K/V projection and head reshape:")
                    print(f"  query_states.shape: {query_states.shape}  (expect: [bsz, num_heads, seq_len, head_dim])")
                    print(f"  key_states.shape: {key_states.shape}  (expect: [bsz, num_kv_heads, seq_len, head_dim])")
                
                cos_3d, sin_3d, point_token_positions = self._spatial_3d_rope_data
                
                if VERBOSE_3D_PE:
                    # # === DEBUG: Compare 3D RoPE cos/sin with 1D RoPE ===
                    # query_states_dbg = query_states.clone()
                    # key_states_dbg = key_states.clone()
                    # cos_dbg, sin_dbg = position_embeddings
                    # query_states_dbg, key_states_dbg = apply_rotary_pos_emb(query_states_dbg, key_states_dbg, cos_dbg, sin_dbg)
                    
                    pt_start_pos = point_token_positions[0][0]  # First point token position
                    cos, sin = position_embeddings  # (1, seq_len, head_dim)
                    cos_1d_pts = cos[0, pt_start_pos, :]
                    print('\n=== [DEBUG] Frequency Comparison ===')
                    print(f'cos_3d.shape: {cos_3d.shape}, dtype: {cos_3d.dtype}')
                    print(f'cos_3d[0,:10]: {cos_3d[0,:10]}')
                    print(f'cos_1d_pts.shape: {cos_1d_pts.shape}, dtype: {cos_1d_pts.dtype}')
                    print(f'cos_1d_pts[:10]: {cos_1d_pts[:10]}')
                    print(f'Match (atol=1e-3)? {torch.allclose(cos_3d[0], cos_1d_pts, atol=1e-3)}')
                    print('=' * 40 + '\n')
                # ===================================================
                
                if VERBOSE_3D_PE:
                    print(f"[CHECKPOINT 2] 3D RoPE data:")
                    print(f"  cos_3d.shape: {cos_3d.shape}  (expect: [num_patches, head_dim])")
                    print(f"  sin_3d.shape: {sin_3d.shape}")
                    print(f"  point_token_positions: {point_token_positions}")
                
                # Apply 3D RoPE to point cloud tokens (per-head, all heads share same pattern)
                for batch_idx, (pt_start, pt_end) in enumerate(point_token_positions):
                    if batch_idx < bsz and pt_start < q_len and pt_end <= q_len:
                        if VERBOSE_3D_PE:
                            print(f"[CHECKPOINT 3] Processing batch {batch_idx}, point tokens [{pt_start}:{pt_end}]:")
                        
                        # Extract point token Q/K: (bsz, num_heads, N_pt, head_dim)
                        pt_q = query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :]
                        pt_k = key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :]
                        if VERBOSE_3D_PE:
                            print(f"  pt_q.shape: {pt_q.shape}  (expect: [1, num_heads, {pt_end-pt_start}, head_dim])")
                            print(f"  pt_k.shape: {pt_k.shape}  (expect: [1, num_kv_heads, {pt_end-pt_start}, head_dim])")
                        
                        # Apply 3D RoPE (per-head, AFTER head split)
                        # All heads share the same 3D spatial encoding pattern
                        pt_q_rotated, pt_k_rotated = apply_rotary_pos_emb_3d(
                            pt_q, pt_k, cos_3d, sin_3d
                        )
                        if VERBOSE_3D_PE:
                            print(f"[CHECKPOINT 4] After apply_rotary_pos_emb_3d:")
                            print(f"  pt_q_rotated.shape: {pt_q_rotated.shape}  (expect: same as pt_q)")
                            print(f"  pt_k_rotated.shape: {pt_k_rotated.shape}")
                        
                        # Sanity check: no broadcasting happened
                        assert pt_q_rotated.shape == pt_q.shape, f"Shape mismatch! {pt_q_rotated.shape} != {pt_q.shape}"
                        assert pt_k_rotated.shape == pt_k.shape, f"Shape mismatch! {pt_k_rotated.shape} != {pt_k.shape}"
                        
                        # Put back
                        query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = pt_q_rotated
                        key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = pt_k_rotated
                        if VERBOSE_3D_PE:
                            print(f"  ✓ Point tokens updated in query_states/key_states")
            # =====================================================================
            
            if has_3d_rope and VERBOSE_3D_PE:
                print(f"[CHECKPOINT 5] After 3D RoPE:")
                print(f"  query_states.shape: {query_states.shape}  (expect: [bsz, num_heads, seq_len, head_dim])")
                print(f"  key_states.shape: {key_states.shape}")
            
            # === SPATIAL-LM MODIFICATION: Apply 1D RoPE, then restore 3D RoPE for point tokens ===
            # Simpler approach: apply 1D RoPE to all tokens, then overwrite point tokens with 3D RoPE
            cos, sin = position_embeddings
            
            if has_3d_rope and len(point_token_positions) > 0:
                if VERBOSE_3D_PE:
                    print(f"[CHECKPOINT 6] 1D RoPE data:")
                    print(f"  cos.shape: {cos.shape}")
                    print(f"  sin.shape: {sin.shape}")
                
                # Save the point cloud Q/K with 3D RoPE applied (per-head)
                saved_point_qk = {}
                for batch_idx, (pt_start, pt_end) in enumerate(point_token_positions):
                    if batch_idx < bsz and pt_start < q_len and pt_end <= q_len:
                        # Save Q/K for point tokens (they already have 3D RoPE applied)
                        saved_q = query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :].clone()
                        saved_k = key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :].clone()
                        saved_point_qk[batch_idx] = (saved_q, saved_k)
                        if VERBOSE_3D_PE:
                            print(f"[CHECKPOINT 7] Saved point tokens for batch {batch_idx}:")
                            print(f"  saved_q.shape: {saved_q.shape}  (expect: [1, num_heads, {pt_end-pt_start}, head_dim])")
                            print(f"  saved_k.shape: {saved_k.shape}")
                
                # Apply standard 1D RoPE to ALL tokens
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
                if VERBOSE_3D_PE:
                    print(f"[CHECKPOINT 8] After apply_rotary_pos_emb (1D RoPE to ALL):")
                    print(f"  query_states.shape: {query_states.shape}  (expect: unchanged)")
                    print(f"  key_states.shape: {key_states.shape}")
                
                # Restore point tokens with 3D RoPE (overwrite the 1D RoPE we just applied)
                for batch_idx, (saved_q, saved_k) in saved_point_qk.items():
                    pt_start, pt_end = point_token_positions[batch_idx]
                    
                    # # # SANITY CHECK: Compare 1D-rotated vs 3D-rotated Q/K
                    # if VERBOSE_3D_PE:
                    #     q_after_1d = query_states_dbg[batch_idx:batch_idx+1, :, pt_start:pt_end, :]
                    #     k_after_1d = key_states_dbg[batch_idx:batch_idx+1, :, pt_start:pt_end, :]
                        
                    #     q_diff = (saved_q - q_after_1d).abs().max().item()
                    #     k_diff = (saved_k - k_after_1d).abs().max().item()
                        
                    #     print(f"[SANITY CHECK] Batch {batch_idx}, point tokens [{pt_start}:{pt_end}]:")
                    #     print(f"  Max |3D_RoPE_Q - 1D_RoPE_Q|: {q_diff:.6e}")
                    #     print(f"  Max |3D_RoPE_K - 1D_RoPE_K|: {k_diff:.6e}")
                        
                    #     threshold = 1e-5
                    #     if q_diff < threshold and k_diff < threshold:
                    #         print(f"  ✅ IDENTICAL (within {threshold}) - ratio_1d=1.0 works as expected!")
                    #     else:
                    #         print(f"  ⚠️  DIFFERENT - 3D RoPE is actually applying different rotations!")
                    
                    query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = saved_q
                    key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = saved_k
                    if VERBOSE_3D_PE:
                        print(f"[CHECKPOINT 9] Restored 3D RoPE for batch {batch_idx}, positions [{pt_start}:{pt_end}]")
                
                if VERBOSE_3D_PE:
                    print("✓ 3D RoPE preserved for point tokens, 1D RoPE applied to text tokens")
                    print("=" * 60 + "\n")
            else:
                # No 3D RoPE: apply standard 1D RoPE to all tokens as usual
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            # ===================================================================================
            
            # === UNCHANGED: KV cache handling ===
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # ====================================
            
            # === UNCHANGED: GQA - repeat k/v heads if needed ===
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            # ==================================================
            
            # === UNCHANGED: Compute attention ===
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            # ===================================
            
            if not output_attentions:
                attn_weights = None
            
            return attn_output, attn_weights, past_key_value


# ===============================================================================
# Llama Custom Attention
# ===============================================================================

if _LLAMA_AVAILABLE:
    class SpatialLlamaAttention(LlamaAttention):
        """
        Modified LlamaAttention to support 3D RoPE for point cloud tokens.
        
        === SPATIAL-LM MODIFICATIONS ===
        1. Apply 3D RoPE to Q/K AFTER head splitting (per-head, all heads share pattern)
        2. Point token positions and 3D RoPE data passed via model attributes
        
        All other logic identical to parent LlamaAttention.
        """
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """
            === MODIFICATIONS FROM PARENT ===
            Lines marked with "=== SPATIAL-LM MODIFICATION ===" are the ONLY changes.
            All other code is identical to LlamaAttention.forward().
            """
            
            bsz, q_len, _ = hidden_states.size()
            
            # === UNCHANGED: Standard Q/K/V projection ===
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            # ============================================
            
            # === SPATIAL-LM MODIFICATION: Check if we need to apply 3D RoPE ===
            # 3D RoPE is only applied during prefill stage
            
            # Detect if this is prefill stage: either past_key_value is None,
            # or if it's a Cache object, check if it's empty for this layer
            is_prefill = (past_key_value is None)
            if past_key_value is not None and hasattr(past_key_value, 'get_seq_length'):
                is_prefill = (past_key_value.get_seq_length(self.layer_idx) == 0)
            
            has_3d_rope = (hasattr(self, '_spatial_3d_rope_data') and 
                          self._spatial_3d_rope_data is not None and
                          is_prefill)  # Only during prefill
            # ==================================================================
            
            # === UNCHANGED: Reshape to multi-head format ===
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # ===============================================
            
            # === SPATIAL-LM MODIFICATION: Apply 3D RoPE to point tokens (AFTER head splitting) ===
            point_token_positions = []
            if has_3d_rope:
                if VERBOSE_3D_PE:
                    print("\n=== [Llama 3D_RoPE Sanity Check] ===")
                    print(f"[CHECKPOINT 1] After Q/K/V projection and head reshape:")
                    print(f"  query_states.shape: {query_states.shape}  (expect: [bsz, num_heads, seq_len, head_dim])")
                    print(f"  key_states.shape: {key_states.shape}  (expect: [bsz, num_kv_heads, seq_len, head_dim])")
                
                cos_3d, sin_3d, point_token_positions = self._spatial_3d_rope_data       
                if VERBOSE_3D_PE:
                    print(f"[CHECKPOINT 2] 3D RoPE data:")
                    print(f"  cos_3d.shape: {cos_3d.shape}  (expect: [num_patches, head_dim])")
                    print(f"  sin_3d.shape: {sin_3d.shape}")
                    print(f"  point_token_positions: {point_token_positions}")
                
                # Apply 3D RoPE to point cloud tokens (per-head, all heads share same pattern)
                for batch_idx, (pt_start, pt_end) in enumerate(point_token_positions):
                    if batch_idx < bsz and pt_start < q_len and pt_end <= q_len:
                        if VERBOSE_3D_PE:
                            print(f"[CHECKPOINT 3] Processing batch {batch_idx}, point tokens [{pt_start}:{pt_end}]:")
                        
                        # Extract point token Q/K: (bsz, num_heads, N_pt, head_dim)
                        pt_q = query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :]
                        pt_k = key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :]
                        if VERBOSE_3D_PE:
                            print(f"  pt_q.shape: {pt_q.shape}  (expect: [1, num_heads, {pt_end-pt_start}, head_dim])")
                            print(f"  pt_k.shape: {pt_k.shape}  (expect: [1, num_kv_heads, {pt_end-pt_start}, head_dim])")
                        
                        # Apply 3D RoPE (per-head, AFTER head split)
                        # All heads share the same 3D spatial encoding pattern
                        pt_q_rotated, pt_k_rotated = apply_rotary_pos_emb_3d(
                            pt_q, pt_k, cos_3d, sin_3d
                        )
                        if VERBOSE_3D_PE:
                            print(f"[CHECKPOINT 4] After apply_rotary_pos_emb_3d:")
                            print(f"  pt_q_rotated.shape: {pt_q_rotated.shape}  (expect: same as pt_q)")
                            print(f"  pt_k_rotated.shape: {pt_k_rotated.shape}")
                        
                        # Sanity check: no broadcasting happened
                        assert pt_q_rotated.shape == pt_q.shape, f"Shape mismatch! {pt_q_rotated.shape} != {pt_q.shape}"
                        assert pt_k_rotated.shape == pt_k.shape, f"Shape mismatch! {pt_k_rotated.shape} != {pt_k.shape}"
                        
                        # Put back
                        query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = pt_q_rotated
                        key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = pt_k_rotated
                        if VERBOSE_3D_PE:
                            print(f"  ✓ Point tokens updated in query_states/key_states")
            # =====================================================================
            
            if has_3d_rope and VERBOSE_3D_PE:
                print(f"[CHECKPOINT 5] After 3D RoPE:")
                print(f"  query_states.shape: {query_states.shape}  (expect: [bsz, num_heads, seq_len, head_dim])")
                print(f"  key_states.shape: {key_states.shape}")
            
            # === SPATIAL-LM MODIFICATION: Apply 1D RoPE, then restore 3D RoPE for point tokens ===
            # Simpler approach: apply 1D RoPE to all tokens, then overwrite point tokens with 3D RoPE
            cos, sin = position_embeddings
            
            if has_3d_rope and len(point_token_positions) > 0:
                if VERBOSE_3D_PE:
                    print(f"[CHECKPOINT 6] 1D RoPE data:")
                    print(f"  cos.shape: {cos.shape}")
                    print(f"  sin.shape: {sin.shape}")
                
                # Save the point cloud Q/K with 3D RoPE applied (per-head)
                saved_point_qk = {}
                for batch_idx, (pt_start, pt_end) in enumerate(point_token_positions):
                    if batch_idx < bsz and pt_start < q_len and pt_end <= q_len:
                        # Save Q/K for point tokens (they already have 3D RoPE applied)
                        saved_q = query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :].clone()
                        saved_k = key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :].clone()
                        saved_point_qk[batch_idx] = (saved_q, saved_k)
                        if VERBOSE_3D_PE:
                            print(f"[CHECKPOINT 7] Saved point tokens for batch {batch_idx}:")
                            print(f"  saved_q.shape: {saved_q.shape}  (expect: [1, num_heads, {pt_end-pt_start}, head_dim])")
                            print(f"  saved_k.shape: {saved_k.shape}")
                
                # Apply standard 1D RoPE to ALL tokens
                query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
                if VERBOSE_3D_PE:
                    print(f"[CHECKPOINT 8] After apply_rotary_pos_emb (1D RoPE to ALL):")
                    print(f"  query_states.shape: {query_states.shape}  (expect: unchanged)")
                    print(f"  key_states.shape: {key_states.shape}")
                
                # Restore point tokens with 3D RoPE (overwrite the 1D RoPE we just applied)
                for batch_idx, (saved_q, saved_k) in saved_point_qk.items():
                    pt_start, pt_end = point_token_positions[batch_idx]
                    query_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = saved_q
                    key_states[batch_idx:batch_idx+1, :, pt_start:pt_end, :] = saved_k
                    if VERBOSE_3D_PE:
                        print(f"[CHECKPOINT 9] Restored 3D RoPE for batch {batch_idx}, positions [{pt_start}:{pt_end}]")
                
                if VERBOSE_3D_PE:
                    print("✓ 3D RoPE preserved for point tokens, 1D RoPE applied to text tokens")
                    print("=" * 60 + "\n")
            else:
                # No 3D RoPE: apply standard 1D RoPE to all tokens as usual
                query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
            # ===================================================================================
            
            # === UNCHANGED: KV cache handling ===
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # ====================================
            
            # === UNCHANGED: GQA - repeat k/v heads if needed ===
            key_states = llama_repeat_kv(key_states, self.num_key_value_groups)
            value_states = llama_repeat_kv(value_states, self.num_key_value_groups)
            # ==================================================
            
            # === UNCHANGED: Compute attention ===
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            # ===================================
            
            if not output_attentions:
                attn_weights = None
            
            return attn_output, attn_weights, past_key_value

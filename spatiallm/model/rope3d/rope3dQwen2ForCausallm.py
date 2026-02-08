# JJ: MixedRoPE3D implementation for Qwen2
# This replaces 1D RoPE with 3D RoPE for point cloud tokens

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import Qwen2ForCausalLM, Qwen2Model, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2MLP,
)
from transformers.cache_utils import Cache

# Import our 3D RoPE utilities
from spatiallm.model.rope3d.mixed_rope3d_sa import (
    init_3d_freqs_unified,
    normalize_point_coords_3d,
    compute_mixed_cis_3d,
    compute_axial_cis_3d,
    apply_rotary_emb as apply_rotary_emb_3d,
)
from functools import partial


class MixedRoPE3DQwen2Attention(Qwen2Attention):
    """
    Qwen2Attention with MixedRoPE3D for point cloud tokens
    
    Key differences from standard Qwen2Attention:
    1. Accepts point_coords and point_token_mask
    2. Applies 3D RoPE to point tokens
    3. Applies standard 1D RoPE to text tokens
    """
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # JJ: 3D RoPE configuration (access after super().__init__() so attributes exist)
        # Note: Qwen2Attention uses config.num_attention_heads, not self.num_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.rope_theta = config.rope_theta  # For text tokens (1D RoPE)
        self.rope_theta_3d = getattr(config, 'rope_theta_3d', 10000.0)  # For point cloud 3D RoPE
        self.rope_mixed = getattr(config, 'rope_mixed', True)
        # self.rope_mixed = getattr(config, 'rope_mixed', False)
        self.norm_strategy = getattr(config, 'norm_strategy', 'virtual_resolution')
        self.virtual_resolution = getattr(config, 'virtual_resolution', 1.0)
        self.rope_mixed_learn_per_axis = getattr(config, 'rope_mixed_learn_per_axis', False)
        self.mixedRoPE_3d_learned_axial_mixing_weight = getattr(config, 'mixedRoPE_3d_learned_axial_mixing_weight', True)
        
        # JJ: Spatial-temporal separation strategy
        self.spatial_temporal_separate_strategy = getattr(config, 'spatial_temporal_separate_strategy', 'half_spatial_half_temp')
        self.mixed_rope_spatial_temporal_interleaved = getattr(config, 'mixed_rope_spatial_temporal_interleaved', True)
        
        # JJ: Self-adapted drift for normed point coordinates
        self.self_adapted_drift_normed_point_coords = getattr(config, 'self_adapted_drift_normed_point_coords', True)
        # JJ: Drift mode - how to compute the drift value
        # - 'anchor_wrt_avg_temporal': use len(point_indices) / 2 (same drift for all points)
        # - 'anchor_wrt_pointwise_temporal': use range(point_indices) (different drift per point)
        self.self_adapted_drift_mode = getattr(config, 'self_adapted_drift_mode', 'anchor_wrt_avg_temporal')
        
        # JJ: Calculate split ratio based on strategy
        if self.spatial_temporal_separate_strategy == 'half_spatial_half_temp':
            # Point tokens: upper half for 3D RoPE (spatial), lower half for 1D RoPE (temporal)
            # self.spatial_ratio = 0.0 # endless repeat digits
            # self.spatial_ratio = 0.25  # endless repeat digits
            self.spatial_ratio = 0.5  # endless repeat digits
            # self.spatial_ratio = 0.75  # varooes a bit in the begining, then endless
            # self.spatial_ratio = 1.0 #dless repeat digits
            self.temporal_ratio = (1 - self.spatial_ratio)  # we force the later part for temporal
        else:
            raise ValueError(f"Unknown spatial_temporal_separate_strategy: {self.spatial_temporal_separate_strategy}")
        
        # Calculate split dimensions
        self.spatial_dim = int(self.head_dim * self.spatial_ratio)  # Dimensions for 3D RoPE
        self.temporal_dim = self.head_dim - self.spatial_dim  # Remaining for 1D RoPE
        
        # JJ: Compute interleaved indices if enabled
        if self.mixed_rope_spatial_temporal_interleaved:
            self.spatial_indices, self.temporal_indices = self._compute_interleaved_indices(
                self.spatial_dim, self.temporal_dim, self.head_dim
            )
        else:
            # Default: contiguous blocks [spatial | temporal]
            self.spatial_indices = None
            self.temporal_indices = None
        
        # Initialize 3D RoPE parameters (for spatial dimensions)
        # JJ: For GQA, Q and K/V have different number of heads, so we need separate freqs
        if self.rope_mixed and self.spatial_dim > 0:
            self.compute_cis_3d = partial(compute_mixed_cis_3d, num_heads=self.num_heads)
            
            # Initialize frequencies for Q (num_attention_heads)
            freqs_base_3d_q = init_3d_freqs_unified(
                dim=self.spatial_dim,
                num_heads=self.num_heads,  # 14 for Qwen2-0.5B
                theta=self.rope_theta_3d,
                rotate=True
            )  # [num_heads, spatial_dim//2]
            
            # Initialize frequencies for K/V (num_key_value_heads) 
            # JJ: For GQA, K/V have fewer heads
            freqs_base_3d_kv = init_3d_freqs_unified(
                dim=self.spatial_dim,
                num_heads=self.num_key_value_heads,  # 2 for Qwen2-0.5B
                theta=self.rope_theta_3d,
                rotate=True
            )  # [num_key_value_heads, spatial_dim//2]
            
            if self.rope_mixed_learn_per_axis:
                # Q frequencies: [3, num_heads, spatial_dim//2]
                freqs_q = freqs_base_3d_q.unsqueeze(0).repeat(3, 1, 1)
                freqs_q = freqs_q + torch.randn_like(freqs_q) * 0.01
                freqs_q = freqs_q.view(3, -1)  # [3, num_heads * (spatial_dim//2)]
                self.freqs_3d_q = nn.Parameter(freqs_q, requires_grad=True)
                
                # K/V frequencies: [3, num_key_value_heads, spatial_dim//2]
                freqs_kv = freqs_base_3d_kv.unsqueeze(0).repeat(3, 1, 1)
                freqs_kv = freqs_kv + torch.randn_like(freqs_kv) * 0.01
                freqs_kv = freqs_kv.view(3, -1)  # [3, num_key_value_heads * (spatial_dim//2)]
                self.freqs_3d_kv = nn.Parameter(freqs_kv, requires_grad=True)
            else:
                # Shared frequencies across x/y/z
                freqs_shared_q = freqs_base_3d_q.view(1, -1)
                self.freqs_3d_q = nn.Parameter(freqs_shared_q, requires_grad=True)
                
                freqs_shared_kv = freqs_base_3d_kv.view(1, -1)
                self.freqs_3d_kv = nn.Parameter(freqs_shared_kv, requires_grad=True)
            
            # Initialize axial mixing weights (for spatial dimensions only)
            # JJ: Also separate for Q and K/V
            if self.mixedRoPE_3d_learned_axial_mixing_weight:
                num_freq_bins_3d = self.spatial_dim // 2
                # Q axial weights
                axial_weights_q = torch.ones(self.num_heads, num_freq_bins_3d, 3) + torch.randn(self.num_heads, num_freq_bins_3d, 3) * 0.01
                self.axial_weights_3d_q = nn.Parameter(axial_weights_q, requires_grad=True)
                # K/V axial weights
                axial_weights_kv = torch.ones(self.num_key_value_heads, num_freq_bins_3d, 3) + torch.randn(self.num_key_value_heads, num_freq_bins_3d, 3) * 0.01
                self.axial_weights_3d_kv = nn.Parameter(axial_weights_kv, requires_grad=True)
            else:
                self.axial_weights_3d_q = None
                self.axial_weights_3d_kv = None
        else:
            # Fallback for axial or no spatial encoding
            if self.spatial_dim > 0:
                # JJ: Separate compute_cis functions for Q and K/V in axial mode
                self.compute_cis_3d_q = partial(compute_axial_cis_3d, dim=self.spatial_dim, theta=self.rope_theta_3d)
                self.compute_cis_3d_kv = partial(compute_axial_cis_3d, dim=self.spatial_dim, theta=self.rope_theta_3d)
            else:
                self.compute_cis_3d_q = None
                self.compute_cis_3d_kv = None
            self.freqs_3d_q = None
            self.freqs_3d_kv = None
            self.axial_weights_3d_q = None
            self.axial_weights_3d_kv = None
    
    def _compute_interleaved_indices(self, spatial_dim: int, temporal_dim: int, head_dim: int):
        """
        Compute interleaved indices for spatial and temporal dimensions.
        
        Example: spatial=8, temporal=24, head_dim=32
        - Ratio: 8:24 = 1:3
        - Pattern: 'ttts' repeated 8 times
        - spatial_indices: [3, 7, 11, 15, 19, 23, 27, 31]
        - temporal_indices: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, ...]
        
        Args:
            spatial_dim: Number of spatial dimensions
            temporal_dim: Number of temporal dimensions
            head_dim: Total head dimension (must equal spatial_dim + temporal_dim)
        
        Returns:
            spatial_indices: torch.Tensor of shape [spatial_dim]
            temporal_indices: torch.Tensor of shape [temporal_dim]
        """
        import math
        
        # Validate
        assert spatial_dim + temporal_dim == head_dim, \
            f"spatial_dim ({spatial_dim}) + temporal_dim ({temporal_dim}) must equal head_dim ({head_dim})"
        
        if spatial_dim == 0 or temporal_dim == 0:
            # No interleaving needed
            return (
                torch.arange(spatial_dim, dtype=torch.long) if spatial_dim > 0 else torch.tensor([], dtype=torch.long),
                torch.arange(temporal_dim, dtype=torch.long) if temporal_dim > 0 else torch.tensor([], dtype=torch.long)
            )
        
        # Compute GCD to find the pattern unit
        gcd = math.gcd(spatial_dim, temporal_dim)
        s_unit = spatial_dim // gcd  # Number of 's' per unit
        t_unit = temporal_dim // gcd  # Number of 't' per unit
        
        # Check if the ratio allows clean interleaving
        # For interleaving to work well, we want small unit sizes (e.g., 1:3, 1:1, 2:3)
        unit_size = s_unit + t_unit
        if unit_size > 8:  # Arbitrary threshold for "reasonable" interleaving
            raise ValueError(
                f"Spatial-temporal ratio {spatial_dim}:{temporal_dim} = {s_unit}:{t_unit} "
                f"results in a unit size of {unit_size}, which is too large for clean interleaving. "
                f"Please choose a ratio with smaller units (e.g., 1:3, 1:1, 2:3, etc.)"
            )
        
        # Generate the pattern: repeat 't' t_unit times, then 's' s_unit times
        # Pattern: [t, t, ..., t (t_unit times), s, s, ..., s (s_unit times)]
        spatial_indices = []
        temporal_indices = []
        
        for i in range(gcd):  # Repeat the pattern 'gcd' times
            base_idx = i * unit_size
            # Add temporal indices first (t_unit times)
            for j in range(t_unit):
                temporal_indices.append(base_idx + j)
            # Then add spatial indices (s_unit times)
            for j in range(s_unit):
                spatial_indices.append(base_idx + t_unit + j)
        
        return (
            torch.tensor(spatial_indices, dtype=torch.long),
            torch.tensor(temporal_indices, dtype=torch.long)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            point_coords: [N_point, 3] or [B, N_point, 3] point cloud coordinates
            point_token_mask: [B, seq_len] boolean mask, True for point tokens
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Standard Q, K, V projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # JJ: Apply RoPE - mixed 1D for text, 3D for points
        if point_coords is not None and point_token_mask is not None:
            # Apply 3D RoPE to point tokens, 1D RoPE to text tokens
            query_states, key_states = self._apply_mixed_rope(
                query_states, key_states, 
                position_embeddings, point_coords, point_token_mask
            )
        else:
            # Standard 1D RoPE for all tokens
            if position_embeddings is not None:
                cos, sin = position_embeddings
                # Apply standard RoPE using rotate_half helper
                query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
                key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        
        # Handle past_key_values caching
        if past_key_values is not None:
            cache_kwargs = {"sin": None, "cos": None, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat k/v heads if num_key_value_groups > 1
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights if output_attentions else None, past_key_values
    
    def _apply_mixed_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        point_coords: torch.Tensor,
        point_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hybrid RoPE to point tokens based on spatial_temporal_separate_strategy
        Apply 1D RoPE to text tokens: full head_dim
        
        KEY DESIGN (for 'half_spatial_half_temp' strategy):
        - Point tokens: 
            * First spatial_dim dimensions: 3D RoPE based on (x,y,z) coordinates
            * Remaining temporal_dim dimensions: 1D RoPE based on temporal position
        - Text tokens:
            * Full head_dim: 1D RoPE based on temporal position
        
        TODO: Support other strategies (full_spatial, full_temporal, custom ratios)
        
        Args:
            query_states: [B, num_heads, seq_len, head_dim]
            key_states: [B, num_key_value_heads, seq_len, head_dim]
            position_embeddings: (cos, sin) for 1D RoPE [B, seq_len, head_dim]
            point_coords: [N_point, 3] or [B, N_point, 3]
            point_token_mask: [B, seq_len] boolean mask
        """
        bsz, num_heads, seq_len, head_dim = query_states.shape
        
        # JJ: Input validation
        # Force batch size = 1 (current implementation assumes uniform point token positions)
        assert bsz == 1, f"Current implementation only supports batch_size=1, got {bsz}"
        
        # Validate point_token_mask shape
        assert point_token_mask.shape == (bsz, seq_len), \
            f"point_token_mask shape mismatch: expected ({bsz}, {seq_len}), got {point_token_mask.shape}"
        
        # Validate point_coords shape
        num_point_tokens = point_token_mask[0].sum().item()
        if point_coords.dim() == 2:  # [N_point, 3]
            assert point_coords.shape == (num_point_tokens, 3), \
                f"point_coords shape mismatch: expected ({num_point_tokens}, 3), got {point_coords.shape}"
        elif point_coords.dim() == 3:  # [B, N_point, 3]
            assert point_coords.shape == (bsz, num_point_tokens, 3), \
                f"point_coords shape mismatch: expected ({bsz}, {num_point_tokens}, 3), got {point_coords.shape}"
        else:
            raise ValueError(f"point_coords must be 2D [N_point, 3] or 3D [B, N_point, 3], got shape {point_coords.shape}")
        
        # Validate position_embeddings is not None
        if position_embeddings is None:
            raise ValueError("position_embeddings cannot be None when applying mixed RoPE")
        
        # Clone to avoid in-place modifications
        query_states_new = query_states.clone()
        key_states_new = key_states.clone()
        
        # Extract point cloud coordinates for the current tokens
        point_indices = point_token_mask[0].nonzero(as_tuple=True)[0]  # [N_point]
        
        if len(point_indices) > 0 and self.spatial_dim > 0:
            # ========== Step 1: Apply 3D RoPE to SPATIAL dimensions of point tokens ==========
            # Prepare point coordinates (ensure correct device)
            point_coords = point_coords.to(query_states.device)
            if point_coords.dim() == 3:  # [B, N_point, 3] -> [N_point, 3]
                point_coords = point_coords[0]
            
            t_x = point_coords[:, 0] # linear from 0 to 1
            t_y = point_coords[:, 1]
            t_z = point_coords[:, 2]
            
            # Normalize coordinates
            t_x, t_y, t_z = normalize_point_coords_3d(
                t_x, t_y, t_z,
                norm_strategy=self.norm_strategy,
                virtual_resolution=self.virtual_resolution
            )
            
            # JJ: Self-adapted drift based on number of point tokens
            if self.self_adapted_drift_normed_point_coords:
                # Goal: Make 3D point coords distinguishable from text token positions
                # while preserving granularity of 3D point embedding
                
                # Compute drift based on mode
                if self.self_adapted_drift_mode == 'anchor_wrt_avg_temporal':
                    # Use average temporal position - same drift for all points (scalar)
                    drift = len(point_indices) / 2
                elif self.self_adapted_drift_mode == 'anchor_wrt_pointwise_temporal':
                    # Use pointwise temporal position - different drift per point (vector)
                    # NOTE: May cause 3D PE granularity issues if theta stays at 10000
                    drift = torch.arange(len(point_indices), dtype=t_x.dtype, device=t_x.device)
                else:
                    raise ValueError(
                        f"Unknown self_adapted_drift_mode: {self.self_adapted_drift_mode}. "
                        f"Expected 'anchor_wrt_avg_temporal' or 'anchor_wrt_pointwise_temporal'"
                    )
                
                # Apply drift to all coordinates
                t_x = t_x + drift
                t_y = t_y + drift
                t_z = t_z + drift
            
            # Compute 3D RoPE frequencies
            if self.rope_mixed:
                # JJ: GQA - separate freqs for Q and K/V
                if self.rope_mixed_learn_per_axis:
                    # Q frequencies
                    freqs_cis_3d_q = self.compute_cis_3d(self.freqs_3d_q, t_x, t_y, t_z, axial_weights=self.axial_weights_3d_q)
                    # K/V frequencies
                    compute_cis_3d_kv = partial(compute_mixed_cis_3d, num_heads=self.num_key_value_heads)
                    freqs_cis_3d_kv = compute_cis_3d_kv(self.freqs_3d_kv, t_x, t_y, t_z, axial_weights=self.axial_weights_3d_kv)
                else:
                    # Shared frequencies: replicate for x/y/z
                    freqs_shared_q = self.freqs_3d_q.repeat(3, 1)
                    freqs_cis_3d_q = self.compute_cis_3d(freqs_shared_q, t_x, t_y, t_z, axial_weights=self.axial_weights_3d_q)
                    
                    freqs_shared_kv = self.freqs_3d_kv.repeat(3, 1)
                    compute_cis_3d_kv = partial(compute_mixed_cis_3d, num_heads=self.num_key_value_heads)
                    freqs_cis_3d_kv = compute_cis_3d_kv(freqs_shared_kv, t_x, t_y, t_z, axial_weights=self.axial_weights_3d_kv)
            else:
                # JJ: Axial mode - compute separately for Q and K/V
                freqs_cis_3d_q = self.compute_cis_3d_q(t_x=t_x, t_y=t_y, t_z=t_z)
                freqs_cis_3d_kv = self.compute_cis_3d_kv(t_x=t_x, t_y=t_y, t_z=t_z)
                
                # Adjust to match head counts
                # freqs_cis_3d shape: [num_heads, N_point, spatial_dim//2]
                if freqs_cis_3d_q.shape[0] != self.num_heads:
                    # Repeat or slice to match
                    freqs_cis_3d_q = freqs_cis_3d_q[:self.num_heads] if freqs_cis_3d_q.shape[0] > self.num_heads else freqs_cis_3d_q.repeat(self.num_heads // freqs_cis_3d_q.shape[0] + 1, 1, 1)[:self.num_heads]
                if freqs_cis_3d_kv.shape[0] != self.num_key_value_heads:
                    freqs_cis_3d_kv = freqs_cis_3d_kv[:self.num_key_value_heads] if freqs_cis_3d_kv.shape[0] > self.num_key_value_heads else freqs_cis_3d_kv.repeat(self.num_key_value_heads // freqs_cis_3d_kv.shape[0] + 1, 1, 1)[:self.num_key_value_heads]
            
            freqs_cis_3d_q = freqs_cis_3d_q.to(query_states.device)
            freqs_cis_3d_kv = freqs_cis_3d_kv.to(query_states.device)
            # freqs_cis_3d_q: [num_heads, N_point, spatial_dim//2]
            # freqs_cis_3d_kv: [num_key_value_heads, N_point, spatial_dim//2]
            
            # Extract point tokens
            q_point = query_states[:, :, point_indices, :]  # [B, num_heads, N_point, head_dim]
            k_point = key_states[:, :, point_indices, :]
            
            # JJ: Split into spatial and temporal dimensions
            # Use interleaved indices if enabled, otherwise use contiguous blocks
            if self.mixed_rope_spatial_temporal_interleaved:
                # Interleaved mode: select dimensions by indices
                spatial_idx = self.spatial_indices.to(q_point.device)
                temporal_idx = self.temporal_indices.to(q_point.device)
                
                q_point_spatial = q_point[:, :, :, spatial_idx]  # [B, num_heads, N_point, spatial_dim]
                q_point_temporal = q_point[:, :, :, temporal_idx]  # [B, num_heads, N_point, temporal_dim]
                k_point_spatial = k_point[:, :, :, spatial_idx]   # [B, num_key_value_heads, N_point, spatial_dim]
                k_point_temporal = k_point[:, :, :, temporal_idx]  # [B, num_key_value_heads, N_point, temporal_dim]
            else:
                # Contiguous mode: first spatial_dim for spatial, rest for temporal
                q_point_spatial = q_point[:, :, :, :self.spatial_dim]  # [B, num_heads, N_point, spatial_dim]
                q_point_temporal = q_point[:, :, :, self.spatial_dim:]  # [B, num_heads, N_point, temporal_dim]
                k_point_spatial = k_point[:, :, :, :self.spatial_dim]   # [B, num_key_value_heads, N_point, spatial_dim]
                k_point_temporal = k_point[:, :, :, self.spatial_dim:]  # [B, num_key_value_heads, N_point, temporal_dim]
            
            # Apply 3D RoPE to spatial dimensions with GQA-aware freqs
            q_point_spatial_rotated, _ = apply_rotary_emb_3d(q_point_spatial, q_point_spatial, freqs_cis_3d_q)
            _, k_point_spatial_rotated = apply_rotary_emb_3d(k_point_spatial, k_point_spatial, freqs_cis_3d_kv)
            
            # ========== Step 2: Apply 1D RoPE to TEMPORAL dimensions of point tokens ==========
            if self.temporal_dim > 0:
                cos, sin = position_embeddings  # [B, seq_len, head_dim]
                
                # JJ: Extract cos/sin for point tokens and temporal dimensions
                # In interleaved mode, temporal indices are non-contiguous
                if cos.dim() == 3:  # [B, seq_len, head_dim]
                    if self.mixed_rope_spatial_temporal_interleaved:
                        temporal_idx = self.temporal_indices.to(cos.device)
                        cos_point = cos[:, point_indices, :][:, :, temporal_idx]  # [B, N_point, temporal_dim]
                        sin_point = sin[:, point_indices, :][:, :, temporal_idx]
                    else:
                        cos_point = cos[:, point_indices, self.spatial_dim:]  # [B, N_point, temporal_dim]
                        sin_point = sin[:, point_indices, self.spatial_dim:]
                    cos_point = cos_point.unsqueeze(1)  # [B, 1, N_point, temporal_dim]
                    sin_point = sin_point.unsqueeze(1)
                else:
                    if self.mixed_rope_spatial_temporal_interleaved:
                        temporal_idx = self.temporal_indices.to(cos.device)
                        cos_point = cos[..., temporal_idx]
                        sin_point = sin[..., temporal_idx]
                    else:
                        cos_point = cos[..., self.spatial_dim:]
                        sin_point = sin[..., self.spatial_dim:]
                
                # Apply 1D RoPE to temporal dimensions
                q_point_temporal_rotated = (q_point_temporal * cos_point) + (self._rotate_half(q_point_temporal) * sin_point)
                k_point_temporal_rotated = (k_point_temporal * cos_point) + (self._rotate_half(k_point_temporal) * sin_point)
            else:
                # No temporal dimensions, skip
                q_point_temporal_rotated = q_point_temporal
                k_point_temporal_rotated = k_point_temporal
            
            # JJ: Merge spatial (3D RoPE) and temporal (1D RoPE) dimensions
            # In interleaved mode, scatter results back to correct positions
            # In contiguous mode, concatenate directly
            if self.mixed_rope_spatial_temporal_interleaved:
                # Create output tensors with original shape
                q_point_final = torch.zeros_like(q_point)
                k_point_final = torch.zeros_like(k_point)
                
                # Scatter spatial and temporal results back to their positions
                spatial_idx = self.spatial_indices.to(q_point.device)
                temporal_idx = self.temporal_indices.to(q_point.device)
                
                q_point_final[:, :, :, spatial_idx] = q_point_spatial_rotated
                q_point_final[:, :, :, temporal_idx] = q_point_temporal_rotated
                k_point_final[:, :, :, spatial_idx] = k_point_spatial_rotated
                k_point_final[:, :, :, temporal_idx] = k_point_temporal_rotated
            else:
                # Contiguous mode: concatenate [spatial | temporal]
                q_point_final = torch.cat([q_point_spatial_rotated, q_point_temporal_rotated], dim=-1)
                k_point_final = torch.cat([k_point_spatial_rotated, k_point_temporal_rotated], dim=-1)
            
            query_states_new[:, :, point_indices, :] = q_point_final
            key_states_new[:, :, point_indices, :] = k_point_final
        
        # ========== Step 3: Apply 1D RoPE to ALL dimensions of text tokens ==========
        # JJ: If spatial_dim=0, apply 1D RoPE to ALL tokens (including point tokens)
        # Otherwise, only apply to text tokens (point tokens already handled above)
        if self.spatial_dim == 0:
            remaining_indices = torch.arange(seq_len, device=query_states.device)
        else:
            text_mask = ~point_token_mask[0]
            remaining_indices = text_mask.nonzero(as_tuple=True)[0]
        
        if len(remaining_indices) > 0:
            cos, sin = position_embeddings
            # Extract remaining tokens (text tokens, or all tokens if spatial_dim=0)
            q_remaining = query_states_new[:, :, remaining_indices, :]  # [B, num_heads, N_remaining, head_dim]
            k_remaining = key_states_new[:, :, remaining_indices, :]   # [B, num_key_value_heads, N_remaining, head_dim]
            
            # Expand cos/sin for multi-head
            if cos.dim() == 3:  # [B, seq_len, head_dim]
                cos_remaining = cos[:, remaining_indices, :]  # [B, N_remaining, head_dim]
                sin_remaining = sin[:, remaining_indices, :]  # [B, N_remaining, head_dim]
                cos_remaining = cos_remaining.unsqueeze(1)  # [B, 1, N_remaining, head_dim]
                sin_remaining = sin_remaining.unsqueeze(1)  # [B, 1, N_remaining, head_dim]
            else:
                cos_remaining = cos
                sin_remaining = sin
            
            # Apply standard 1D RoPE formula to FULL head_dim
            q_embed = (q_remaining * cos_remaining) + (self._rotate_half(q_remaining) * sin_remaining)
            k_embed = (k_remaining * cos_remaining) + (self._rotate_half(k_remaining) * sin_remaining)
            
            query_states_new[:, :, remaining_indices, :] = q_embed
            key_states_new[:, :, remaining_indices, :] = k_embed
        
        return query_states_new, key_states_new
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads if needed for GQA"""
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MixedRoPE3DQwen2DecoderLayer(Qwen2DecoderLayer):
    """Qwen2 Decoder Layer with MixedRoPE3D Attention"""
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace attention with our custom one
        self.self_attn = MixedRoPE3DQwen2Attention(config, layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Extended forward to pass point_coords and point_token_mask to attention
        """
        residual = hidden_states
        
        # Layer norm before attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention - pass our custom params
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            point_coords=point_coords,
            point_token_mask=point_token_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class Qwen2ModelMixedRoPE3D(Qwen2Model):
    """Qwen2Model with MixedRoPE3D support"""
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # Replace decoder layers with our custom ones
        self.layers = nn.ModuleList(
            [MixedRoPE3DQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Re-initialize post init
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Extended forward to support point_coords and point_token_mask
        
        JJ: Key modification - pass point_coords and point_token_mask through all layers
        """
        # Simply pass through kwargs and let parent handle most logic
        # But we need to intercept layer calls to pass our custom params
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # We cannot directly call parent's forward because it doesn't support our custom params
        # So we replicate the essential forward logic here
        hidden_states = inputs_embeds
        
        # Normalize attention mask if provided
        # JJ: Prepare attention mask to handle KV cache correctly
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, inputs_embeds.shape[:2], inputs_embeds, past_key_values
            )
        
        # Position embeddings (1D RoPE for text tokens)
        position_embeddings = None
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    0, inputs_embeds.shape[1], dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)
            
            # Generate cos, sin for 1D RoPE
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # Pass through all decoder layers
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # JJ: Call layer with our custom parameters
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                point_coords=point_coords,
                point_token_mask=point_token_mask,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values):
        """
        Prepare causal attention mask with proper handling of KV cache
        
        Key insight: During generation with KV cache, attention_mask.shape[1] may not match
        the cached sequence length because point tokens were inserted during prefill.
        We need to handle this mismatch correctly.
        """
        batch_size, seq_length = input_shape
        
        # Get the past sequence length
        past_key_values_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, 'get_seq_length'):
                past_key_values_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                # Legacy cache format: list of (key, value) tuples
                past_key_values_length = past_key_values[0][0].shape[2]
        
        # Handle attention_mask mismatch during generation
        # In generation, attention_mask may not include point tokens that were inserted during prefill
        if attention_mask is not None and past_key_values_length > 0:
            expected_length = past_key_values_length + seq_length
            actual_length = attention_mask.shape[1]
            
            if actual_length < expected_length:
                # attention_mask is shorter than KV cache, need to expand it
                # This happens when point tokens were inserted during prefill
                # We need to insert 1s (can attend) at the point token positions
                # Simple solution: pad with 1s to match the expected length
                padding_length = expected_length - actual_length
                attention_mask = torch.nn.functional.pad(
                    attention_mask, 
                    (0, padding_length), 
                    value=1  # Can attend to point tokens
                )
        
        # Create causal mask: [batch_size, 1, tgt_len, src_len]
        if seq_length > 1:
            # Prefill or multi-token generation: create full causal mask
            causal_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        else:
            # Single token generation: no causal masking needed within this token
            # But we still need a mask for attending to past tokens
            causal_mask = None
        
        # Expand the user-provided attention_mask
        if attention_mask is not None:
            # attention_mask is [batch_size, seq_len] where seq_len may include point tokens
            # We need to expand it to [batch_size, 1, tgt_len, src_len]
            expanded_attn_mask = self._expand_mask(
                attention_mask, 
                inputs_embeds.dtype, 
                tgt_len=seq_length
            )
            
            if causal_mask is not None:
                combined_attention_mask = expanded_attn_mask + causal_mask
            else:
                combined_attention_mask = expanded_attn_mask
        else:
            combined_attention_mask = causal_mask
        
        return combined_attention_mask
    
    def _make_causal_mask(self, input_shape, dtype, device, past_key_values_length=0):
        """
        Make causal mask for autoregressive decoding
        Returns: [batch_size, 1, tgt_len, src_len + past_key_values_length]
        """
        bsz, tgt_len = input_shape
        
        # Create causal mask: upper triangle is masked (filled with -inf)
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        
        # If we have past key values, prepend zeros for those positions (can attend to all past)
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        
        # Expand to batch: [bsz, 1, tgt_len, tgt_len + past_key_values_length]
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    def _expand_mask(self, mask, dtype, tgt_len=None):
        """
        Expand attention mask from [bsz, src_len] to [bsz, 1, tgt_len, src_len]
        Inverts the mask: 0 becomes -inf (masked), 1 stays 0 (unmasked)
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        
        # Expand: [bsz, src_len] -> [bsz, 1, tgt_len, src_len]
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        
        # Invert: 1.0 -> 0.0 (can attend), 0.0 -> -inf (cannot attend)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class Qwen2ForCausalLMMixedRoPE3D(Qwen2ForCausalLM):
    """Qwen2ForCausalLM with MixedRoPE3D support"""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace model with our custom one
        self.model = Qwen2ModelMixedRoPE3D(config)
        # Re-initialize weights
        self.post_init()


if __name__ == "__main__":
    import torch
    from transformers import AutoConfig
    
    # Test loading
    print("Testing MixedRoPE3D Qwen2 implementation...")
    
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Add 3D RoPE config
    config.rope_mixed = True
    config.norm_strategy = "virtual_resolution"
    config.virtual_resolution = 32.0
    config.rope_mixed_learn_per_axis = True
    config.mixedRoPE_3d_learned_axial_mixing_weight = False
    
    print(f"Config: {config}")
    print("✓ Configuration ready for MixedRoPE3D")

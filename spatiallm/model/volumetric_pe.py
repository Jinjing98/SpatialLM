"""
3D Position Encodings for Point Cloud Tokens

Implements two variants:
1. 3D_RoPE: Rotary Position Embedding extended to 3D
2. 3D_Sinusoidal: Sinusoidal Position Embedding extended to 3D

**CRITICAL DESIGN DECISION (3D_RoPE):**
- 3D_RoPE applies to head_dim (e.g., 64 = 21 + 21 + 22) AFTER head splitting
- All attention heads share the same 3D spatial encoding pattern
- This works with GQA (Grouped Query Attention) where K/V have fewer heads than Q
- Contrast with standard 1D RoPE which uses sequence position (1D)

Key differences from standard 1D positional encodings:
- Standard: All dims use sequence position (1D)
- 3D variants: Split dims across X, Y, Z spatial coordinates (3D)
"""

import math
import torch
from torch import nn
import logging

from spatiallm.model.dim_allocation_utils import allocate_rope_dimensions

logger = logging.getLogger(__name__)

# Global flag to track if 3D RoPE is available
_3D_ROPE_AVAILABLE = True


class RotaryEmbedding3D(nn.Module):
    """
    3D Rotary Position Embedding for point cloud tokens.
    
    **DESIGN:** Applies to head_dim AFTER head splitting (per-head application).
    All attention heads share the same 3D spatial encoding pattern.
    
    Splits the head_dim into components for 1D (temporal) and 3D (X, Y, Z) spatial
    coordinates using the allocate_rope_dimensions() utility function.
    
    Dimension Allocation:
        - "3D_only": ratio_1d=0.0 → (0, 21, 21, 22) for head_dim=64
        - "3D_with_1D": ratio_1d=0.5 → (32, 10, 10, 12) for head_dim=64
    
    Args:
        head_dim: Dimension per attention head (e.g., 64)
        max_position_embeddings: Maximum sequence length
        base: Base for inverse frequency calculation (default: 10000)
        device: Device to place tensors on
        merge_rule: "3D_only" (pure spatial) or "3D_with_1D" (hybrid temporal+spatial)
    """
    
    def __init__(self, head_dim, max_position_embeddings=32768, base=10000, base_3d=10000, device=None, merge_rule="3D_only"):
        super().__init__()
        
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.base_3d = base_3d
        self.merge_rule = merge_rule
        
        # Dimension allocation based on merge_rule
        if merge_rule == "3D_with_1D":
            # Hybrid: 50% for 1D (temporal), 50% for 3D (spatial)
            # Use divide=1 to allow (21,21,22) split without forcing even dimensions
            self.d_1d, self.d_x, self.d_y, self.d_z = allocate_rope_dimensions(
                dim=head_dim,
                ratio_1d=1.0, divide=0, # as RoPE_1D
                # ratio_1d=0.75, divide=2,  # No strict divisibility constraint
                # ratio_1d=0.0, divide=2,  # 3D rope only
            )
            
            # logger.info(
            print(
                f"[3D_RoPE] merge_rule='3D_with_1D': "
                f"head_dim={head_dim} split into d_1d={self.d_1d}, "
                f"d_x={self.d_x}, d_y={self.d_y}, d_z={self.d_z}"
            )
        else:  # "3D_only"
            # Pure 3D: no 1D component
            # Use divide=1 to allow (21,21,22) split for head_dim=64
            self.d_1d, self.d_x, self.d_y, self.d_z = allocate_rope_dimensions(
                dim=head_dim,
                ratio_1d=0.0,
                divide=2  # No strict divisibility constraint
            )
            
            # logger.info(
            print(
                f"[3D_RoPE] merge_rule='3D_only': "
                f"head_dim={head_dim} split into d_x={self.d_x}, d_y={self.d_y}, d_z={self.d_z}"
            )
        
        # Compute inverse frequencies for each dimension
        # For dimension d, we need (d+1)//2 frequency bands to handle both even and odd d
        # Each frequency band will be duplicated to form pairs for RoPE
        
        if self.d_1d > 0:
            # Compute inv_freq for 1D component
            inv_freq_1d = 1.0 / (self.base ** (torch.arange(0, self.d_1d, 2, dtype=torch.float32).to(device) / self.d_1d))
            self.register_buffer("inv_freq_1d", inv_freq_1d, persistent=False)
        else:
            self.register_buffer("inv_freq_1d", None, persistent=False)
        
        inv_freq_x = 1.0 / (self.base_3d ** (torch.arange(0, self.d_x, 2, dtype=torch.float32).to(device) / self.d_x))
        inv_freq_y = 1.0 / (self.base_3d ** (torch.arange(0, self.d_y, 2, dtype=torch.float32).to(device) / self.d_y))
        inv_freq_z = 1.0 / (self.base_3d ** (torch.arange(0, self.d_z, 2, dtype=torch.float32).to(device) / self.d_z))
        
        self.register_buffer("inv_freq_x", inv_freq_x, persistent=False)
        self.register_buffer("inv_freq_y", inv_freq_y, persistent=False)
        self.register_buffer("inv_freq_z", inv_freq_z, persistent=False)
    
    def forward(self, coords_3d, seq_len, position_ids=None):
        """
        Compute 3D RoPE cos and sin values for point cloud tokens.
        
        Args:
            coords_3d: (N, 3) - Normalized 3D coordinates [X, Y, Z] in range [0, 1]
            seq_len: Total sequence length (for compatibility, not used)
            position_ids: (N,) - Optional position IDs for 1D RoPE component (used when merge_rule="3D_with_1D")
        
        Returns:
            cos_cached: (N, head_dim) - Cosine values for RoPE
            sin_cached: (N, head_dim) - Sine values for RoPE
        """
        device = coords_3d.device
        dtype = coords_3d.dtype
        N = coords_3d.shape[0]
        
        # === Part 1: Compute 1D RoPE component (if merge_rule="3D_with_1D") ===
        if self.d_1d > 0:
            if position_ids is None:
                raise ValueError(
                    f"[3D_RoPE] merge_rule='3D_with_1D' requires position_ids, but got None!"
                )
            
            # Ensure position_ids is the right shape: (N,) or (N, 1)
            if position_ids.dim() == 1:
                position_ids_1d = position_ids.unsqueeze(1).float()  # (N, 1)
            else:
                position_ids_1d = position_ids.float()  # (N, 1)
            
            # Compute 1D frequencies
            inv_freq_1d = self.inv_freq_1d.to(device)
            freqs_1d = torch.matmul(position_ids_1d, inv_freq_1d.unsqueeze(0))  # (N, d_1d//2)
        else:
            freqs_1d = None
        
        # === Part 2: Compute 3D RoPE components ===
        # Extract X, Y, Z coordinates (N, 1)
        x_coords = coords_3d[:, 0:1]  # (N, 1)
        y_coords = coords_3d[:, 1:2]  # (N, 1)
        z_coords = coords_3d[:, 2:3]  # (N, 1)
        
        # Move inv_freq to correct device
        inv_freq_x = self.inv_freq_x.to(device)
        inv_freq_y = self.inv_freq_y.to(device)
        inv_freq_z = self.inv_freq_z.to(device)
        
        # Compute frequencies: coords * inv_freq
        # (N, 1) x (d//2,) -> (N, d//2)
        freqs_x = torch.matmul(x_coords, inv_freq_x.unsqueeze(0))  # (N, d_x//2)
        freqs_y = torch.matmul(y_coords, inv_freq_y.unsqueeze(0))  # (N, d_y//2)
        freqs_z = torch.matmul(z_coords, inv_freq_z.unsqueeze(0))  # (N, d_z//2)


        # Duplicate frequencies for RoPE: [θ0, θ1, ..., θ0, θ1, ...]
        # This matches native RoPE (concatenate, not interleave)
        # For odd dimensions, we truncate or pad to match exact dimension
        def expand_freqs(freqs, target_dim):
            # freqs: (N, ceil(d/2)) or (N, floor(d/2)), target: (N, d)
            # Native RoPE: torch.cat([freqs, freqs], dim=-1)
            freqs_expanded = torch.cat([freqs, freqs], dim=-1)  # (N, d//2 * 2)
            current_dim = freqs_expanded.shape[-1]
            if current_dim > target_dim:
                # Truncate (for odd dimensions where we have one extra)
                freqs_expanded = freqs_expanded[:, :target_dim]
            elif current_dim < target_dim:
                assert False, "This should not happen"
                pad_size = target_dim - current_dim
                freqs_expanded = torch.cat([freqs_expanded, torch.zeros(freqs_expanded.shape[0], pad_size, device=freqs.device, dtype=freqs.dtype)], dim=-1)
            return freqs_expanded
        
        freqs_x = expand_freqs(freqs_x, self.d_x)  # (N, d_x)
        freqs_y = expand_freqs(freqs_y, self.d_y)  # (N, d_y)
        freqs_z = expand_freqs(freqs_z, self.d_z)  # (N, d_z)
        
        # Concatenate along feature dimension
        if self.d_1d > 0:
            # merge_rule="3D_with_1D": [1D_freqs, X_freqs, Y_freqs, Z_freqs]
            freqs_1d_expanded = expand_freqs(freqs_1d, self.d_1d)  # (N, d_1d)
            freqs = torch.cat([freqs_1d_expanded, freqs_x, freqs_y, freqs_z], dim=-1)  # (N, head_dim)
        else:
            # merge_rule="3D_only": [X_freqs, Y_freqs, Z_freqs]
            freqs = torch.cat([freqs_x, freqs_y, freqs_z], dim=-1)  # (N, head_dim)
        
        # Compute cos and sin
        cos_cached = freqs.cos().to(dtype)  # (N, head_dim)
        sin_cached = freqs.sin().to(dtype)  # (N, head_dim)
        
        return cos_cached, sin_cached



def rotate_half(x):
    """Rotates half the hidden dims of the input (for RoPE application)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_3d(q, k, cos_3d, sin_3d):
    """
    Apply 3D RoPE to query and key tensors AFTER head splitting (per-head).
    
    **DESIGN:** All attention heads share the same 3D spatial encoding pattern.
    The rotation is applied after the tensor is reshaped into heads.
    
    Args:
        q: Query tensor (bsz, num_heads, seq_len, head_dim) - AFTER head split!
        k: Key tensor (bsz, num_key_value_heads, seq_len, head_dim) - AFTER head split!
        cos_3d: Cosine values (seq_len, head_dim)
        sin_3d: Sine values (seq_len, head_dim)
    
    Returns:
        q_embed, k_embed: Rotated query and key tensors (same shape as input)
    """
    # Reshape to broadcast: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos_3d = cos_3d.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin_3d = sin_3d.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    
    # Apply rotation (broadcasts across batch and heads)
    q_embed = (q * cos_3d) + (rotate_half(q) * sin_3d)
    k_embed = (k * cos_3d) + (rotate_half(k) * sin_3d)
    
    return q_embed, k_embed


def compute_3d_sinusoidal_pe(coords_3d, hidden_size, num_heads, base=10000, base_3d=10000, device=None, merge_rule="3D_only", position_ids=None):
    """
    Compute 3D Sinusoidal Position Encoding for point cloud tokens.
    
    **DESIGN:** Unlike 3D_RoPE (per-head), this applies to FULL hidden_size,
    allowing each attention head to learn from different spatial dimensions (no sharing).
    
    Args:
        coords_3d: (N, 3) - Normalized 3D coordinates [X, Y, Z] in range [0, 1]
        hidden_size: Full embedding dimension (e.g., 896)
        num_heads: Number of attention heads (not used, kept for compatibility)
        base: Base for frequency calculation (default: 10000)
        device: Device to place tensors on
        merge_rule: "3D_only" or "3D_with_1D" (see pcd_pe_merge_rule)
        position_ids: (N,) - Optional position IDs for 1D component (required for "3D_with_1D")
    
    Returns:
        pe: (N, hidden_size) - 3D Sinusoidal position encoding
        
    Dimension Allocation:
        Uses allocate_rope_dimensions() utility function:
        - "3D_only": ratio_1d=0.0, divide=1 → (0, 298, 298, 300) for hidden_size=896
        - "3D_with_1D": ratio_1d=0.5, divide=1 → (448, 149, 149, 150) for hidden_size=896
    """
    N = coords_3d.shape[0]
    
    # Dimension allocation based on merge_rule
    if merge_rule == "3D_with_1D":
        # Hybrid: 50% for 1D (temporal), 50% for 3D (spatial)
        # Use divide=1 to allow flexible splits
        d_1d, d_x, d_y, d_z = allocate_rope_dimensions(
            dim=hidden_size,
            ratio_1d=0.5,
            divide=2  # No strict divisibility constraint
        )
        
        logger.info(
            f"[3D_Sinusoidal] merge_rule='3D_with_1D': "
            f"hidden_size={hidden_size} split into d_1d={d_1d}, "
            f"d_x={d_x}, d_y={d_y}, d_z={d_z}"
        )
    else:  # "3D_only"
        # Pure 3D: no 1D component
        # Use divide=1 to allow (298,298,300) split for hidden_size=896
        d_1d, d_x, d_y, d_z = allocate_rope_dimensions(
            dim=hidden_size,
            ratio_1d=0.0,
            divide=2  # No strict divisibility constraint
        )
        
        logger.info(
            f"[3D_Sinusoidal] merge_rule='3D_only': "
            f"hidden_size={hidden_size} split into d_x={d_x}, d_y={d_y}, d_z={d_z}"
        )
    
    # Extract X, Y, Z coordinates
    device = device or coords_3d.device
    dtype = coords_3d.dtype
    
    # Compute position encodings for each dimension (same as standard sinusoidal PE)
    # PE(pos, 2i) = sin(pos / base^(2i/d))
    # PE(pos, 2i+1) = cos(pos / base^(2i/d))
    
    def compute_sinusoidal_dim(coords, d, theta):
        """Compute sinusoidal PE for one dimension (spatial or temporal)."""
        # Frequency term: base^(-2i/d) for i in [0, d//2)
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(theta) / d)
        ).unsqueeze(0)  # (1, d//2)
        
        # Compute sin and cos
        angles = coords * div_term  # (N, 1) * (1, d//2) = (N, d//2)
        sin_vals = torch.sin(angles)  # (N, d//2)
        cos_vals = torch.cos(angles)  # (N, d//2)
        
        # Interleave: [sin, cos, sin, cos, ...]
        pe = torch.stack([sin_vals, cos_vals], dim=-1).flatten(-2)  # (N, d//2 * 2)
        
        # Handle odd dimensions
        if pe.shape[-1] > d:
            pe = pe[:, :d]
        elif pe.shape[-1] < d:
            # Pad with zeros (shouldn't happen, but just in case)
            pad_size = d - pe.shape[-1]
            pe = torch.cat([pe, torch.zeros(N, pad_size, device=device, dtype=dtype)], dim=-1)
        
        return pe  # (N, d)
    
    # === Part 1: Compute 1D temporal component (if merge_rule="3D_with_1D") ===
    if d_1d > 0:
        if position_ids is None:
            raise ValueError(
                f"[3D_Sinusoidal] merge_rule='3D_with_1D' requires position_ids, but got None!"
            )
        
        # Ensure position_ids is the right shape: (N,) or (N, 1)
        if position_ids.dim() == 1:
            position_ids_1d = position_ids.unsqueeze(1).float()  # (N, 1)
        else:
            position_ids_1d = position_ids.float()  # (N, 1)
        
        pe_1d = compute_sinusoidal_dim(position_ids_1d, d_1d, theta=base)  # (N, d_1d)
    else:
        pe_1d = None
    
    # === Part 2: Compute 3D spatial components ===
    x_coords = coords_3d[:, 0:1]  # (N, 1)
    y_coords = coords_3d[:, 1:2]  # (N, 1)
    z_coords = coords_3d[:, 2:3]  # (N, 1)
    
    pe_x = compute_sinusoidal_dim(x_coords, d_x, theta=base_3d)  # (N, d_x)
    pe_y = compute_sinusoidal_dim(y_coords, d_y, theta=base_3d)  # (N, d_y)
    pe_z = compute_sinusoidal_dim(z_coords, d_z, theta=base_3d)  # (N, d_z)
    
    # === Part 3: Concatenate all components ===
    if pe_1d is not None:
        # merge_rule="3D_with_1D": [1D, X, Y, Z]
        pe = torch.cat([pe_1d, pe_x, pe_y, pe_z], dim=-1)  # (N, hidden_size)
    else:
        # merge_rule="3D_only": [X, Y, Z]
        pe = torch.cat([pe_x, pe_y, pe_z], dim=-1)  # (N, hidden_size)
    
    # Verify shape
    assert pe.shape == (N, hidden_size), f"PE shape mismatch: {pe.shape} != ({N}, {hidden_size})"
    
    # Detach to avoid gradient flow
    if pe.requires_grad:
        pe = pe.detach()
    
    return pe.to(dtype)

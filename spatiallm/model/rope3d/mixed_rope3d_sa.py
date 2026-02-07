"""
This code was originally obtained from:
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

import torch
import torch.nn as nn
import math
from functools import partial

# JJ: Unified initialization for 3D RoPE (方案B)
def init_3d_freqs_unified(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """
    Initialize unified frequencies for 3D point cloud RoPE
    
    This generates a single set of base frequencies that can be:
    - Used directly for shared learning (rope_mixed_learn_per_axis=False)
    - Replicated with perturbation for per-axis learning (rope_mixed_learn_per_axis=True)
    
    Args:
        dim: head dimension
        num_heads: number of attention heads
        theta: base for frequency calculation
        rotate: whether to use random rotation angles per head
    
    Returns:
        freqs: [num_heads, dim//2] tensor - base frequency for all axes
    """
    freqs = []
    # JJ: Standard RoPE frequency calculation
    # Generate dim//2 frequency bins: θ^(-2i/d) for i=0,1,2,...,dim/2-1
    num_freqs = dim // 2
    freq_indices = torch.arange(num_freqs, dtype=torch.float32)
    freqs_base = 1.0 / (theta ** (2 * freq_indices / dim))
    
    for i in range(num_heads):
        # Each head can have different rotation angle (learnable diversity)
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        # Apply rotation to the frequency basis
        f = freqs_base * torch.cos(angles) + freqs_base * torch.sin(angles) * 1j
        # Take real part as the learnable parameter (imaginary part is implicit via RoPE)
        freqs.append(freqs_base * torch.cos(angles))
    
    freqs = torch.stack(freqs, dim=0)  # [num_heads, dim//2]
    print('init 3d freqs unified: ', freqs.shape)
    print(freqs)
    return freqs

# JJ: 3D normalization strategies
def normalize_point_coords_3d(
    t_x: torch.Tensor, 
    t_y: torch.Tensor, 
    t_z: torch.Tensor, 
    norm_strategy: str = "virtual_resolution",
    virtual_resolution: float = 32.0
):
    """
    Normalize 3D point cloud coordinates with different strategies
    
    Args:
        t_x, t_y, t_z: Raw coordinates [N] or [B, N]
        norm_strategy: "virtual_resolution" or "unit_range"
        virtual_resolution: Target range for virtual_resolution strategy
    
    Returns:
        Normalized t_x, t_y, t_z
    """
    if norm_strategy == "unit_range":
        # Strategy 1: Normalize to [0, 1]
        # Suitable for scenes with uniform importance across space
        t_x = (t_x - t_x.min()) / (t_x.max() - t_x.min() + 1e-8)
        t_y = (t_y - t_y.min()) / (t_y.max() - t_y.min() + 1e-8)
        t_z = (t_z - t_z.min()) / (t_z.max() - t_z.min() + 1e-8)
        
    elif norm_strategy == "virtual_resolution":
        # Strategy 2: Normalize to [0, virtual_resolution]
        # Mimics image resolution, better for frequency encoding
        t_x = (t_x - t_x.min()) / (t_x.max() - t_x.min() + 1e-8) * virtual_resolution
        t_y = (t_y - t_y.min()) / (t_y.max() - t_y.min() + 1e-8) * virtual_resolution
        t_z = (t_z - t_z.min()) / (t_z.max() - t_z.min() + 1e-8) * virtual_resolution
    
    else:
        raise ValueError(f"Unknown norm_strategy: {norm_strategy}. Choose 'unit_range' or 'virtual_resolution'")
    
    return t_x, t_y, t_z

# JJ: 3D version for point clouds
def compute_mixed_cis_3d(
    freqs: torch.Tensor, 
    t_x: torch.Tensor, 
    t_y: torch.Tensor, 
    t_z: torch.Tensor,
    num_heads: int
):
    """
    Compute mixed 3D RoPE frequencies for point clouds
    
    KEY DIFFERENCE from 2D:
    - 2D: freqs_cis = polar(1, freqs_x + freqs_y)
          where freqs_x and freqs_y are DIFFERENT frequency bands
          Shape: freqs[0] → [num_heads, dim//4], freqs[1] → [num_heads, dim//4]
          Combined via concat in apply_rotary_emb
    
    - 3D: freqs_cis = polar(1, freqs_x*t_x + freqs_y*t_y + freqs_z*t_z)
          where freqs_x, freqs_y, freqs_z share SAME frequency basis
          Shape: freqs[0/1/2] → [num_heads, dim//2] each
          Combined via addition of coordinate-weighted frequencies
    
    Args:
        freqs: [3, num_heads * (dim//2)] flattened frequency parameters
        t_x, t_y, t_z: [N] normalized coordinates
        num_heads: number of attention heads
    
    Returns:
        freqs_cis: [num_heads, N, dim//2] complex tensor
    """
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        # Compute frequency encoding for each dimension
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_z = (t_z.unsqueeze(-1) @ freqs[2].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        print('mixed cis 3d: ')
        print('freqs_x shape: ', freqs_x.shape)
        print('freqs_y shape: ', freqs_y.shape)
        print('freqs_z shape: ', freqs_z.shape)
        # JJ: KEY OPERATION - Addition combines 3D spatial information
        # Each point's encoding = sum of x/y/z contributions
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y + freqs_z)
        print('freqs_cis shape: ', freqs_cis.shape)
    return freqs_cis

# JJ: 3D version for point clouds
def compute_axial_cis_3d(
    dim: int, 
    t_x: torch.Tensor, 
    t_y: torch.Tensor, 
    t_z: torch.Tensor,
    theta: float = 100.0
):
    """
    Compute axial 3D RoPE frequencies for point clouds
    
    Args:
        dim: head dimension
        t_x, t_y, t_z: [N] normalized coordinates
        theta: base for frequency calculation
    
    Returns:
        freqs_cis: [N, dim//2] complex tensor
            Uses same frequency bins for x/y/z, combined via addition
    """
    # JJ: Standard RoPE frequency: θ^(-2i/d) shared across all axes
    num_freqs = (dim // 3) // 2
    freq_indices = torch.arange(num_freqs, dtype=torch.float32)
    freqs_base = 1.0 / (theta ** (freq_indices / ((dim // 3) // 2)))
    
    # Compute for each dimension with the same frequency basis
    freqs_x = torch.outer(t_x, freqs_base)
    freqs_y = torch.outer(t_y, freqs_base)
    freqs_z = torch.outer(t_z, freqs_base)
    print('axial cis 3d: ')
    print('freqs_x shape: ', freqs_x.shape)
    print('freqs_y shape: ', freqs_y.shape)
    print('freqs_z shape: ', freqs_z.shape)

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
    print('freqs_cis_x shape: ', freqs_cis_x.shape)
    print('freqs_cis_y shape: ', freqs_cis_y.shape)
    print('freqs_cis_z shape: ', freqs_cis_z.shape)
    print('freqs_cis shape: ', torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z], dim=-1).shape)
    return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape freqs_cis to broadcast with x
    
    Args:
        freqs_cis: frequency tensor, can be:
            - [seq_len, head_dim] for 2D axial RoPE
            - [num_heads, seq_len, head_dim] for 2D/3D mixed RoPE
        x: query/key tensor [B, num_heads, seq_len, head_dim]
    
    Returns:
        reshaped freqs_cis
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    
    # Case 1: [seq_len, head_dim] - standard axial RoPE
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    # Case 2: [num_heads, seq_len, head_dim] - mixed RoPE  
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    # Case 3: Fallback - try to infer
    else:
        # Assume it's [num_heads, seq_len, head_dim] if 3D
        if freqs_cis.ndim == 3:
            shape = [1] + list(freqs_cis.shape)  # [1, num_heads, seq_len, head_dim]
        # Assume it's [seq_len, head_dim] if 2D
        elif freqs_cis.ndim == 2:
            shape = [1, 1] + list(freqs_cis.shape)  # [1, 1, seq_len, head_dim]
        else:
            raise ValueError(f"Unsupported freqs_cis shape: {freqs_cis.shape} for x shape: {x.shape}")
    
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    print('freqs_cis before broadcast: ', freqs_cis.shape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    print('xq shape: ', xq.shape)
    print('xq_ shape: ', xq_.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    print('freqs_cis shape: ', freqs_cis.shape)
    print('xq_out shape: ', xq_out.shape)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# JJ: 3D version for point clouds
class RoPEAttention_3D(Attention):
    """Multi-head Attention block with 3D rotary position embeddings for point clouds."""
    def __init__(
        self, 
        *args, 
        rope_theta=10.0, 
        rope_mixed=True,
        norm_strategy="virtual_resolution",
        virtual_resolution=32.0,
        rope_mixed_learn_per_axis=True,  # JJ: Flag to control per-axis frequency learning
        **kwargs
    ):
        """
        Args:
            rope_mixed_learn_per_axis: If True, x/y/z axes have independent learnable frequencies.
                                If False, all axes share the same frequency parameters.
                                Only effective when rope_mixed=True.
        """
        super().__init__(*args, **kwargs)
        
        self.rope_mixed = rope_mixed
        self.norm_strategy = norm_strategy
        self.virtual_resolution = virtual_resolution
        self.rope_mixed_learn_per_axis = rope_mixed_learn_per_axis
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis_3d, num_heads=self.num_heads)
            
            # JJ: Use unified initialization (方案B)
            freqs_base = init_3d_freqs_unified(
                dim=self.dim // self.num_heads, 
                num_heads=self.num_heads, 
                theta=rope_theta, 
                rotate=True
            )  # [num_heads, dim//2]
            
            if rope_mixed_learn_per_axis:
                # JJ: Independent learnable frequencies for x/y/z
                # Start from same base but allow independent learning
                freqs = freqs_base.unsqueeze(0).repeat(3, 1, 1)  # [3, num_heads, dim//2]
                # Add small random perturbation to make x/y/z initially different
                freqs = freqs + torch.randn_like(freqs) * 0.01
                freqs = freqs.view(3, -1)  # [3, num_heads * (dim//2)]
                print('Learned param freqs shape (per-axis): ', freqs.shape)
                self.freqs = nn.Parameter(freqs, requires_grad=True)
            else:
                # JJ: Shared frequencies across x/y/z axes
                # All axes use the SAME parameters (truly shared)
                freqs_shared = freqs_base.view(1, -1)  # [1, num_heads * (dim//2)]
                print('Learned param freqs shape (shared): ', freqs_shared.shape)
                self.freqs = nn.Parameter(freqs_shared, requires_grad=True)
            print('done with init mixed rope 3d.')
        else:
            self.compute_cis = partial(compute_axial_cis_3d, dim=self.dim // self.num_heads, theta=rope_theta)
    
    def forward(self, x, point_coords):
        """
        Args:
            x: [B, N, C] token embeddings
            point_coords: [B, N, 3] or [N, 3] point cloud coordinates (x, y, z)
        """
        print('RoPEAttention_3D forward: ')
        B, N, C = x.shape
        print('x shape: ', x.shape)
        print('point_coords shape: ', point_coords.shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ###### Apply 3D rotary position embedding
        # Extract and normalize coordinates
        if point_coords.dim() == 2:  # [N, 3]
            assert B == 1, "Batch size must be 1 for 2D point coordinates"
            point_coords = point_coords.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]
        
        # Use first batch's coordinates (assuming same scene structure)
        t_x = point_coords[0, :, 0]  # [N]
        t_y = point_coords[0, :, 1]  # [N]
        t_z = point_coords[0, :, 2]  # [N]
        print('t_x shape (before norm): ', t_x.shape)
        print('t_y shape (before norm): ', t_y.shape)
        print('t_z shape (before norm): ', t_z.shape)
        
        # Normalize coordinates
        t_x, t_y, t_z = normalize_point_coords_3d(
            t_x, t_y, t_z, 
            norm_strategy=self.norm_strategy,
            virtual_resolution=self.virtual_resolution
        )
        print('t_x shape (after norm): ', t_x.shape)
        print('t_y shape (after norm): ', t_y.shape)
        print('t_z shape (after norm): ', t_z.shape)
        
        # Compute frequency encodings
        if self.rope_mixed:
            # JJ: Handle per-axis vs shared frequency learning
            if self.rope_mixed_learn_per_axis:
                # Independent frequencies for x/y/z
                print('Using per-axis frequencies')
                print('freqs shape: ', self.freqs.shape)
                freqs_cis = self.compute_cis(self.freqs, t_x, t_y, t_z)
            else:
                # Shared frequencies: replicate for x/y/z
                print('Using shared frequencies')
                print('freqs shape: ', self.freqs.shape)
                freqs_shared = self.freqs.repeat(3, 1)  # [1, D] -> [3, D]
                print('freqs_shared shape: ', freqs_shared.shape)
                freqs_cis = self.compute_cis(freqs_shared, t_x, t_y, t_z)
            print('freqs_cis shape: ', freqs_cis.shape)
        else:
            freqs_cis = self.compute_cis(t_x=t_x, t_y=t_y, t_z=t_z)
            print('freqs_cis shape (axial): ', freqs_cis.shape)
        
        freqs_cis = freqs_cis.to(x.device)
        
        # Apply RoPE to all tokens (no CLS token skip)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        #########
        
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

if __name__ == "__main__":

# 维度	2D RoPE (Image)	3D RoPE (Point Cloud)
# 频率分配策略	独立频段 (INDEPENDENT)	共享频率基 (SHARED)
# x方向	dim//4 独立频率	dim//2 共享频率基
# y方向	dim//4 独立频率 (≠ x)	dim//2 共享频率基 (= x)
# z方向	N/A	dim//2 共享频率基 (= x)
# 组合方式	Concat: [freqs_x, freqs_y]	Addition: freqs_x*t_x + freqs_y*t_y + freqs_z*t_z
# 输出维度	dim//2	dim//2
# 公式	polar(1, freqs_x + freqs_y)	polar(1, Σ freqs_i * t_i)


    
    batch_size = 2
    num_points = 1024
    seq_len_3d = num_points
    num_heads = 2
    head_dim = 12
    dim = head_dim * num_heads
    # rope_mixed=True
    rope_mixed=False
    rope_mixed_learn_per_axis = True
    # rope_mixed_learn_per_axis = False
    norm_strategy = "virtual_resolution"
    virtual_res = 32.0
    rope_theta=10.0

 
    # Generate synthetic point cloud
    point_coords = torch.randn(num_points, 3) * 10  # Random point cloud in range [-10, 10]
 
    x_3d = torch.randn(batch_size, seq_len_3d, dim)
    
    
    model_3d = RoPEAttention_3D(
        dim=dim, 
        num_heads=num_heads, 
        rope_theta=rope_theta, 
        rope_mixed=rope_mixed,
        norm_strategy=norm_strategy,
        virtual_resolution=virtual_res,
        rope_mixed_learn_per_axis=rope_mixed_learn_per_axis
    )
    print(f"  Virtual resolution: {virtual_res}")
    
    print(f"\n[LEARNABLE FREQS 3D]")
    if rope_mixed:
        if rope_mixed_learn_per_axis:
            # [3, num_heads * (dim//2)]
            freq_bins_per_dim = model_3d.freqs.shape[1] // num_heads
            freqs_reshaped = model_3d.freqs.view(3, num_heads, freq_bins_per_dim)

        else:
            print(f"  Mode: SHARED frequencies across axes (x=y=z)")
            # [1, num_heads * (dim//2)]
            freq_bins_per_dim = model_3d.freqs.shape[1] // num_heads
            freqs_reshaped = model_3d.freqs.view(1, num_heads, freq_bins_per_dim)

    # Test normalization
    t_x = point_coords[:, 0]
    t_y = point_coords[:, 1]
    t_z = point_coords[:, 2]
    t_x_norm, t_y_norm, t_z_norm = normalize_point_coords_3d(
        t_x, t_y, t_z,
        norm_strategy=norm_strategy,
        virtual_resolution=virtual_res,
    )
    
    print(f"\n[NORMALIZATION EFFECT]")
    print(f"  X: [{t_x.min():.2f}, {t_x.max():.2f}] → [{t_x_norm.min():.2f}, {t_x_norm.max():.2f}]")
    
    # Forward pass
    output_3d = model_3d(x_3d, point_coords)
    

    axis_mode = "per-axis" if rope_mixed_learn_per_axis else "shared"
    print(f" rope_mixed_learn_per_axis Mode: {axis_mode} frequencies across axes (x=y=z)")


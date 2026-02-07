"""
This code was originally obtained from:
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

import torch
import torch.nn as nn
import math
from functools import partial

def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
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


class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""
    def __init__(self, *args, rope_theta=10.0, rope_mixed=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.rope_mixed = rope_mixed        
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = init_2d_freqs(
                dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta, 
                rotate=True
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x=14, end_y=14)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=self.dim // self.num_heads, theta=rope_theta)
            freqs_cis = self.compute_cis(end_x=14, end_y=14)
            self.freqs_cis = freqs_cis
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ###### Apply rotary position embedding
        w = h = math.sqrt(x.shape[1] - 1)
        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(end_x=w, end_y=h)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            freqs_cis = self.freqs_cis
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                freqs_cis = self.compute_cis(end_x=w, end_y=h)
            freqs_cis = freqs_cis.to(x.device)
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)        
        #########
        
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

if __name__ == "__main__":
    # JJ: Enhanced test to understand core design
    print("="*80)
    print("Testing RoPEAttention with detailed outputs")
    print("="*80)
    
    # Configuration
    batch_size = 2
    seq_len = 197  # 1 CLS token + 196 spatial tokens (14x14)
    dim = 768
    num_heads = 12
    head_dim = dim // num_heads
    
    print(f"\n[CONFIG]")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len} (1 CLS + {seq_len-1} spatial tokens)")
    print(f"  Hidden dim: {dim}, Num heads: {num_heads}, Head dim: {head_dim}")
    
    # Test both modes
    for rope_mixed in [True, False]:
        print("\n" + "="*80)
        mode_name = "MIXED RoPE (learnable per-head)" if rope_mixed else "AXIAL RoPE (standard)"
        print(f"Testing {mode_name}")
        print("="*80)
        
        # Create model
        model = RoPEAttention(dim=dim, num_heads=num_heads, rope_theta=10.0, rope_mixed=rope_mixed)
        
        # Show learnable parameters
        if rope_mixed:
            print(f"\n[LEARNABLE FREQS]")
            print(f"  freqs shape: {model.freqs.shape}")  # [2, num_heads * head_dim//2]
            print(f"  freqs is flattened: [2, num_heads * (head_dim//2)] = [2, {num_heads * (head_dim//2)}]")
            # Reshape to see per-head structure
            freqs_reshaped = model.freqs.view(2, num_heads, head_dim//2)
            print(f"  freqs reshaped: {freqs_reshaped.shape}")
            print(f"  freqs_x[head 0, first 4 dims]: {freqs_reshaped[0, 0, :4]}")
            print(f"  freqs_y[head 0, first 4 dims]: {freqs_reshaped[1, 0, :4]}")
            print(f"  freqs_x[head 1, first 4 dims]: {freqs_reshaped[0, 1, :4]}")
            print(f"  ↑ Each head has different frequencies (learnable)")
            print(f"  Registered buffers: freqs_t_x shape={model.freqs_t_x.shape}, freqs_t_y shape={model.freqs_t_y.shape}")
        else:
            print(f"\n[STATIC FREQS]")
            print(f"  freqs_cis shape: {model.freqs_cis.shape}")  # [196, head_dim]
            print(f"  freqs_cis dtype: {model.freqs_cis.dtype}")
        
        # Create input
        x = torch.randn(batch_size, seq_len, dim)
        print(f"\n[INPUT]")
        print(f"  x shape: {x.shape}")
        
        # Forward pass with intermediate outputs
        print(f"\n[FORWARD PASS]")
        B, N, C = x.shape
        qkv = model.qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        print(f"  QKV shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        print(f"    Format: [B, num_heads, seq_len, head_dim]")
        
        # Compute position embeddings
        w = h = math.sqrt(N - 1)
        print(f"\n[POSITION ENCODING]")
        print(f"  Spatial grid: {int(w)}x{int(h)}")
        
        if rope_mixed:
            t_x, t_y = model.freqs_t_x, model.freqs_t_y
            if model.freqs_t_x.shape[0] != N - 1:
                t_x, t_y = init_t_xy(end_x=int(w), end_y=int(h))
            print(f"  t_x shape: {t_x.shape}, sample: {t_x[:5]}")
            print(f"  t_y shape: {t_y.shape}, sample: {t_y[:5]}")
            
            freqs_cis = model.compute_cis(model.freqs, t_x, t_y)
            print(f"  freqs_cis shape after compute_mixed_cis: {freqs_cis.shape}")
            print(f"    Format: [num_heads, num_spatial_tokens, head_dim//2]")
        else:
            freqs_cis = model.freqs_cis
            if model.freqs_cis.shape[0] != N - 1:
                freqs_cis = model.compute_cis(end_x=int(w), end_y=int(h))
            print(f"  freqs_cis shape: {freqs_cis.shape}")
            print(f"    Format: [num_spatial_tokens, head_dim]")
        
        # Apply RoPE (only to spatial tokens, skip CLS)
        print(f"\n[ROPE APPLICATION]")
        print(f"  Before RoPE - q[:,:,1:] shape: {q[:, :, 1:].shape}")
        print(f"  Before RoPE - k[:,:,1:] shape: {k[:, :, 1:].shape}")
        print(f"  CLS token (position 0) is NOT rotated")
        
        q_spatial = q[:, :, 1:]
        k_spatial = k[:, :, 1:]
        q_rotated, k_rotated = apply_rotary_emb(q_spatial, k_spatial, freqs_cis=freqs_cis)
        
        print(f"  After RoPE - q_rotated shape: {q_rotated.shape}")
        print(f"  After RoPE - k_rotated shape: {k_rotated.shape}")
        
        # Check rotation effect
        rotation_diff_q = (q_spatial - q_rotated).abs().mean().item()
        rotation_diff_k = (k_spatial - k_rotated).abs().mean().item()
        print(f"  Rotation effect (mean abs diff) - Q: {rotation_diff_q:.6f}, K: {rotation_diff_k:.6f}")
        
        # Full forward
        output = model(x)
        print(f"\n[OUTPUT]")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: {x.shape} (should match input)")
        
        print(f"\n[SUMMARY]")
        if rope_mixed:
            print(f"  ✓ Each attention head has unique learnable frequency parameters")
            print(f"  ✓ Frequencies are trainable (requires_grad=True)")
            print(f"  ✓ More expressive but more parameters")
        else:
            print(f"  ✓ All heads share the same frequency basis")
            print(f"  ✓ Frequencies are fixed (standard RoPE)")
            print(f"  ✓ Fewer parameters, standard axial approach")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)
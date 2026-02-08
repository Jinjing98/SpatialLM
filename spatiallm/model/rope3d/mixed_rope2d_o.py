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
    # ORIGINAL
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)

    # JJ: TODO potentially more reasonable?
    # mag_JJ = 1 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # for i in range(num_heads):
    #     angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
    #     fx_JJ = mag_JJ * torch.cos(angles)
    #     fy_JJ = mag_JJ * torch.sin(angles)
    #     freqs_x.append(fx_JJ)
    #     freqs_y.append(fy_JJ)

    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    print('init freqs x: ', freqs_x.shape)
    print(freqs_x)
    print('init freqs y: ', freqs_y.shape)
    print(freqs_y)
    freqs = torch.stack([freqs_x, freqs_y], dim=0) # [2, num_heads, dim//4]
    print('freqs shape: ', freqs.shape)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    assert isinstance(end_x, int) and isinstance(end_y, int), f'end_x and end_y must be integers, got {end_x} and {end_y}'
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2) # [num_heads, N, dim//4]
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        print('mixed cis: ')
        print('freqs_x shape: ', freqs_x.shape)
        print('freqs_y shape: ', freqs_y.shape)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y) # [num_heads, N, dim//4]
        print('freqs_cis shape: ', freqs_cis.shape)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    print('axial cis: ')
    print('freqs_x shape: ', freqs_x.shape)
    print('freqs_y shape: ', freqs_y.shape)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    print('freqs_cis_x shape: ', freqs_cis_x.shape)
    print('freqs_cis_y shape: ', freqs_cis_y.shape)
    print('freqs_cis shape: ', torch.cat([freqs_cis_x, freqs_cis_y], dim=-1).shape)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
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
        print('Attention forward: ')
        print('x shape: ', x.shape)
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
    def __init__(self, *args, rope_theta=10.0, rope_mixed=True, img_w=14, img_h=14, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.rope_mixed = rope_mixed
        self.img_w = img_w
        self.img_h = img_h
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = init_2d_freqs(
                dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta, 
                rotate=True
            ).view(2, -1)
            print('Leanred param freqs shape: ', freqs.shape)
            self.freqs = nn.Parameter(freqs, requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x=self.img_w, end_y=self.img_h)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
            print('done with init mixed rope.')
        else:
            self.compute_cis = partial(compute_axial_cis, dim=self.dim // self.num_heads, theta=rope_theta)
            freqs_cis = self.compute_cis(end_x=self.img_w, end_y=self.img_h)
            self.freqs_cis = freqs_cis
    
    def forward(self, x, online_adapt_h_w=False, online_adapt_given_h=None):
        print('RoPEAttention forward: ')
        B, N, C = x.shape

        if N != self.img_w*self.img_h+1:
            assert online_adapt_h_w, f'enable online_adapt_h_w!: x shape varied from init shape, N: {N}, img_w: {self.img_w}, img_h: {self.img_h}'
            assert online_adapt_given_h is not None, f'online_adapt_given_h must be provided when online_adapt_h_w is True'
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ###### Apply rotary position embedding
        if online_adapt_h_w:
            # w = h = math.sqrt(x.shape[1] - 1)
            # w = int(w)
            h = int(online_adapt_given_h)
            w = int((x.shape[1] - 1) / h)
            print('Warning. Assuming square image. Adapted w and h: ', w, h)
        else:
            w = self.img_w
            h = self.img_h
            print('Using init w and h: ', w, h)

        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            print('t_x shape: ', t_x.shape)
            print('t_y shape: ', t_y.shape)
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                print('Oops, freqs_t_x shape[0] != x.shape[1] - 1', self.freqs_t_x.shape[0], x.shape[1] - 1)
                print('Reinit tx ty with w and h: ', w, h)
                t_x, t_y = init_t_xy(end_x=w, end_y=h)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
                print('Reinitialized t_x and t_y')
                print('t_x shape: ', t_x.shape)
                print('t_y shape: ', t_y.shape)
            print('freqs shape: ', self.freqs.shape)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            print('freqs_cis shape: ', freqs_cis.shape)
        else:
            freqs_cis = self.freqs_cis
            print('freqs_cis shape: ', freqs_cis.shape)
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                print('Oops, freqs_cis shape[0] != x.shape[1] - 1', self.freqs_cis.shape[0], x.shape[1] - 1)
                print('Recompute freqs_cis with new w and h: ', w, h)
                freqs_cis = self.compute_cis(end_x=w, end_y=h)
                print('New freqs_cis shape: ', freqs_cis.shape)
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

# JJ: below shows how axis differ from mixed given the same other setting.
# 1) mixed learned per heads
# 2) mixed spead over the head_dim
# 3) mixed learned freq

# mixed cis: 
# Leanred param freqs shape:  torch.Size([2, 16])
# freqs_x shape:  torch.Size([2, 196, 8])
# freqs_y shape:  torch.Size([2, 196, 8])
# freqs_cis shape:  torch.Size([2, 196, 8])
# freqs_cis shape:  torch.Size([2, 196, 8])
# freqs_cis before broadcast:  torch.Size([2, 196, 8])

# axial cis: 
# axial cis: 
# freqs_x shape:  torch.Size([196, 4])
# freqs_y shape:  torch.Size([196, 4])
# freqs_cis_x shape:  torch.Size([196, 4])
# freqs_cis_y shape:  torch.Size([196, 4])
# freqs_cis shape:  torch.Size([196, 8])




    head_dim = 16
    num_heads = 2
    dim = head_dim * num_heads
    rope_theta = 10.0
    rope_mixed = True
    # rope_mixed = False
    img_w = 14
    img_h = 14 #140
    bs = 3
    online_adapt_h_w = True
    x = torch.randn(bs, img_w*img_h + 1, dim)# B N C
    model = RoPEAttention(dim=dim, num_heads=num_heads, rope_theta=rope_theta, rope_mixed=rope_mixed)
    model(x, online_adapt_h_w=(img_w*img_h!=14**2), online_adapt_given_h=img_h)
    # print(model(x).shape)
# MixedRoPE3D: 核心设计与实现陷阱

## 概述

MixedRoPE3D 是一个为 Point Cloud VLM (Vision-Language Model) 设计的位置编码方案,能够同时处理:
- **3D 空间位置编码** (基于 x, y, z 坐标) 用于 point cloud tokens
- **1D 时序位置编码** (基于 token 顺序) 用于 text tokens 和 point tokens 的时序信息

---

## 核心设计

### 1. 整体架构

```
SpatialLMQwenForCausalLMMixedRoPE3D (最外层 - VLM 模型)
  ├─ forward_point_cloud()
  │   └─ 提取 point features + grid coordinates
  │
  ├─ forward()
  │   ├─ 插入 point tokens 到 text embeddings
  │   ├─ 创建 point_token_mask [B, seq_len] (标记哪些是 point tokens)
  │   └─ 收集 point_coords_list [N_tokens, 3] (grid coordinates)
  │
  └─ Qwen2ModelMixedRoPE3D (Transformer 模型)
      └─ MixedRoPE3DQwen2DecoderLayer × 24 layers
          └─ MixedRoPE3DQwen2Attention
              └─ _apply_mixed_rope()
                  ├─ Point tokens: 3D RoPE (spatial) + 1D RoPE (temporal)
                  └─ Text tokens: 1D RoPE (full head_dim)
```

### 2. 空间-时序分离策略 (Spatial-Temporal Separation)

**核心思想**: Point tokens 包含 4 种位置信息:
- `temporal_order`: 在整个序列中的位置
- `(x, y, z)`: 3D 空间坐标

为了同时编码这两种信息,`head_dim` 被分成两部分:

```python
# 默认策略: 'half_spatial_half_temp'
head_dim = 32  # Qwen2-7B

spatial_dim = head_dim // 2 = 16   # 上半部分: 用于 3D RoPE
temporal_dim = head_dim // 2 = 16  # 下半部分: 用于 1D RoPE
```

**应用方式**:

```python
# Point tokens
q_point = query_states[:, :, point_indices, :]  # [B, num_heads, N_point, head_dim]

# 分割 head_dim
q_point_spatial = q_point[:, :, :, :spatial_dim]     # [B, num_heads, N_point, 16]
q_point_temporal = q_point[:, :, :, spatial_dim:]    # [B, num_heads, N_point, 16]

# 分别应用 RoPE
q_point_spatial_rotated = apply_rotary_emb_3d(q_point_spatial, freqs_cis_3d, point_coords)
q_point_temporal_rotated = apply_rotary_emb_1d(q_point_temporal, position_embeddings)

# 合并
q_point_rotated = torch.cat([q_point_spatial_rotated, q_point_temporal_rotated], dim=-1)

# Text tokens: 全部使用 1D RoPE
q_text = query_states[:, :, text_indices, :]  # [B, num_heads, N_text, 32]
q_text_rotated = apply_rotary_emb_1d(q_text, position_embeddings)
```

**设计理由**:
- ✅ 同时保留空间和时序信息
- ✅ 避免信息冲突
- ✅ 可以独立优化两种位置编码
- ⚠️ TODO: 支持其他分配策略 (e.g., `full_spatial`, custom ratios)

### 3. Learned Frequencies with Axial Mixing

**三层可学习性**:

1. **Per-axis frequencies**: 每个轴 (x, y, z) 有独立的频率
2. **Per-head specialization**: 每个 attention head 有不同的频率
3. **Per-frequency-bin**: 每个频率 bin 有独立的参数

```python
# Base frequencies: [3 axes, num_heads, dim//2 freq_bins]
freqs_base_3d = torch.stack([
    1.0 / (theta ** (torch.arange(0, spatial_dim, 2) / spatial_dim))
    for _ in range(3)  # x, y, z
])  # [3, spatial_dim//2]

# Learned per-axis, per-head frequencies
freqs_3d = nn.Parameter(
    freqs_base_3d.unsqueeze(1).expand(3, num_heads, -1) 
    + torch.randn(3, num_heads, spatial_dim//2) * 0.01
)  # [3, 14, 16] for Query (14 heads)

# Learned axial mixing weights: [num_heads, dim//2, 3]
axial_weights = nn.Parameter(
    torch.ones(num_heads, spatial_dim//2, 3) 
    + torch.randn(num_heads, spatial_dim//2, 3) * 0.01
)  # [14, 16, 3]
```

**计算混合频率**:

```python
def compute_mixed_cis_3d(freqs_3d, point_coords, axial_weights):
    """
    Args:
        freqs_3d: [3, num_heads, dim//2] - learned frequencies
        point_coords: [N_tokens, 3] - (x, y, z) coordinates
        axial_weights: [num_heads, dim//2, 3] - mixing weights
    
    Returns:
        freqs_cis: [num_heads, N_tokens, dim//2] - complex frequencies
    """
    # Compute per-axis angle contributions
    freqs_x = freqs_3d[0] * point_coords[:, 0:1, None]  # [N, 1, 1] * [num_heads, dim//2]
    freqs_y = freqs_3d[1] * point_coords[:, 1:2, None]  # → [num_heads, N, dim//2]
    freqs_z = freqs_3d[2] * point_coords[:, 2:3, None]
    
    # Apply learned mixing weights
    w_x = axial_weights[:, :, 0].unsqueeze(1)  # [num_heads, 1, dim//2]
    w_y = axial_weights[:, :, 1].unsqueeze(1)
    w_z = axial_weights[:, :, 2].unsqueeze(1)
    
    # Weighted combination
    freqs_combined = w_x * freqs_x + w_y * freqs_y + w_z * freqs_z  # [num_heads, N, dim//2]
    
    # Convert to complex exponentials
    freqs_cis = torch.polar(torch.ones_like(freqs_combined), freqs_combined)
    return freqs_cis
```

### 4. GQA (Grouped Query Attention) 适配

**挑战**: Qwen2 使用 GQA
- `num_attention_heads` = 14 (Query heads)
- `num_key_value_heads` = 2 (Key/Value heads)

**解决方案**: 分别为 Q 和 KV 初始化 learned parameters

```python
class MixedRoPE3DQwen2Attention:
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        
        self.num_heads = config.num_attention_heads  # 14
        self.num_key_value_heads = config.num_key_value_heads  # 2
        
        # Query frequencies: [3, 14, 16]
        self.freqs_3d_q = nn.Parameter(
            freqs_base_3d.unsqueeze(1).expand(3, self.num_heads, -1) + ...
        )
        self.axial_weights_3d_q = nn.Parameter(
            torch.ones(self.num_heads, spatial_dim//2, 3) + ...
        )
        
        # Key/Value frequencies: [3, 2, 16]
        self.freqs_3d_kv = nn.Parameter(
            freqs_base_3d.unsqueeze(1).expand(3, self.num_key_value_heads, -1) + ...
        )
        self.axial_weights_3d_kv = nn.Parameter(
            torch.ones(self.num_key_value_heads, spatial_dim//2, 3) + ...
        )
    
    def _apply_mixed_rope(self, query_states, key_states, ...):
        # Compute separate freqs for Q and KV
        freqs_cis_3d_q = compute_mixed_cis_3d(
            self.freqs_3d_q, point_coords, self.axial_weights_3d_q
        )  # [14, N, 16]
        
        freqs_cis_3d_kv = compute_mixed_cis_3d(
            self.freqs_3d_kv, point_coords, self.axial_weights_3d_kv
        )  # [2, N, 16]
        
        # Apply separately
        q_rotated = apply_rotary_emb_3d(q_spatial, freqs_cis_3d_q)
        k_rotated = apply_rotary_emb_3d(k_spatial, freqs_cis_3d_kv)
```

### 5. Grid Coordinates for Sparse Encoders

**问题**: Sparse point cloud encoders 压缩点云
- Input: 106,328 个原始点 `[N_raw, 3]`
- Output: 556 个 encoded tokens `[N_tokens, D]`

**解决方案**: 使用 voxel grid 的中心坐标

```python
def forward_point_cloud(self, point_cloud, device, dtype):
    if self.config.point_backbone_type == PointBackboneType.SCENESCRIPT:
        # Sparse voxel processing
        sparse_output = self.point_backbone.sparse_resnet(pc_sparse_tensor)
        sparse_list = sparse_uncollate(sparse_output)
        
        # Extract grid coordinates: [N_tokens, 3]
        grid_coords = sparse_list[0].C.float()
        
        # Normalize to [0, 1]
        grid_coords = grid_coords / (self.point_backbone.reduced_grid_size - 1)
        
        # Get encoded features
        encoded_features = self.point_backbone.input_proj(...)
        
        return self.point_proj(encoded_features), grid_coords
    
    elif self.config.point_backbone_type == PointBackboneType.SONATA:
        point = self.point_backbone.enc(point)
        
        # Extract grid coordinates
        grid_coords = point["grid_coord"].float()  # [N_tokens, 3]
        
        context = point["sparse_conv_feat"].features
        return self.point_proj(context), grid_coords
```

---

## 关键陷阱与解决方案

### 🔴 陷阱 1: KV Cache 与 Attention Mask 维度不匹配

**问题描述**:

在 autoregressive generation 中存在两个阶段:

1. **Prefill 阶段** (第一次 forward):
   ```
   input_ids: [1, 231] text tokens
   → 插入 point tokens
   inputs_embeds: [1, 786] (231 - 2 + 556 + 1)
   attention_mask: [1, 786] ✓
   ```

2. **Generation 阶段** (后续每次生成 1 个 token):
   ```
   input_ids: [1, 1] (新生成的 token)
   attention_mask: [1, 232] (transformers 库自动更新,只记录 text tokens!)
   past_key_values: 786 tokens (包含 point tokens)
   
   结果:
   attn_weights: [1, 14, 1, 787] (1 query × 787 keys)
   attention_mask: [1, 1, 1, 232] 
   
   → RuntimeError: size mismatch 787 vs 232
   ```

**根本原因**: 
transformers 库的 `prepare_inputs_for_generation` 方法不知道我们在 prefill 阶段插入了额外的 point tokens,所以 `attention_mask` 只追踪原始的 text tokens。

**解决方案**: 在 `_prepare_decoder_attention_mask` 中检测并修复

```python
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values):
    batch_size, seq_length = input_shape
    
    # Get past sequence length from KV cache
    past_key_values_length = 0
    if past_key_values is not None:
        if hasattr(past_key_values, 'get_seq_length'):
            past_key_values_length = past_key_values.get_seq_length()
        elif isinstance(past_key_values, (list, tuple)):
            past_key_values_length = past_key_values[0][0].shape[2]
    
    # Fix length mismatch during generation
    if attention_mask is not None and past_key_values_length > 0:
        expected_length = past_key_values_length + seq_length  # 786 + 1 = 787
        actual_length = attention_mask.shape[1]  # 232
        
        if actual_length < expected_length:
            # Pad with 1s (can attend to point tokens)
            padding_length = expected_length - actual_length  # 555
            attention_mask = torch.nn.functional.pad(
                attention_mask, 
                (0, padding_length), 
                value=1
            )
            # Now: [1, 787] ✓
    
    # Continue with mask expansion...
```

**关键要点**:
- ✅ 填充值为 1 (可以 attend)
- ✅ 填充在末尾 (假设 point tokens 插入在中间位置,但 mask 只需要长度匹配)
- ⚠️ 这个假设在 prefill 阶段创建的 attention_mask 是正确的基础上成立

---

### 🔴 陷阱 2: Prefill vs Generation 的 RoPE 应用逻辑

**问题描述**:

Generation 阶段每次只生成 1 个新的 text token,不应该传递 `point_coords` 和 `point_token_mask`,否则会尝试对新 token 应用 3D RoPE。

**错误实现**:
```python
# ✗ 每次都传递 point_coords
outputs = self.model(
    inputs_embeds=inputs_embeds,
    point_coords=point_coords_list,
    point_token_mask=point_token_mask,
)
```

**正确实现**:
```python
# ✓ 根据 past_key_values 判断阶段
if past_key_values is None:
    # Prefill: 传递 point 相关参数
    model_point_coords = point_coords_list
    model_point_token_mask = point_token_mask
else:
    # Generation: 不传递,只用标准 1D RoPE
    model_point_coords = None
    model_point_token_mask = None

outputs = self.model(
    inputs_embeds=inputs_embeds,
    past_key_values=past_key_values,
    point_coords=model_point_coords,
    point_token_mask=model_point_token_mask,
)
```

**关键要点**:
- ✅ `past_key_values is None` → Prefill 阶段
- ✅ `past_key_values is not None` → Generation 阶段
- ✅ Generation 阶段新 token 只用标准 1D RoPE

---

### 🔴 陷阱 3: GQA 导致的 RoPE 维度错误

**问题描述**:

```python
# ✗ 错误: 用 Query heads 的 freqs 应用到 Key states
freqs_cis_3d = compute_mixed_cis_3d(
    freqs_3d,  # [3, 14, 16] - 14 query heads
    point_coords,
    axial_weights
)  # → [14, N, 16]

# Apply to both Q and K
q_rotated = apply_rotary_emb_3d(query_states, freqs_cis_3d)  # ✓ [B, 14, N, 16]
k_rotated = apply_rotary_emb_3d(key_states, freqs_cis_3d)    # ✗ [B, 2, N, 16]

# RuntimeError: shape [2, ...] vs [14, ...]
```

**解决方案**: 分别为 Q 和 KV 生成 frequencies

```python
# ✓ 正确: 分别处理
freqs_cis_3d_q = compute_mixed_cis_3d(
    self.freqs_3d_q,  # [3, 14, 16]
    point_coords,
    self.axial_weights_3d_q
)  # [14, N, 16]

freqs_cis_3d_kv = compute_mixed_cis_3d(
    self.freqs_3d_kv,  # [3, 2, 16]
    point_coords,
    self.axial_weights_3d_kv
)  # [2, N, 16]

q_rotated = apply_rotary_emb_3d(query_states, freqs_cis_3d_q)   # ✓ [B, 14, N, 16]
k_rotated = apply_rotary_emb_3d(key_states, freqs_cis_3d_kv)    # ✓ [B, 2, N, 16]
```

**关键要点**:
- ⚠️ GQA 中 `num_attention_heads` ≠ `num_key_value_heads`
- ✅ 必须分别初始化 Q 和 KV 的 learned parameters
- ✅ 在 `_apply_mixed_rope` 中分别应用

---

### 🔴 陷阱 4: Point Coordinates 数量与 Token 数量不匹配

**问题描述**:

```python
# ✗ 错误: 返回原始点云坐标
def forward_point_cloud(self, point_cloud, device, dtype):
    encoded_features = self.point_backbone(point_cloud)  # [556, D]
    raw_coords = point_cloud[:, :3]  # [106328, 3]
    return encoded_features, raw_coords  # 数量不匹配!

# 导致错误:
# attn_weights: [1, 14, 556, 556]
# freqs_cis_3d: [14, 106328, 16]
# RuntimeError: size mismatch
```

**解决方案**: 从 sparse encoder 提取 grid coordinates

```python
# ✓ 正确: 返回编码后 token 对应的坐标
def forward_point_cloud(self, point_cloud, device, dtype):
    sparse_output = self.point_backbone.sparse_resnet(pc_sparse_tensor)
    sparse_list = sparse_uncollate(sparse_output)
    
    # Extract grid coordinates for encoded tokens
    grid_coords = sparse_list[0].C.float()  # [556, 3] ✓
    
    encoded_features = self.point_backbone.input_proj(...)  # [556, D]
    
    return encoded_features, grid_coords
```

**关键要点**:
- ✅ `grid_coords.shape[0]` 必须等于 `encoded_features.shape[0]`
- ✅ Grid coordinates 表示每个 token 对应的 voxel 中心
- ⚠️ 需要归一化到合理范围 (e.g., [0, 1] 或 [-1, 1])

---

### 🔴 陷阱 5: 父类方法调用导致卡死

**问题描述**:

```python
# ✗ 尝试调用父类的 attention mask 处理方法
if attention_mask is not None:
    attention_mask = super()._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, 
        past_key_values, output_attentions
    )

# 结果: 程序卡死,没有任何输出
```

**原因**:
- `super()` 返回的代理对象,方法签名可能不兼容
- transformers 不同版本的 API 变化
- 可能触发了无限递归或其他未知行为

**解决方案**: 完全自己实现 mask 处理逻辑

```python
# ✓ 正确: 自己实现完整逻辑
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, ...):
    # 1. Get past length
    # 2. Fix length mismatch
    # 3. Create causal mask
    # 4. Expand attention mask
    # 5. Combine masks
    return combined_attention_mask
```

**关键要点**:
- ✅ 不依赖父类的 mask 处理方法
- ✅ 完全控制 mask 的生成和处理流程
- ⚠️ 需要仔细实现 causal mask 的逻辑

---

### 🔴 陷阱 6: Batch Dimension 处理不一致

**问题描述**:

`point_coords` 可能有不同的 shape:
- Prefill 单样本: `[N_tokens, 3]`
- Prefill 多样本: `[B, N_tokens, 3]`

**解决方案**: 统一处理

```python
def _apply_mixed_rope(self, query_states, key_states, position_embeddings,
                     point_coords, point_token_mask):
    # Ensure 3D: [B, N_tokens, 3]
    if point_coords.dim() == 2:
        point_coords = point_coords.unsqueeze(0)
    
    # Extract point indices from mask
    point_indices = point_token_mask[0].nonzero(as_tuple=True)[0]
    
    # Get coordinates for this batch
    if point_coords.shape[0] == 1:
        curr_point_coords = point_coords[0, :, :]  # [N_point, 3]
    else:
        curr_point_coords = point_coords[0, :, :]  # TODO: handle batch
    
    # Now process...
```

---

## Hyperparameters

```python
# 3D RoPE Configuration
rope_theta_3d: float = 100.0
    # Base frequency for 3D RoPE
    # Lower values → slower position decay
    # Recommended: 100 for point clouds

spatial_temporal_separate_strategy: str = 'half_spatial_half_temp'
    # How to split head_dim between spatial and temporal
    # Options: 'half_spatial_half_temp', 'full_spatial', 'full_temporal', ...

# Learned Frequencies
rope_mixed_3d: bool = True
    # Whether to use learned frequencies (vs fixed)

mixedRoPE_3d_learned_per_axis: bool = True
    # Whether each axis (x,y,z) has independent learned frequencies

mixedRoPE_3d_learned_axial_mixing_weight: bool = True
    # Whether the mixing weights between axes are learned
    # If False: fixed 1:1:1 mixing
    # If True: learned per-head, per-bin weights [num_heads, dim//2, 3]
```

---

## Testing & Debugging

### 关键 Debug Points

1. **Prefill 阶段**:
   ```python
   print(f'input_ids: {input_ids.shape}')
   print(f'inputs_embeds after insertion: {inputs_embeds.shape}')
   print(f'attention_mask: {attention_mask.shape}')
   print(f'point_token_mask: {point_token_mask.shape}, num True: {point_token_mask.sum()}')
   print(f'point_coords: {point_coords.shape}')
   ```

2. **Generation 阶段**:
   ```python
   print(f'past_key_values_length: {past_key_values_length}')
   print(f'attention_mask before fix: {attention_mask.shape}')
   print(f'attention_mask after fix: {attention_mask.shape}')
   print(f'attn_weights: {attn_weights.shape}')
   ```

3. **RoPE 应用**:
   ```python
   print(f'freqs_cis_3d_q: {freqs_cis_3d_q.shape}')
   print(f'freqs_cis_3d_kv: {freqs_cis_3d_kv.shape}')
   print(f'q_point_spatial: {q_point_spatial.shape}')
   print(f'k_point_spatial: {k_point_spatial.shape}')
   ```

### 常见错误信号

| 错误 | 可能原因 | 检查点 |
|------|----------|--------|
| `RuntimeError: size mismatch ... 787 vs 232` | KV cache attention mask 不匹配 | `_prepare_decoder_attention_mask` |
| `RuntimeError: size mismatch ... 14 vs 2` | GQA freqs 未分离 | `freqs_3d_q` vs `freqs_3d_kv` |
| `RuntimeError: size mismatch ... 556 vs 106328` | Point coords 数量错误 | `forward_point_cloud` 返回值 |
| 程序卡死无输出 | 父类方法调用问题 | 移除 `super()._update_causal_mask` |
| `AssertionError: shape mismatch` | Batch dimension 不一致 | `point_coords.dim()` 检查 |

---

## Future Improvements

1. **更多分离策略**:
   - [ ] `full_spatial`: 全部用于 3D RoPE
   - [ ] `custom_ratio`: 可配置比例 (e.g., 70% spatial, 30% temporal)
   - [ ] `adaptive`: 根据 task 动态调整

2. **更好的 Grid Coordinates**:
   - [ ] 支持非均匀 voxel grid
   - [ ] 考虑 local neighborhood 信息
   - [ ] Multi-scale grid coordinates

3. **性能优化**:
   - [ ] 缓存 `freqs_cis` 计算结果
   - [ ] Flash Attention 集成
   - [ ] Mixed precision 优化

4. **Attention Mask 改进**:
   - [ ] 更智能的 point token position tracking
   - [ ] 支持动态插入/删除 tokens
   - [ ] Per-token attention control

---

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)

---

## Changelog

- **2026-02-07**: Initial implementation
  - Basic 3D RoPE with spatial-temporal separation
  - GQA adaptation
  - KV cache fix for generation
  - Learned frequencies with axial mixing

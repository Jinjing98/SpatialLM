# Positional Encoding Methods Comparison

**Last Updated**: 2026-01-24  
**Status**: ✅ Complete Reference

---

## 📊 Quick Comparison Table

| VLM_PE | Attention Class | Custom Attention? | Where PE Applied | PCD_PE_Merge_Rule |
|--------|----------------|-------------------|------------------|-------------------|
| `None` (default) | `Qwen2Attention`<br>`LlamaAttention` | ❌ No | Standard 1D RoPE | N/A |
| `"CCA_2DProj"` | `Qwen2Attention`<br>`LlamaAttention` | ❌ No | 2D projection + CCA mask | N/A |
| `"3D_RoPE"` | `SpatialQwen2Attention` ✅<br>`SpatialLlamaAttention` ✅ | ✅ **Yes** | Custom attention forward | ✅ Supported |
| `"3D_Sinusoidal"` | `Qwen2Attention`<br>`LlamaAttention` | ❌ No | Added to embeddings | ✅ Supported |

---

## 🔍 Detailed Breakdown

### 1. `VLM_PE=None` (Default SpatialLM)

**Attention:** 
- Qwen2: `Qwen2Attention` (original from transformers)
- Llama: `LlamaAttention` (original from transformers)

**Positional Encoding:**
- All tokens (text + point cloud) use standard **1D RoPE**
- Applied to sequence positions only (temporal)
- No spatial (X, Y, Z) information encoded

**Implementation:**
```python
# No modifications - 100% original SpatialLM
model.layers[0].self_attn  # → Qwen2Attention (original)
```

**When to use:**
- Baseline comparison
- When 3D spatial PE is not needed
- Original SpatialLM behavior

**Files modified:** None

---

### 2. `VLM_PE="CCA_2DProj"` (Concentric Causal Attention)

**Attention:** 
- Qwen2: `Qwen2Attention` (original)
- Llama: `LlamaAttention` (original)

**Positional Encoding:**
- Point cloud features projected to **2D BEV (Bird's Eye View)**
- **Concentric Causal Attention** masking applied
  - Tokens can only attend to points in same or inner circles
  - Enforces spatial locality in 2D
- Still uses standard 1D RoPE in attention

**Implementation:**
```python
# 2D projection in model forward
if self.vlm_pe == "CCA_2DProj":
    from .cca_utils import project_to_2d, create_cca_mask
    point_features_2d = project_to_2d(point_features, patch_coords)
    attention_mask = create_cca_mask(patch_coords, num_circles=8)

# Standard attention with custom mask
model.layers[0].self_attn  # → Qwen2Attention (original)
```

**When to use:**
- Enforce spatial locality in 2D
- Scene layout understanding
- Computationally lighter than 3D methods

**Files used:**
- `spatiallm/model/cca_utils.py` - CCA masking logic
- No custom attention classes

---

### 3. `VLM_PE="3D_RoPE"` ⭐ (3D Rotary Position Embedding)

**Attention:** 
- Qwen2: `SpatialQwen2Attention` ✅ (custom)
- Llama: `SpatialLlamaAttention` ✅ (custom)

**Positional Encoding:**
- Point tokens: **3D_RoPE** (rotates Q/K based on X, Y, Z)
- Text tokens: Standard 1D RoPE
- Applied **per-head** (all heads share same spatial pattern)
- Supports two merge rules:
  - `"3D_only"`: Pure 3D spatial (21-21-22 split)
  - `"3D_with_1D"`: Hybrid spatial+temporal (32-10-10-12 split)

**Implementation:**
```python
# Custom attention classes
class SpatialQwen2Attention(Qwen2Attention):
    def forward(self, ...):
        # 1. Apply 3D_RoPE to point tokens
        # 2. Save 3D results
        # 3. Apply 1D_RoPE to all tokens
        # 4. Restore 3D_RoPE for points

# Attention layers replaced at model init
for layer in model.layers:
    layer.self_attn = SpatialQwen2Attention(...)
```

**Command line:**
```bash
python inference.py \
    --VLM_PE "3D_RoPE" \
    --PCD_PE_Merge_Rule "3D_only"
```

**When to use:**
- Full 3D spatial understanding
- Relative position encoding in 3D space
- Best for geometric reasoning

**Files used:**
- `spatiallm/model/spatial_attention.py` - Custom attention classes
- `spatiallm/model/volumetric_pe.py` - RotaryEmbedding3D class

**See:** [3D_ROPE.md](./3D_ROPE.md) for complete guide

---

### 4. `VLM_PE="3D_Sinusoidal"` (3D Sinusoidal Position Embedding)

**Attention:** 
- Qwen2: `Qwen2Attention` (original)
- Llama: `LlamaAttention` (original)

**Positional Encoding:**
- Point tokens: **3D sinusoidal PE** added to embeddings
- Text tokens: Keep original embeddings
- Applied to **full hidden_size** (896 dims)
- Each head gets different spatial dimensions (no sharing)
- Supports two merge rules:
  - `"3D_only"`: Pure 3D spatial (298-298-300 split)
  - `"3D_with_1D"`: Hybrid spatial+temporal (448-148-148-152 split)

**Implementation:**
```python
# PE computed and added to embeddings
if self.vlm_pe == "3D_Sinusoidal":
    pe_3d = compute_3d_sinusoidal_pe(coords_3d=patch_coords, ...)
    inputs_embeds[point_positions] += pe_3d

# Standard attention (no modification)
model.layers[0].self_attn  # → Qwen2Attention (original)
```

**Command line:**
```bash
python inference.py \
    --VLM_PE "3D_Sinusoidal" \
    --PCD_PE_Merge_Rule "3D_only"
```

**When to use:**
- Additive PE (simpler than rotary)
- When custom attention is not desired
- Per-head diversity in spatial features

**Files used:**
- `spatiallm/model/volumetric_pe.py` - compute_3d_sinusoidal_pe()
- No custom attention classes

**See:** [3D_SINUSOIDAL.md](./3D_SINUSOIDAL.md) for complete guide

---

## 🎯 Key Differences

### Method Type

| Method | Type | Mechanism |
|--------|------|-----------|
| None | Temporal | 1D sequence position |
| CCA_2DProj | Spatial (2D) | BEV projection + masking |
| 3D_RoPE | Spatial (3D) | Rotary Q/K transformation |
| 3D_Sinusoidal | Spatial (3D) | Additive embedding |

### Computational Cost

| Method | Cost | Notes |
|--------|------|-------|
| None | Baseline | Standard RoPE |
| CCA_2DProj | Low | 2D projection + mask creation |
| 3D_RoPE | Medium | Custom attention + save-restore |
| 3D_Sinusoidal | Low | One-time PE computation |

### Complexity

| Method | Implementation | Custom Attention |
|--------|----------------|------------------|
| None | Simple | ❌ No |
| CCA_2DProj | Moderate | ❌ No |
| 3D_RoPE | Complex | ✅ Yes |
| 3D_Sinusoidal | Simple | ❌ No |

### Head Behavior

| Method | Per-Head Pattern | Head Diversity |
|--------|------------------|----------------|
| None | All heads identical | Low |
| CCA_2DProj | All heads see 2D mask | Low |
| 3D_RoPE | All heads share 3D pattern | Low |
| 3D_Sinusoidal | Each head different dims | High |

---

## 📁 File Organization

### Custom Attention Classes
**File:** `spatiallm/model/spatial_attention.py`
- `SpatialQwen2Attention` - Used by 3D_RoPE only
- `SpatialLlamaAttention` - Used by 3D_RoPE only

### Positional Encoding Functions
**File:** `spatiallm/model/volumetric_pe.py`
- `RotaryEmbedding3D` - Used by 3D_RoPE
- `apply_rotary_pos_emb_3d()` - Used by 3D_RoPE
- `compute_3d_sinusoidal_pe()` - Used by 3D_Sinusoidal

### CCA Utilities
**File:** `spatiallm/model/cca_utils.py`
- `project_to_2d()` - Used by CCA_2DProj
- `create_cca_mask()` - Used by CCA_2DProj

### Main Model Files
**Files:** 
- `spatiallm/model/spatiallm_qwen.py` - Qwen2 model
- `spatiallm/model/spatiallm_llama.py` - Llama model

**Logic:**
```python
if vlm_pe == "3D_RoPE":
    # Replace attention layers
    from .spatial_attention import SpatialQwen2Attention
    for layer in self.model.layers:
        layer.self_attn = SpatialQwen2Attention(...)
elif vlm_pe == "3D_Sinusoidal":
    # Add PE to embeddings
    pe_3d = compute_3d_sinusoidal_pe(...)
    inputs_embeds[point_positions] += pe_3d
elif vlm_pe == "CCA_2DProj":
    # Apply CCA masking
    attention_mask = create_cca_mask(...)
# else: use standard attention (None)
```

---

## 🔧 PCD_PE_Merge_Rule Support

Only **3D_RoPE** and **3D_Sinusoidal** support `PCD_PE_Merge_Rule`:

### For 3D_RoPE (head_dim=64)

| Mode | 1D | X | Y | Z | Description |
|------|----|----|----|----|-------------|
| `3D_only` | 0 | 21 | 21 | 22 | Pure 3D spatial |
| `3D_with_1D` | 32 | 10 | 10 | 12 | Hybrid spatial+temporal |

### For 3D_Sinusoidal (hidden_size=896)

| Mode | 1D | X | Y | Z | Description |
|------|----|----|----|----|-------------|
| `3D_only` | 0 | 298 | 298 | 300 | Pure 3D spatial |
| `3D_with_1D` | 448 | 148 | 148 | 152 | Hybrid spatial+temporal |

**Note:** `None` and `CCA_2DProj` do not use `PCD_PE_Merge_Rule`.

---

## 🚀 Usage Examples

### Test All Methods

```bash
# 1. Baseline (original SpatialLM)
python inference.py --point_cloud scene.ply

# 2. CCA with 2D projection
python inference.py --VLM_PE "CCA_2DProj" --point_cloud scene.ply

# 3. 3D_RoPE (3D_only)
python inference.py --VLM_PE "3D_RoPE" --PCD_PE_Merge_Rule "3D_only" --point_cloud scene.ply

# 4. 3D_RoPE (3D_with_1D)
python inference.py --VLM_PE "3D_RoPE" --PCD_PE_Merge_Rule "3D_with_1D" --point_cloud scene.ply

# 5. 3D_Sinusoidal (3D_only)
python inference.py --VLM_PE "3D_Sinusoidal" --PCD_PE_Merge_Rule "3D_only" --point_cloud scene.ply

# 6. 3D_Sinusoidal (3D_with_1D)
python inference.py --VLM_PE "3D_Sinusoidal" --PCD_PE_Merge_Rule "3D_with_1D" --point_cloud scene.ply
```

---

## 🎓 When to Use Each Method

### Use `VLM_PE=None` when:
- ✅ Baseline comparison needed
- ✅ No spatial PE required
- ✅ Reproducing original SpatialLM results

### Use `VLM_PE="CCA_2DProj"` when:
- ✅ 2D spatial understanding sufficient (e.g., floor plans)
- ✅ Want spatial locality enforcement
- ✅ Lower computational cost preferred

### Use `VLM_PE="3D_RoPE"` when:
- ✅ Full 3D geometric reasoning needed
- ✅ Relative position encoding important
- ✅ Willing to use custom attention
- ✅ Best performance for 3D tasks

### Use `VLM_PE="3D_Sinusoidal"` when:
- ✅ 3D spatial PE desired
- ✅ Want to avoid custom attention
- ✅ Prefer additive over rotary PE
- ✅ Want per-head diversity

---

## 📚 Related Documentation

- [3D_ROPE.md](./3D_ROPE.md) - Complete 3D RoPE guide
- [3D_SINUSOIDAL.md](./3D_SINUSOIDAL.md) - Complete 3D Sinusoidal guide
- [README.md](./README.md) - Main SpatialLM documentation

---

## 🔍 Quick Decision Tree

```
Do you need spatial PE?
│
├─ No → Use VLM_PE=None
│
└─ Yes → Is 2D sufficient?
    │
    ├─ Yes → Use VLM_PE="CCA_2DProj"
    │
    └─ No (need 3D) → Do you want rotary or additive?
        │
        ├─ Rotary (better for relative positions)
        │   └─ Use VLM_PE="3D_RoPE"
        │       ├─ PCD_PE_Merge_Rule="3D_only" (pure spatial)
        │       └─ PCD_PE_Merge_Rule="3D_with_1D" (hybrid)
        │
        └─ Additive (simpler, no custom attention)
            └─ Use VLM_PE="3D_Sinusoidal"
                ├─ PCD_PE_Merge_Rule="3D_only" (pure spatial)
                └─ PCD_PE_Merge_Rule="3D_with_1D" (hybrid)
```

---

**Last Updated:** 2026-01-24  
**Maintained by:** SpatialLM Team

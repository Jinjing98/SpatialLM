# 3D Sinusoidal Position Embedding - Complete Guide

**Implementation Date**: 2026-01-24  
**Status**: ✅ Production Ready

---

## 📋 Quick Start

### Command Line
```bash
python inference.py \
    --VLM_PE "3D_Sinusoidal" \
    --PCD_PE_Merge_Rule "3D_only" \
    --point_cloud scene.ply
```

### Jupyter Notebook
```python
# Cell 2: Set verbose flag (optional)
os.environ["SPATIALLM_VERBOSE"] = "1"  # "0" or "1"

# Cell 4: Set parameters
VLM_PE = '3D_Sinusoidal'
PCD_PE_Merge_Rule = '3D_only'  # or '3D_with_1D'

# Cell 6: Load model
config.VLM_PE = VLM_PE
config.PCD_PE_Merge_Rule = PCD_PE_Merge_Rule
```

---

## 🎯 What is 3D_Sinusoidal?

Additive 3D positional encoding using sinusoidal functions for X, Y, Z coordinates. Unlike 3D_RoPE (which rotates Q/K), this directly adds position information to token embeddings.

**Key features:**
- Applied to **full `hidden_size`** (896 dims) before attention
- Each head gets **different spatial dimensions** (no sharing)
- **Additive** (not rotary) - simpler than RoPE
- Uses standard attention (no custom classes)

---

## 🔧 Architecture

### Dimension Allocation

**Two modes controlled by `PCD_PE_Merge_Rule`:**

#### 1. `"3D_only"` (Default) - Pure 3D Spatial
```
hidden_size = 896
├─ X: 298 dims  (33%)
├─ Y: 298 dims  (33%)
└─ Z: 300 dims  (34%)
```
Point tokens get **only 3D_Sinusoidal**, text tokens keep standard embeddings.

#### 2. `"3D_with_1D"` - Hybrid Spatial + Temporal
```
hidden_size = 896
├─ 1D: 448 dims  (50%) - Sequence position
├─ X:  148 dims  (17%)
├─ Y:  148 dims  (17%)
└─ Z:  152 dims  (16%)
```
Point tokens get **both 3D_Sinusoidal + 1D positional encoding**.

### Sinusoidal Formula

For each dimension (X, Y, or Z) with `d` allocated dimensions:

```python
for i in range(d // 2):
    freq = 1.0 / (base ** (2 * i / d))
    pe[..., 2*i] = sin(coord * freq)      # Even indices
    pe[..., 2*i+1] = cos(coord * freq)    # Odd indices
```

**Properties:**
- Higher frequencies capture fine-grained positions
- Lower frequencies capture coarse positions
- Smooth, continuous encoding

### Application Flow

```
┌─────────────────┐
│ Point Cloud     │  N points × 3 coords (X,Y,Z)
│ Patch Coords    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Normalize       │  Preserve aspect ratio
│ Coordinates     │  max_range = max(X_range, Y_range, Z_range)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Compute 3D PE   │  [N, hidden_size=896]
│ Sinusoidal(X,Y,Z)│  sin/cos at different frequencies
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Add to          │  point_embeddings += pe_3d
│ Point Tokens    │  [N, hidden_size]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Standard        │  No custom attention needed
│ Attention       │
└─────────────────┘
```

---

## 🔬 Critical Implementation Details

### 1. Coordinate Normalization
**Same as 3D_RoPE** - Preserves aspect ratio using global scale:
```python
max_range = (coord_max - coord_min).max()  # Single scalar
normalized = (coords - coord_min) / max_range
```

### 2. No Custom Attention Required
Unlike 3D_RoPE, 3D_Sinusoidal uses **standard attention**:
- No need for `SpatialQwen2Attention` or `SpatialLlamaAttention`
- Simpler implementation
- PE is added during embedding, not during attention

### 3. Applied Once During Prefill
```python
# During forward_point_cloud()
if self.vlm_pe == "3D_Sinusoidal":
    # Compute 3D sinusoidal PE
    pe_3d = compute_3d_sinusoidal_pe(
        coords_3d=patch_coords,
        hidden_size=896,
        merge_rule=self.pcd_pe_merge_rule_sinusoidal,
        position_ids=position_ids  # For 3D_with_1D mode
    )
    # Store for later use
    self._point_3d_sinusoidal_pe = pe_3d
```

Then added to point embeddings:
```python
# In forward()
if self._point_3d_sinusoidal_pe is not None:
    inputs_embeds[point_positions] += self._point_3d_sinusoidal_pe
```

### 4. Per-Head Diversity
**Key difference from 3D_RoPE:**
- 3D_RoPE: All heads share same spatial pattern (applied per-head)
- 3D_Sinusoidal: Each head gets different dims → different spatial features

Example for `hidden_size=896`, `num_heads=14`:
- Head 0 gets dims [0:64] → mostly X information
- Head 7 gets dims [448:512] → mostly Y information  
- Head 13 gets dims [832:896] → mostly Z information

This allows different heads to specialize in different spatial aspects.

---

## 📊 Comparison: 3D_RoPE vs 3D_Sinusoidal

| Aspect | 3D_RoPE | 3D_Sinusoidal |
|--------|---------|---------------|
| **Method** | Rotary (multiply Q/K) | Additive (add to embeddings) |
| **Complexity** | High (custom attention) | Low (standard attention) |
| **Application** | Per-head (head_dim=64) | Full hidden_size (896) |
| **Head Sharing** | All heads share pattern | Each head gets different dims |
| **Custom Attention** | Required | Not required |
| **Performance** | May be better for relative positions | Simpler, faster |
| **Dimension Split** | 21-21-22 or 32-10-10-12 | 298-298-300 or 448-148-148-152 |

---

## 🐛 Critical Implementation Notes

### 1. Consistent with 3D_RoPE Design
The implementation follows the same design patterns as 3D_RoPE:
- Same `PCD_PE_Merge_Rule` parameter
- Same coordinate normalization strategy
- Same prefill/decoding behavior
- Same verbose debugging support

### 2. No Head Sharing
Each attention head processes different dimensions of the 896-dim embedding, so they naturally get different spatial information:
```python
# hidden_size=896, num_heads=14, head_dim=64
# Head 0: dims [0:64]    → portions of X,Y,Z encoding
# Head 1: dims [64:128]  → different portions of X,Y,Z
# ...
# Head 13: dims [832:896] → different portions again
```

### 3. Merge Rule Implementation
**`3D_only` mode:**
```python
d_x = 298  # ~33% for X
d_y = 298  # ~33% for Y
d_z = 300  # ~34% for Z (gets remainder)
total = 896
```

**`3D_with_1D` mode:**
```python
d_1d = 448  # 50% for temporal/sequence
d_x = 148   # ~17% for X
d_y = 148   # ~17% for Y  
d_z = 152   # ~16% for Z
total = 896
```

---

## 📁 Files Modified

### Core Implementation
- `spatiallm/model/spatiallm_qwen.py` - Main model
- `spatiallm/model/spatiallm_llama.py` - Main model
- `spatiallm/model/volumetric_pe.py` - `compute_3d_sinusoidal_pe()` function

### Entry Points
- `inference.py` - Command-line interface
- `scripts/inference.sh` - Example script
- `scripts/test_notebook.ipynb` - Jupyter notebook

**Note:** No changes to `spatial_attention.py` - uses standard attention!

---

## ✅ Validation

### Test Cases
1. **3D_Sinusoidal + 3D_only**: Pure 3D spatial encoding
2. **3D_Sinusoidal + 3D_with_1D**: Hybrid spatial + temporal encoding

### Expected Output
```
[Config] VLM_PE = 3D_Sinusoidal (from command line)
[Config] PCD_PE_Merge_Rule = 3D_only
[3D_Sinusoidal] Initialized with PCD_PE_Merge_Rule=3D_only
[3D PE] Original patch_coords range: X=[0.069, 6.667], Y=[0.069, 5.304], Z=[0.154, 3.005]
[3D PE] Normalized with max_range=6.598
[3D_Sinusoidal] Computing PE for 556 patches with merge_rule=3D_only
[3D_Sinusoidal] Computed PE shape: torch.Size([556, 896])
```

No attention checkpoints (uses standard attention).

---

## 🚨 Common Issues

### Issue: Quality worse than 3D_RoPE
**Possible causes:**
1. Additive PE may be less effective than rotary for relative positions
2. Try adjusting `base` parameter (default: 10000)
3. Try different `PCD_PE_Merge_Rule`

### Issue: NaN or inf in PE
**Cause:** Coordinates not properly normalized  
**Fix:** Check coordinate normalization in verbose mode

### Issue: No PE applied
**Cause:** `VLM_PE` not set correctly  
**Fix:** Verify `[Config] VLM_PE = 3D_Sinusoidal` in output

---

## 🔗 See Also

- [3D_ROPE.md](./3D_ROPE.md) - Alternative 3D PE method (rotary)
- [README.md](./README.md) - General SpatialLM documentation

---

## 📝 Implementation Reference

### Compute 3D Sinusoidal PE (Simplified)

```python
def compute_3d_sinusoidal_pe(coords_3d, hidden_size, merge_rule="3D_only", base=10000):
    """
    coords_3d: [N, 3] - normalized X,Y,Z coordinates
    hidden_size: 896
    merge_rule: "3D_only" or "3D_with_1D"
    
    Returns: [N, hidden_size] positional encoding
    """
    # Allocate dimensions
    if merge_rule == "3D_only":
        d_x, d_y, d_z = 298, 298, 300
    else:  # "3D_with_1D"
        d_1d, d_x, d_y, d_z = 448, 148, 148, 152
    
    # Compute PE for each dimension
    pe_x = sinusoidal_encoding(coords_3d[:, 0], d_x, base)  # [N, d_x]
    pe_y = sinusoidal_encoding(coords_3d[:, 1], d_y, base)  # [N, d_y]
    pe_z = sinusoidal_encoding(coords_3d[:, 2], d_z, base)  # [N, d_z]
    
    # Concatenate
    if merge_rule == "3D_only":
        pe = torch.cat([pe_x, pe_y, pe_z], dim=-1)  # [N, 896]
    else:
        pe_1d = sinusoidal_encoding(position_ids, d_1d, base)
        pe = torch.cat([pe_1d, pe_x, pe_y, pe_z], dim=-1)
    
    return pe

def sinusoidal_encoding(positions, d_model, base=10000):
    """Standard sinusoidal encoding for a single dimension"""
    pe = torch.zeros(len(positions), d_model)
    for i in range(d_model // 2):
        freq = 1.0 / (base ** (2 * i / d_model))
        pe[:, 2*i] = torch.sin(positions * freq)
        pe[:, 2*i+1] = torch.cos(positions * freq)
    return pe
```

# 3D Rotary Position Embedding (3D_RoPE) - Complete Guide

**Implementation Date**: 2026-01-24  
**Status**: ✅ Production Ready

---

## 📋 Quick Start

### Command Line
```bash
python inference.py \
    --VLM_PE "3D_RoPE" \
    --PCD_PE_Merge_Rule "3D_only" \
    --point_cloud scene.ply
```

### Jupyter Notebook
```python
# Cell 2: Set verbose flag (optional)
os.environ["SPATIALLM_VERBOSE"] = "1"  # "0" or "1"

# Cell 4: Set parameters
VLM_PE = '3D_RoPE'
PCD_PE_Merge_Rule = '3D_only'  # or '3D_with_1D'

# Cell 6: Load model
config.VLM_PE = VLM_PE
config.PCD_PE_Merge_Rule = PCD_PE_Merge_Rule
```

---

## 🎯 What is 3D_RoPE?

Extends standard RoPE (1D rotary position embedding) to 3D coordinates (X, Y, Z) for point cloud tokens, allowing the model to understand spatial relationships in 3D scenes.

**Key features:**
- Applied **per-head** (on `head_dim=64`) after Q/K projection
- All attention heads **share the same 3D spatial pattern**
- Works with **GQA** (Grouped Query Attention)
- Only applied during **prefill stage** (cached for decoding)

---

## 🔧 Architecture

### Dimension Allocation

**Two modes controlled by `PCD_PE_Merge_Rule`:**

#### 1. `"3D_only"` (Default) - Pure 3D Spatial
```
head_dim = 64
├─ X: 21 dims  (33%)
├─ Y: 21 dims  (33%)
└─ Z: 22 dims  (34%)
```
Point tokens get **only 3D_RoPE**, text tokens get standard 1D_RoPE.

#### 2. `"3D_with_1D"` - Hybrid Spatial + Temporal
```
head_dim = 64
├─ 1D: 32 dims  (50%) - Sequence position
├─ X:  10 dims  (16%)
├─ Y:  10 dims  (16%)
└─ Z:  12 dims  (18%)
```
Point tokens get **both 3D_RoPE + 1D_RoPE**, text tokens get standard 1D_RoPE.

### Application Flow

```
┌─────────────────┐
│ Q/K Projection  │  [bsz, seq_len, hidden_size]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reshape Heads   │  [bsz, num_heads, seq_len, head_dim]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Apply 3D_RoPE   │  Point tokens only (per-head)
│ to Point Tokens │  [bsz, num_heads, N_point, head_dim]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Apply 1D_RoPE   │  All tokens (standard RoPE)
│ to ALL Tokens   │  [bsz, num_heads, seq_len, head_dim]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Restore 3D_RoPE │  Overwrite point tokens
│ for Points      │  (save → apply 1D → restore)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Attention       │  Text: 1D_RoPE, Points: 3D_RoPE
└─────────────────┘
```

---

## 🔬 Critical Implementation Details

### 1. Coordinate Normalization
**Preserves aspect ratio** using global scale:
```python
# Always normalize with max range across all dims
max_range = (coord_max - coord_min).max()  # Single scalar
normalized = (coords - coord_min) / max_range

# Example: Room (8.5m × 5.2m × 2.8m)
# After: X∈[0, 1.0], Y∈[0, 0.61], Z∈[0, 0.33]  ✅ Aspect ratio preserved
```

**Why?** Per-dimension normalization distorts geometry (a long hallway becomes a cube).

### 2. Prefill vs Decoding
- **Prefill (first forward pass):**
  - Compute 3D_RoPE from `patch_coords`
  - Set `_spatial_3d_rope_data` on all attention layers
  - Apply to point tokens in sequence
  
- **Decoding (subsequent passes):**
  - Use cached K/V (already have 3D_RoPE)
  - New tokens (text only) get standard 1D_RoPE
  - No 3D_RoPE needed

**Detection:** Check if `past_key_value.get_seq_length(layer_idx) == 0`

### 3. GQA Compatibility
**Challenge:** Qwen2 uses GQA where Q and K/V have different dimensions:
- Q: `[bsz, num_heads, seq_len, head_dim]` (14 heads)
- K/V: `[bsz, num_kv_heads, seq_len, head_dim]` (2 heads)

**Solution:** Apply 3D_RoPE **per-head** (on `head_dim=64`) after head splitting. Both Q and K have same `head_dim`, so rotation works.

### 4. Save-Restore Mechanism
**Problem:** Standard 1D_RoPE overwrites 3D_RoPE for point tokens.

**Solution:**
```python
# 1. Apply 3D_RoPE to point tokens
pt_q_3d, pt_k_3d = apply_3d_rope(pt_q, pt_k, coords)

# 2. Save 3D results
saved_q = pt_q_3d.clone()
saved_k = pt_k_3d.clone()

# 3. Apply 1D_RoPE to ALL tokens (overwrites points)
all_q, all_k = apply_1d_rope(all_q, all_k, position_ids)

# 4. Restore 3D_RoPE for points
all_q[point_indices] = saved_q
all_k[point_indices] = saved_k
```

### 5. Custom Attention Classes
**Only when `VLM_PE="3D_RoPE"`:**
- Replace `Qwen2Attention` → `SpatialQwen2Attention`
- Replace `LlamaAttention` → `SpatialLlamaAttention`
- If `VLM_PE=None`, uses **100% original** attention (no modifications)

---

## 🐛 Critical Bugs Fixed

### Bug #1: Shared PE Across Heads
**Problem:** Initial implementation applied 3D_RoPE to full `hidden_size` before head splitting, causing all heads to share the same PE pattern.

**Fix:** Apply per-head after splitting, so each head can learn different spatial features.

### Bug #2: GQA Dimension Mismatch
**Problem:** Applying to `hidden_size` caused Q/K dimension mismatch in GQA.

**Fix:** Apply to `head_dim` (64) where Q and K have matching dimensions.

### Bug #3: 1D_RoPE Overwriting 3D
**Problem:** Standard 1D_RoPE applied to all tokens, erasing 3D_RoPE.

**Fix:** Save-restore mechanism to preserve 3D_RoPE for point tokens.

### Bug #4: Prefill Detection Failure
**Problem:** `past_key_values is None` always False (it's a `DynamicCache` object).

**Fix:** Check `past_key_value.get_seq_length(layer_idx) == 0` instead.

### Bug #5: VLM_PE="None" Using Custom Attention
**Problem:** String `"None"` from CLI not converted to Python `None`, causing custom attention to activate.

**Fix:** Explicit conversion + only replace attention when `VLM_PE == "3D_RoPE"`.

### Bug #6: Aspect Ratio Distortion
**Problem:** Per-dimension normalization distorted scene geometry.

**Fix:** Global scale normalization preserving aspect ratio.

---

## 📊 Parameters

### `--VLM_PE "3D_RoPE"`
Enables 3D RoPE for point cloud tokens.

### `--PCD_PE_Merge_Rule {3D_only, 3D_with_1D}`
Controls dimension allocation:
- `3D_only`: Pure 3D spatial (21-21-22 split)
- `3D_with_1D`: Hybrid spatial+temporal (32-10-10-12 split)

### `SPATIALLM_VERBOSE=1` (Environment Variable)
Enable detailed checkpoint logging:
```bash
SPATIALLM_VERBOSE=1 python inference.py ...
```

Shows 9 checkpoints per attention layer during prefill:
1. After Q/K/V projection
2. 3D_RoPE data verification
3. Point token extraction
4. After 3D_RoPE application
5. After 3D_RoPE (full Q/K)
6. 1D_RoPE data
7. Saved point tokens
8. After 1D_RoPE to all
9. After restoring 3D_RoPE

**Note:** Verbose mode adds overhead, use only for debugging.

---

## 📁 Files Modified

### Core Implementation
- `spatiallm/model/spatiallm_qwen.py` - Main model (Qwen2)
- `spatiallm/model/spatiallm_llama.py` - Main model (Llama)
- `spatiallm/model/spatial_attention.py` - Custom attention classes
- `spatiallm/model/volumetric_pe.py` - `RotaryEmbedding3D` class

### Entry Points
- `inference.py` - Command-line interface
- `scripts/inference.sh` - Example script
- `scripts/test_notebook.ipynb` - Jupyter notebook

---

## ✅ Validation

### Test Cases
1. **VLM_PE=None**: No custom attention, 100% original behavior
2. **3D_RoPE + 3D_only**: Pure 3D spatial encoding
3. **3D_RoPE + 3D_with_1D**: Hybrid spatial + temporal encoding

### Expected Output (with SPATIALLM_VERBOSE=1)
```
[Config] VLM_PE = 3D_RoPE (from command line)
[Config] PCD_PE_Merge_Rule = 3D_only
[3D_RoPE] Initialized with PCD_PE_Merge_Rule=3D_only
[3D PE] Original patch_coords range: X=[0.069, 6.667], Y=[0.069, 5.304], Z=[0.154, 3.005]
[3D PE] Normalized with max_range=6.598, final range: X=[0.000, 1.000], Y=[0.000, 0.793], Z=[0.000, 0.432]

=== [Qwen2 3D_RoPE Sanity Check] ===
[CHECKPOINT 1] After Q/K/V projection and head reshape:
  query_states.shape: torch.Size([1, 14, 776, 64])
  key_states.shape: torch.Size([1, 2, 776, 64])
[CHECKPOINT 2] 3D RoPE data:
  cos_3d.shape: torch.Size([556, 64])
  sin_3d.shape: torch.Size([556, 64])
  point_token_positions: [(1, 557)]
...
[CHECKPOINT 9] Restored 3D RoPE for batch 0, positions [1:557]
✓ 3D RoPE preserved for point tokens, 1D RoPE applied to text tokens
```

---

## 🚨 Common Issues

### Issue: No checkpoint output
**Cause:** `SPATIALLM_VERBOSE` not set or set after imports  
**Fix:** `export SPATIALLM_VERBOSE=1` before running, or set `os.environ["SPATIALLM_VERBOSE"] = "1"` before importing spatiallm

### Issue: "Infinite walls" in predictions
**Cause:** `VLM_PE="None"` (string) not converted to `None` (Python)  
**Fix:** Now handled automatically in `inference.py`

### Issue: Aspect ratio looks wrong
**Cause:** Coordinates not normalized  
**Fix:** Now automatic - global scale normalization preserves aspect ratio

### Issue: AssertionError in shape checks
**Cause:** Batch size > 1 or unexpected tensor shapes  
**Fix:** Check `CHECKPOINT` outputs to identify shape mismatch

---

## 🔗 See Also

- [3D_SINUSOIDAL.md](./3D_SINUSOIDAL.md) - Alternative 3D PE method
- [README.md](./README.md) - General SpatialLM documentation

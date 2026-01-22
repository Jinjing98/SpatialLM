"""
CCA (Concentric Causal Attention) Utilities for SpatialLM

This module implements CCA for 3D point clouds by projecting them onto a 2D plane.
Core logic is extracted and adapted from modelling_fca_Feng.py (lines 1390-1497)
for image tokens, with modifications to support variable-length point cloud tokens.

Key Differences from modelling_fca_Feng.py:
1. Image tokens: Fixed 576 tokens (24x24 grid), directly from img_token_pos
   Point cloud: Variable N tokens, starting at point_token_pos+1 (after <point_start>)
   
2. 2D coordinates source:
   - Image: Pre-defined 24x24 grid positions
   - Point cloud: 3D→2D orthographic projection (top-down view)
   
3. Position offset:
   - Image: No offset, img_token_pos is first image token
   - Point cloud: +1 offset to skip <point_start> token

Author: Adapted from modelling_fca_Feng.py
"""

import torch

# ============================================================================
# Global CCA Configuration
# ============================================================================
CCA_GRID_SIZE = 24  # 2D grid size (24x24 = 576, same as image tokens in FCA)
CCA_PROJECTION = "top_down"  # Orthographic projection direction


# ============================================================================
# 3D to 2D Projection
# ============================================================================
def project_pointcloud_to_2d_grid(coords_3d, grid_size=CCA_GRID_SIZE, projection=CCA_PROJECTION):
    """
    Project 3D point cloud coordinates to 2D grid using orthographic projection.
    
    This is the key adaptation for extending CCA from 2D images to 3D point clouds.
    We use simple orthographic projection (dropping one dimension) rather than
    perspective projection, as point clouds don't have associated camera parameters.
    
    Args:
        coords_3d: (N, 3) torch.Tensor, normalized coordinates in [0, 1]
                   # TODO: Verify column order is [X, Y, Z]
        grid_size: int, size of 2D grid (default 24)
        projection: str, projection plane
                    - "top_down": XY plane (drop Z, bird's eye view) - DEFAULT
                    - "front": XZ plane (drop Y, front view)
                    - "side": YZ plane (drop X, side view)
    
    Returns:
        grid_row: (N,) torch.LongTensor, row indices in [0, grid_size-1]
        grid_col: (N,) torch.LongTensor, column indices in [0, grid_size-1]
    """
    device = coords_3d.device
    
    # Select 2D plane based on projection type
    if projection == "top_down":
        # Top-down view: use X (horizontal) and Y (depth)
        # TODO: Verify that coords_3d[:, 0] is X and coords_3d[:, 1] is Y
        x = coords_3d[:, 0]  # Assuming X is first column
        y = coords_3d[:, 1]  # Assuming Y is second column
        # Z (height) is projected out
    elif projection == "front":
        # Front view: use X (horizontal) and Z (height)
        x = coords_3d[:, 0]  # X
        y = coords_3d[:, 2]  # Z (mapped to Y axis in 2D)
    elif projection == "side":
        # Side view: use Y (depth) and Z (height)
        x = coords_3d[:, 1]  # Y (mapped to X axis in 2D)
        y = coords_3d[:, 2]  # Z (mapped to Y axis in 2D)
    else:
        raise ValueError(f"Unknown projection type: {projection}")
    
    # Map [0, 1] coordinates to grid indices [0, grid_size-1]
    grid_col = (x * (grid_size - 1)).long().clamp(0, grid_size - 1)
    grid_row = (y * (grid_size - 1)).long().clamp(0, grid_size - 1)
    
    return grid_row, grid_col


# ============================================================================
# Function 1: Build Concentric Position Matrix
# Directly copied from modelling_fca_Feng.py lines 1390-1405
# ============================================================================
def build_concentric_position_matrix(grid_size=CCA_GRID_SIZE, device='cpu'):
    """
    Build 2D concentric position matrix for CCA.
    
    This function is 100% copied from modelling_fca_Feng.py lines 1390-1405.
    It generates a 24x24 matrix where each cell contains its "shell ID" in a
    concentric pattern, with center points having ID 0 and outer points having ID 11.
    
    Example for 24x24 grid:
        Position matrix (center region):
        [[11, 11, 11, 10, 10, ..., 10, 11],
         [11, 10,  9,  9,  8, ...,  9, 10, 11],
         [11,  9,  8,  7,  6, ...,  7,  8,  9, 11],
         ...
         [11, 11, 11, 10, 10, ..., 10, 11]]
    
    Args:
        grid_size: int, size of grid (default 24)
        device: torch.device or str
    
    Returns:
        concentric_pos: (grid_size, grid_size) torch.LongTensor with position IDs [0-11]
    """
    H = W = grid_size
    concentric_pos = torch.zeros(H, W, dtype=torch.int64, device=device)
    
    # a pointer to assign concentric positions.
    pos_pt = [H // 2 - 1, W // 2 - 1, H // 2, W // 2]
    for pos in range(11, -1, -1):
        concentric_pos[pos_pt[0]: pos_pt[2] + 1, pos_pt[1]] = pos
        concentric_pos[pos_pt[0]: pos_pt[2] + 1, pos_pt[3]] = pos
        concentric_pos[pos_pt[0], pos_pt[1]: pos_pt[3] + 1] = pos
        concentric_pos[pos_pt[2], pos_pt[1]: pos_pt[3] + 1] = pos
        pos_pt = [pos_pt[0] - 1, pos_pt[1] - 1, pos_pt[2] + 1, pos_pt[3] + 1]
    
    return concentric_pos


# ============================================================================
# Function 2: Build CCA Position IDs
# Adapted from modelling_fca_Feng.py lines 1436-1466
# ============================================================================
def build_cca_position_ids(
    position_ids,
    point_token_pos,     # Adapted: img_token_pos → point_token_pos
    num_point_tokens,    # Adapted: New parameter, replaces IMG_TOKEN_LEN constant
    concentric_pos,      # Adapted: Passed as parameter for reusability
    device,
    seq_len,
    past_key_values=None
):
    """
    Build CCA position IDs for a single sample in the batch.
    
    Adapted from modelling_fca_Feng.py lines 1436-1466 with key modifications:
    1. Handles variable number of point tokens (vs fixed 576 image tokens)
    2. Accounts for <point_start> token offset (+1)
    3. Processes single sample instead of batched samples
    
    Sequence structure in SpatialLM:
        [text] <point_start> [P1, P2, ..., Pn] <point_end> [text]
        0...k     k+1         k+2 ... k+n+1      k+n+2    ...
        
        point_token_pos = k+1 (position of <point_start>)
        Actual point tokens start at k+2
    
    Args:
        position_ids: (seq_len,) original position IDs (can be None)
        point_token_pos: int, position of <point_start> token
                         If -1, this is text-only (no point cloud)
        num_point_tokens: int, number of point cloud tokens
        concentric_pos: (H, W) concentric position matrix
        device: torch.device
        seq_len: int, total sequence length after embedding fusion
        past_key_values: optional, for generation mode
    
    Returns:
        cca_position_ids: (seq_len,) torch.LongTensor with CCA-modified position IDs
    """
    H = concentric_pos.shape[0]
    
    # Text only (no point cloud)
    if point_token_pos == -1:
        if position_ids is not None:
            return position_ids.squeeze(0) if position_ids.dim() > 1 else position_ids
        else:
            return torch.arange(seq_len, dtype=torch.long, device=device)
    
    # Generation mode (decoding, seq_len == 1)
    if seq_len == 1 and past_key_values is not None:
        pos = past_key_values[0][0].shape[-2]
        cca_position_ids = torch.ones(1, dtype=torch.long, device=device) * (
            pos - num_point_tokens + H // 2  # Offset by max concentric position
        )
        return cca_position_ids
    
    # Prefill mode (encoding)
    if point_token_pos > seq_len - num_point_tokens:
        # Fallback: point cloud region is truncated, use standard position IDs
        cca_position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    else:
        # Construct CCA position IDs:
        # [text before] + [<point_start>] + [CCA positions for points] + [text after]
        cca_position_ids = torch.cat([
            # Text before point cloud + <point_start> token
            torch.arange(0, point_token_pos + 1, device=device),
            
            # Point cloud tokens: use concentric positions + offset
            # +1 offset because point tokens start AFTER <point_start>
            concentric_pos.flatten() + point_token_pos + 1,
            
            # Text after point cloud (offset by max concentric position)
            torch.arange(
                point_token_pos + 1 + H // 2,  # Start after concentric positions
                seq_len - num_point_tokens + H // 2,
                device=device
            )
        ]).to(torch.long)
    
    return cca_position_ids


# ============================================================================
# Function 3: Build CCA Attention Mask
# Adapted from modelling_fca_Feng.py lines 1468-1497
# ============================================================================
def build_cca_attention_mask(
    attention_mask,
    batch_point_token_pos,    # Adapted: img_token_pos → point_token_pos
    batch_cca_position_ids,
    num_point_tokens,         # Adapted: IMG_TOKEN_LEN → num_point_tokens
    batch_size,               # Adapted: Passed as parameter (no self)
    seq_len,                  # Adapted: Passed as parameter
    device,                   # Adapted: Passed as parameter
    dtype,                    # Adapted: Passed as parameter
    grid_size=CCA_GRID_SIZE
):
    """
    Build CCA attention mask for point cloud tokens.
    
    Adapted from modelling_fca_Feng.py lines 1468-1497 with modifications:
    1. Point cloud region starts at point_token_pos+1 (not point_token_pos)
    2. Variable number of point tokens (not fixed 576)
    3. All parameters passed explicitly (no self dependencies)
    
    CCA Mechanism:
    - Initially mask out all point-to-point attention
    - Gradually "open" attention in concentric shells:
      * Shell 0 (center): points can only see themselves
      * Shell 1: points can see shell 0 and shell 1
      * Shell 11 (outer): points can see all shells [0-11]
    - This creates a hierarchical spatial attention pattern
    
    Args:
        attention_mask: (B, 1, seq_len, seq_len) or None
        batch_point_token_pos: list of int, <point_start> positions for each sample
        batch_cca_position_ids: (B, seq_len) CCA position IDs
        num_point_tokens: int, number of point cloud tokens
        batch_size: int
        seq_len: int
        device: torch.device
        dtype: torch.dtype
        grid_size: int, CCA grid size (default 24)
    
    Returns:
        cca_attention_mask: (B, 1, seq_len, seq_len) attention mask with CCA applied
    """
    H = grid_size
    
    # Generation mode (decoding single token)
    if seq_len == 1:
        return attention_mask
    
    # Initialize with standard causal mask
    if attention_mask is not None:
        cca_attention_mask = attention_mask.clone()
    else:
        # Create upper triangular causal mask
        cca_attention_mask = torch.triu(
            float('-inf') * torch.ones(batch_size, 1, seq_len, seq_len, device=device),
            diagonal=1,
        ).to(dtype)
    
    # Apply CCA for each sample in batch
    for (b_idx, point_token_pos), cca_position_ids in zip(
        enumerate(batch_point_token_pos), batch_cca_position_ids
    ):
        # Skip text-only samples
        if point_token_pos == -1:
            continue
        
        # Skip if point cloud region is truncated
        if point_token_pos > seq_len - num_point_tokens:
            continue
        
        # === KEY ADAPTATION: Point cloud token indexing ===
        # In modelling_fca_Feng.py: image tokens are at [img_token_pos : img_token_pos + 576]
        # In SpatialLM: point tokens are at [point_token_pos+1 : point_token_pos+1 + N]
        #   because point_token_pos is <point_start>, actual points come after
        point_start = point_token_pos + 1
        point_end = point_start + num_point_tokens
        
        # Step 1: Mask out ALL point-to-point attention initially
        cca_attention_mask[
            b_idx, :,
            point_start:point_end,
            point_start:point_end
        ] = float('-inf') * torch.ones((num_point_tokens, num_point_tokens), device=device)
        
        # Step 2: Gradually open concentric shells
        # For each shell position [0, 1, 2, ..., 11]:
        for pos in torch.arange(point_token_pos + 1, point_token_pos + 1 + H // 2):
            # Find all keys with position <= current pos (can be attended to)
            k_pos = torch.nonzero(
                (cca_position_ids <= pos) & (cca_position_ids >= point_token_pos + 1)
            )[:, 0]
            
            # Find all queries with position == current pos (current shell)
            q_pos = torch.nonzero(
                (cca_position_ids == pos) & (cca_position_ids >= point_token_pos + 1)
            )[:, 0]
            
            # Allow attention between current shell (queries) and all inner shells (keys)
            if len(q_pos) > 0 and len(k_pos) > 0:
                m_pos = torch.cartesian_prod(q_pos, k_pos)
                cca_attention_mask[b_idx, 0, m_pos[:, 0], m_pos[:, 1]] = 0.0
    
    return cca_attention_mask

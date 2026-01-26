"""
Dimension Allocation Utility for 3D RoPE

This module provides utilities to allocate dimensions between 1D RoPE (temporal/sequence)
and 3D RoPE (spatial X, Y, Z) components with divisibility constraints.
"""

from typing import Tuple


def allocate_rope_dimensions(
    dim: int,
    ratio_1d: float,
    divide: int = 2
) -> Tuple[int, int, int, int]:
    """
    Allocate dimensions for 1D and 3D RoPE components.
    
    Args:
        dim: Total number of dimensions to allocate
        ratio_1d: Target ratio for 1D component (e.g., 0.5 for 50%)
        divide: Divisibility requirement for X, Y, Z (default: 2)
    
    Returns:
        Tuple of (d_1d, d_x, d_y, d_z) where:
        - d_1d: Dimensions for 1D RoPE (sequence position)
        - d_x, d_y, d_z: Dimensions for 3D spatial components
    
    Constraints:
        - All outputs are integers
        - d_1d + d_x + d_y + d_z == dim (exact)
        - d_1d is divisible by 2
        - d_x, d_y, d_z are each divisible by 'divide'
        - X, Y, Z are as equal as possible
        - 1D ratio is approximated as closely as possible
    
    Example:
        >>> allocate_rope_dimensions(64, 0.503, 2)
        (32, 10, 10, 12)
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if not 0 <= ratio_1d <= 1:
        raise ValueError(f"ratio_1d must be in [0, 1], got {ratio_1d}")
    if divide == 0:
        # allocate all for 1D
        return (dim, 0, 0, 0)
    
    # Step 1: Compute target d_1d and round to nearest even number
    target_d_1d = int(dim * ratio_1d)
    
    # Make d_1d divisible by 2 (round to nearest even)
    if target_d_1d % 2 != 0:
        # Try both rounding up and down, pick the one closer to target ratio
        d_1d_down = target_d_1d - 1 if target_d_1d > 0 else 0
        d_1d_up = target_d_1d + 1
        
        # Pick the one that gives a better ratio and leaves enough space for 3D
        # Need at least 3*divide dimensions for X, Y, Z
        min_3d_dims = 3 * divide
        
        if d_1d_up + min_3d_dims <= dim:
            # Both are valid, pick closer to target
            if abs(d_1d_up - target_d_1d) <= abs(d_1d_down - target_d_1d):
                d_1d = d_1d_up
            else:
                d_1d = d_1d_down
        else:
            # d_1d_up is too large
            d_1d = d_1d_down
    else:
        d_1d = target_d_1d
    
    # Ensure d_1d leaves enough space for 3D components
    max_d_1d = dim - 3 * divide
    if d_1d > max_d_1d:
        d_1d = max_d_1d
        # Round down to nearest even
        if d_1d % 2 != 0:
            d_1d -= 1
    
    # Ensure d_1d is non-negative and even
    d_1d = max(0, d_1d)
    if d_1d % 2 != 0:
        d_1d -= 1
    
    # Step 2: Allocate remaining dimensions to X, Y, Z
    remaining = dim - d_1d
    
    if remaining < 3 * divide:
        raise ValueError(
            f"Cannot allocate dimensions: dim={dim}, d_1d={d_1d}, "
            f"remaining={remaining}, need at least {3*divide} for X,Y,Z"
        )
    
    # Check if perfect division is possible
    if remaining % divide != 0:
        # Remaining is not divisible by 'divide'
        # We need to adjust d_1d to make remaining divisible by 'divide'
        # Try reducing d_1d by 2 until remaining is divisible
        attempts = 0
        max_attempts = (d_1d // 2) + 1
        
        while remaining % divide != 0 and attempts < max_attempts:
            d_1d -= 2  # Keep d_1d even
            remaining = dim - d_1d
            attempts += 1
            
            if d_1d < 0:
                # Can't make it work by reducing d_1d
                raise ValueError(
                    f"Cannot allocate dimensions with divisibility constraints: "
                    f"dim={dim}, ratio_1d={ratio_1d}, divide={divide}. "
                    f"The remaining dimensions after 1D allocation cannot be evenly divided."
                )
        
        # One more check
        if remaining % divide != 0:
            raise ValueError(
                f"Cannot satisfy divisibility constraint: "
                f"dim={dim}, remaining={remaining}, divide={divide}"
            )
    
    # Now remaining is divisible by 'divide'
    # Divide remaining as equally as possible among X, Y, Z
    # Each must be divisible by 'divide'
    
    # Strategy: Distribute in multiples of 'divide'
    # Start with base allocation, then distribute remainder chunks
    num_chunks = remaining // divide  # How many chunks of 'divide' we have
    base_chunks_per_dim = num_chunks // 3  # Each dimension gets at least this many
    extra_chunks = num_chunks % 3  # Leftover chunks to distribute
    
    # Allocate base
    d_x = base_chunks_per_dim * divide
    d_y = base_chunks_per_dim * divide
    d_z = base_chunks_per_dim * divide
    
    # Distribute extra chunks (Z first, then Y, then X)
    if extra_chunks >= 1:
        d_z += divide
    if extra_chunks >= 2:
        d_y += divide
    
    # Final sanity checks
    total = d_1d + d_x + d_y + d_z
    assert total == dim, f"Dimension mismatch: {d_1d}+{d_x}+{d_y}+{d_z}={total} != {dim}"
    assert d_1d % 2 == 0, f"d_1d={d_1d} is not divisible by 2"
    assert d_x % divide == 0, f"d_x={d_x} is not divisible by {divide}"
    assert d_y % divide == 0, f"d_y={d_y} is not divisible by {divide}"
    assert d_z % divide == 0, f"d_z={d_z} is not divisible by {divide}"
    
    return (d_1d, d_x, d_y, d_z)


def allocate_rope_dimensions_verbose(
    dim: int,
    ratio_1d: float,
    divide: int = 2,
    name: str = ""
) -> Tuple[int, int, int, int]:
    """
    Same as allocate_rope_dimensions but with verbose output for debugging.
    
    Args:
        dim: Total dimensions
        ratio_1d: Target ratio for 1D component
        divide: Divisibility requirement
        name: Optional name for this allocation (for logging)
    
    Returns:
        Tuple of (d_1d, d_x, d_y, d_z)
    """
    result = allocate_rope_dimensions(dim, ratio_1d, divide)
    d_1d, d_x, d_y, d_z = result
    
    actual_ratio = d_1d / dim if dim > 0 else 0
    ratio_error = abs(actual_ratio - ratio_1d)
    max_diff_3d = max(d_x, d_y, d_z) - min(d_x, d_y, d_z)
    
    prefix = f"[{name}] " if name else ""
    print(f"{prefix}dim={dim}, target_ratio={ratio_1d:.3f}, divide={divide}")
    print(f"{prefix}  Result: 1D={d_1d}, X={d_x}, Y={d_y}, Z={d_z}")
    print(f"{prefix}  Actual ratio: {actual_ratio:.3f} (error: {ratio_error:.3f})")
    print(f"{prefix}  3D equality: max_diff={max_diff_3d}")
    print(f"{prefix}  Checks: sum={d_1d+d_x+d_y+d_z}, 1D%2={d_1d%2}, X%{divide}={d_x%divide}, Y%{divide}={d_y%divide}, Z%{divide}={d_z%divide}")
    print()
    
    return result

def main():
    print("="*80)
    print("DIMENSION ALLOCATION UTILITY - USAGE EXAMPLES")
    print("="*80)
    print()
    
    # Example 1: Standard Qwen2-0.5B configuration
    print("Example 1: Qwen2-0.5B (head_dim=64, 50% for 1D RoPE)")
    print("-" * 80)
    d_1d, d_x, d_y, d_z = allocate_rope_dimensions_verbose(
        dim=64,
        ratio_1d=0.5,
        divide=2,
        name="Qwen2-0.5B"
    )
    print(f"Usage in code:")
    print(f"  inv_freq_1d = compute_inv_freq(d_1d={d_1d}, base=10000)")
    print(f"  inv_freq_x  = compute_inv_freq(d_x={d_x}, base=10000)")
    print(f"  inv_freq_y  = compute_inv_freq(d_y={d_y}, base=10000)")
    print(f"  inv_freq_z  = compute_inv_freq(d_z={d_z}, base=10000)")
    print()
    
    # Example 2: Pure 3D RoPE (no temporal component)
    print("Example 2: Pure 3D RoPE (no 1D component)")
    print("-" * 80)
    d_1d, d_x, d_y, d_z = allocate_rope_dimensions_verbose(
        dim=64,
        ratio_1d=0.0,  # No 1D component
        divide=2,
        name="Pure 3D"
    )
    print()
    
    # Example 3: Custom ratio
    print("Example 3: Custom Configuration (33% for 1D RoPE)")
    print("-" * 80)
    d_1d, d_x, d_y, d_z = allocate_rope_dimensions_verbose(
        dim=128,
        ratio_1d=0.33,
        divide=2,
        name="Custom 33%"
    )
    print()
    
    # Example 4: Integration with RotaryEmbedding3D
    print("Example 4: Integration with RotaryEmbedding3D class")
    print("-" * 80)
    print("```python")
    print("from spatiallm.model.dim_allocation_utils import allocate_rope_dimensions")
    print()
    print("class RotaryEmbedding3D(nn.Module):")
    print("    def __init__(self, head_dim, merge_rule='3D_with_1D', base=10000):")
    print("        super().__init__()")
    print("        ")
    print("        if merge_rule == '3D_only':")
    print("            # Pure 3D: no 1D component")
    print("            d_1d, d_x, d_y, d_z = allocate_rope_dimensions(")
    print("                dim=head_dim,")
    print("                ratio_1d=0.0,  # No 1D")
    print("                divide=2")
    print("            )")
    print("        elif merge_rule == '3D_with_1D':")
    print("            # Hybrid: 50% for 1D (temporal), 50% for 3D (spatial)")
    print("            d_1d, d_x, d_y, d_z = allocate_rope_dimensions(")
    print("                dim=head_dim,")
    print("                ratio_1d=0.5,  # 50% for 1D")
    print("                divide=2")
    print("            )")
    print("        ")
    print("        self.d_1d, self.d_x, self.d_y, self.d_z = d_1d, d_x, d_y, d_z")
    print("        ")
    print("        # Compute inverse frequencies for each component")
    print("        if d_1d > 0:")
    print("            self.inv_freq_1d = 1.0 / (base ** (torch.arange(0, d_1d, 2).float() / d_1d))")
    print("        self.inv_freq_x = 1.0 / (base ** (torch.arange(0, d_x, 2).float() / d_x))")
    print("        self.inv_freq_y = 1.0 / (base ** (torch.arange(0, d_y, 2).float() / d_y))")
    print("        self.inv_freq_z = 1.0 / (base ** (torch.arange(0, d_z, 2).float() / d_z))")
    print("```")
    print()
    
    # Example 5: Testing different configurations
    print("Example 5: Quick Comparison of Different Ratios")
    print("-" * 80)
    print(f"{'Ratio':<8} {'1D':<4} {'X':<4} {'Y':<4} {'Z':<4} {'Actual Ratio':<15} {'3D Balance'}")
    print("-" * 80)
    for ratio in [0.0, 0.25, 0.5, 0.75]:
        d_1d, d_x, d_y, d_z = allocate_rope_dimensions(64, ratio, 2)
        actual_ratio = d_1d / 64
        balance = max(d_x, d_y, d_z) - min(d_x, d_y, d_z)
        print(f"{ratio:<8.2f} {d_1d:<4} {d_x:<4} {d_y:<4} {d_z:<4} {actual_ratio:<15.3f} ±{balance}")
    print()
    
    print("="*80)
    print("For more examples, see test_dim_allocation.py")
    print("="*80)


if __name__ == "__main__":
    main()

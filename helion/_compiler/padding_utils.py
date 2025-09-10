"""Utilities for handling matrix multiplication padding for small dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ast
    from typing import Any
    
    import torch

from functools import reduce

from .._compat import min_dot_size
from .ast_extension import expr_from_string


def emit_tl_dot_with_small_dim_padding(
    lhs: ast.AST,
    rhs: ast.AST,
    acc: ast.AST | None,
    device: torch.device,
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
    emit_dot_func: Any,
    cast_func: Any,
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    acc_dtype: torch.dtype | None = None,
    input_precision: Any = None,
    out_dtype: torch.dtype | None = None,
) -> ast.AST:
    """Handle padding for small dimensions in matrix multiplication."""
    
    # Get hardware minimums
    min_m, min_n, min_k = min_dot_size(device, lhs_dtype, rhs_dtype)
    
    # Check if padding is needed
    pad_m = m is not None and m < min_m
    pad_n = n is not None and n < min_n
    pad_k = k is not None and k < min_k
    
    # Build kwargs for dot function
    kwargs = {}
    if input_precision:
        kwargs["input_precision"] = input_precision
    if out_dtype:
        kwargs["out_dtype"] = out_dtype
    
    # Fast path: no padding needed
    if not (pad_m or pad_n or pad_k):
        if acc:
            kwargs["acc"] = acc
        result = emit_dot_func(lhs, rhs, **kwargs)
        if acc_dtype:
            result = cast_func(result, acc_dtype)
        return result
    
    # Helper to create and combine masks
    def create_mask(dims_and_pads):
        """Create a combined mask from dimension specifications.
        
        Args:
            dims_and_pads: List of (dim_val, min_dim, pad_needed, slice_str) tuples
        """
        masks = []
        for dim_val, min_dim, pad_needed, slice_str in dims_and_pads:
            if pad_needed and dim_val:
                masks.append(expr_from_string(f"tl.arange(0, {min_dim}){slice_str} < {dim_val}"))
        
        if not masks:
            return None
        if len(masks) == 1:
            return masks[0]
        return reduce(lambda a, b: expr_from_string("{a} & {b}", a=a, b=b), masks)
    
    # General padding case for all dimensions
    # Pad LHS if needed (last 2 dims: [..., M, K])
    mask_lhs = create_mask([(m, min_m, pad_m, "[:, None]"), (k, min_k, pad_k, "[None, :]")])
    lhs_to_use = expr_from_string("tl.where({m}, {lhs}, 0)", m=mask_lhs, lhs=lhs) if mask_lhs else lhs
    
    # Pad RHS if needed (last 2 dims: [..., K, N])
    mask_rhs = create_mask([(k, min_k, pad_k, "[:, None]"), (n, min_n, pad_n, "[None, :]")])
    rhs_to_use = expr_from_string("tl.where({m}, {rhs}, 0)", m=mask_rhs, rhs=rhs) if mask_rhs else rhs
    
    # Pad accumulator if needed (last 2 dims: [..., M, N])
    acc_to_use = acc
    if acc and (pad_m or pad_n):
        mask_acc = create_mask([(m, min_m, pad_m, "[:, None]"), (n, min_n, pad_n, "[None, :]")])
        if mask_acc:
            acc_to_use = expr_from_string("tl.where({m}, {acc}, 0)", m=mask_acc, acc=acc)
    
    # Perform the dot operation
    if acc_to_use:
        kwargs["acc"] = acc_to_use
    result = emit_dot_func(lhs_to_use, rhs_to_use, **kwargs)
    
    if acc_dtype:
        result = cast_func(result, acc_dtype)
    
    # Extract valid region from result
    if pad_m or pad_n:
        # Handle dimension = 1 cases with sum reduction
        # When a dimension is 1 and needs padding (e.g., from 1 to 16), the output
        # will have the padded size. We MUST use sum reduction because:
        # 1. Masking preserves tensor shape ([16, N] → [16, N]), doesn't reduce to [1, N]
        # 2. Triton doesn't support dynamic slicing to extract just the first row
        # 3. Sum naturally reduces the padded dimension: [16, N] → [1, N]
        # Since padding uses zeros and only the first position has data, sum gives
        # the correct result while also handling the shape reduction
        if m == 1 and pad_m:
            # Sum over M dimension and restore shape
            result = expr_from_string("tl.sum({x}, -2)", x=result)
            result = expr_from_string("tl.expand_dims({x}, -2)", x=result)
        elif n == 1 and pad_n:
            # Sum over N dimension and restore shape
            result = expr_from_string("tl.sum({x}, -1)", x=result)
            result = expr_from_string("tl.expand_dims({x}, -1)", x=result)
        else:
            # Regular masking for non-dim=1 cases
            mask_result = create_mask([(m, min_m, pad_m, "[:, None]"), 
                                       (n, min_n, pad_n, "[None, :]")])
            if mask_result:
                result = expr_from_string("tl.where({m}, {r}, tl.zeros_like({r}))", m=mask_result, r=result)
    
    return result
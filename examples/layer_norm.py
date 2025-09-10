"""
Helion Layer Normalization Forward and Backward Example
========================================================
This example demonstrates a Helion kernel implementation of 1D layer normalization
with both forward and backward passes using FP16 inputs and compares it against
PyTorch's built-in layer_norm function.
"""

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel
def layer_norm_fwd_kernel(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs 1D layer normalization on the input tensor using Helion.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim], expected to be FP16.
        normalized_shape (list[int]): List containing the dimension to normalize over (should be length 1).
        weight (torch.Tensor): Learnable scale parameter of shape [dim].
        bias (torch.Tensor): Learnable bias parameter of shape [dim].
        eps (float, optional): Small value added to variance for numerical stability. Default is 1e-5.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - The layer-normalized output tensor of shape [batch_size, dim], in FP16.
            - Mean tensor of shape [batch_size], in FP32.
            - Reciprocal standard deviation tensor of shape [batch_size], in FP32.
    """
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {m}"
    assert len(normalized_shape) == 1, (
        "Helion layer norm only supports 1D layer norm currently"
    )
    assert normalized_shape[0] == n, (
        f"normalized shape mismatch {normalized_shape[0]} != {n}"
    )
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    mean = torch.empty([m], dtype=torch.float32, device=x.device)
    rstd = torch.empty([m], dtype=torch.float32, device=x.device)

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(torch.float32)
        # Compute mean
        mean_val = torch.sum(acc, dim=-1) / n
        # Compute variance
        centered = acc - mean_val[:, None]
        var_val = torch.sum(centered * centered, dim=-1) / n
        # Compute reciprocal standard deviation
        rstd_val = torch.rsqrt(var_val + eps)
        # Normalize
        normalized = centered * rstd_val[:, None]
        # Apply affine transformation
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))
        out[tile_m, :] = acc.to(x.dtype)
        mean[tile_m] = mean_val
        rstd[tile_m] = rstd_val
    return out, mean, rstd


# %%
@helion.kernel
def layer_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes gradients for layer normalization backward pass.
    Args:
        grad_out (torch.Tensor): Gradient w.r.t. output tensor of shape [batch_size, dim].
        x (torch.Tensor): Input tensor of shape [batch_size, dim].
        weight (torch.Tensor): Weight tensor of shape [dim].
        mean (torch.Tensor): Mean tensor of shape [batch_size].
        rstd (torch.Tensor): Reciprocal standard deviation tensor of shape [batch_size].
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Gradient w.r.t. input x of shape [batch_size, dim].
            - Gradient w.r.t. weight of shape [dim].
            - Gradient w.r.t. bias of shape [dim].
    """
    m, n = x.shape
    n = hl.specialize(n)

    # Allocate output tensors
    grad_x = torch.empty_like(x)
    grad_weight = torch.zeros([n], dtype=weight.dtype, device=weight.device)
    grad_bias = torch.zeros([n], dtype=weight.dtype, device=weight.device)

    for tile_m in hl.tile(m):
        # Load data for this tile
        x_tile = x[tile_m, :].to(torch.float32)
        grad_out_tile = grad_out[tile_m, :].to(torch.float32)
        weight_f32 = weight[:].to(torch.float32)
        mean_tile = mean[tile_m]
        rstd_tile = rstd[tile_m]

        # Compute normalized input
        x_hat = (x_tile - mean_tile[:, None]) * rstd_tile[:, None]

        # Compute intermediate values
        wdy = weight_f32 * grad_out_tile
        c1 = torch.sum(x_hat * wdy, dim=-1) / n
        c2 = torch.sum(wdy, dim=-1) / n

        # Compute gradient w.r.t. x
        dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd_tile[:, None]
        grad_x[tile_m, :] = dx.to(x.dtype)

        # Compute gradients w.r.t. weight and bias by reducing across batch/tile dim
        dw_vec = torch.sum(grad_out_tile * x_hat, dim=0).to(weight.dtype)
        db_vec = torch.sum(grad_out_tile, dim=0).to(weight.dtype)

        # Use atomic add for weight and bias gradients (1D accumulation over n)
        hl.atomic_add(grad_weight, [hl.arange(0, n)], dw_vec)
        hl.atomic_add(grad_bias, [hl.arange(0, n)], db_vec)

    return grad_x, grad_weight, grad_bias


# %%
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        """Forward pass for layer normalization."""
        y, mean, rstd = layer_norm_fwd_kernel(x, normalized_shape, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.normalized_shape = normalized_shape
        return y

    @staticmethod
    def backward(ctx, grad_out):
        """Backward pass for layer normalization."""
        x, weight, mean, rstd = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = layer_norm_bwd(grad_out, x, weight, mean, rstd)
        return grad_x, None, grad_weight, grad_bias, None


# %%
def layer_norm_autograd(x, normalized_shape, weight, bias, eps=1e-5):
    """Layer normalization with forward + backward support."""
    return LayerNormFunction.apply(x, normalized_shape, weight, bias, eps)


# %%
def main() -> None:
    """
    Main execution function for the layer normalization example.
    - Generates random input, weight, and bias tensors.
    - Runs the Helion layer normalization kernel and compares its output to PyTorch's
      built-in layer_norm function using the run_example utility.
    - Prints comparison results and checks for correctness within specified tolerances.
    """
    batch_size = 32
    dim = 64
    device = "cuda"

    # Test forward pass only
    print("\n=== Forward Pass Test ===")
    x = torch.randn([batch_size, dim], device=device, dtype=torch.float16)
    weight = torch.randn([dim], device=device, dtype=torch.float16)
    bias = torch.randn([dim], device=device, dtype=torch.float16)
    eps = 1e-4
    run_example(
        layer_norm_autograd,
        torch.nn.functional.layer_norm,
        (x, [dim], weight, bias, eps),
        kernel_name="helion_fwd_kernel",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )

    # Test forward + backward pass
    print("\n\n=== Forward + Backward Pass Test ===")
    x_grad = torch.randn(
        [batch_size, dim], device=device, dtype=torch.float16, requires_grad=True
    )
    weight_grad = torch.randn(
        [dim], device=device, dtype=torch.float16, requires_grad=True
    )
    bias_grad = torch.randn(
        [dim], device=device, dtype=torch.float16, requires_grad=True
    )
    run_example(
        layer_norm_autograd,
        torch.nn.functional.layer_norm,
        (x_grad, [dim], weight_grad, bias_grad, eps),
        kernel_name="helion_autograd",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
        bwd=True,
    )


# %%
if __name__ == "__main__":
    main()

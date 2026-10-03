"""Kernels copied verbatim from upstream python/test/unit/language/test_core.py::test_argmax_argmin_with_nan
at triton 93ba8872b6d783665af1cf662ebd48bda49e60b9 (they are nested inside the test there). Used only
to classify the reduction reach of the existing upstream NaN test; never launched."""
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(x_ptr, val_ptr, idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    val = tl.max(x, axis=0)
    idx = tl.argmax(x, axis=0)
    tl.store(val_ptr, val)
    tl.store(idx_ptr, idx)


@triton.jit
def argmin_kernel(x_ptr, val_ptr, idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
    val = tl.min(x, axis=0)
    idx = tl.argmin(x, axis=0)
    tl.store(val_ptr, val)
    tl.store(idx_ptr, idx)

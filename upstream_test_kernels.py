"""The four kernels of upstream test_argmax_argmin_with_nan and test_argmax_argmin_tie_break_fast_with_nan
(python/test/unit/language/test_core.py @ triton 93ba8872b6d783665af1cf662ebd48bda49e60b9), copied verbatim.
In the tests they are nested in the test functions, so they are repeated here to be launched directly."""
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


@triton.jit
def argmax_fast_kernel(x_ptr, val_ptr, idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    val = tl.max(x, axis=0)
    idx = tl.argmax(x, axis=0, tie_break_left=False)
    tl.store(val_ptr, val)
    tl.store(idx_ptr, idx)


@triton.jit
def argmin_fast_kernel(x_ptr, val_ptr, idx_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=float("inf"))
    val = tl.min(x, axis=0)
    idx = tl.argmin(x, axis=0, tie_break_left=False)
    tl.store(val_ptr, val)
    tl.store(idx_ptr, idx)


import triton, triton.language as tl
@triton.jit
def kern2d(x_ptr, val_ptr, idx_ptr, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, OP: tl.constexpr, TB: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    x = tl.load(x_ptr + om[:, None] * N + on[None, :])
    if AXIS == 0:
        oo = on
    else:
        oo = om
    tl.store(val_ptr + oo, tl.sum(x, AXIS))

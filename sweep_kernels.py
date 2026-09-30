import triton
import triton.language as tl

# OP: 0=max 1=min 2=sum 3=argmax 4=argmin (constexpr, static branch)


@triton.jit
def kern1d(x_ptr, val_ptr, idx_ptr, N: tl.constexpr, OP: tl.constexpr, TB: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, N))
    if OP == 0:
        tl.store(val_ptr + 0, tl.max(x, 0))
    elif OP == 1:
        tl.store(val_ptr + 0, tl.min(x, 0))
    elif OP == 2:
        tl.store(val_ptr + 0, tl.sum(x, 0))
    elif OP == 3:
        tl.store(idx_ptr + 0, tl.argmax(x, 0, tie_break_left=TB))
    else:
        tl.store(idx_ptr + 0, tl.argmin(x, 0, tie_break_left=TB))


@triton.jit
def kern2d(x_ptr, val_ptr, idx_ptr, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr,
           OP: tl.constexpr, TB: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    x = tl.load(x_ptr + om[:, None] * N + on[None, :])
    if AXIS == 0:
        oo = on
    else:
        oo = om
    if OP == 0:
        tl.store(val_ptr + oo, tl.max(x, AXIS))
    elif OP == 1:
        tl.store(val_ptr + oo, tl.min(x, AXIS))
    elif OP == 2:
        tl.store(val_ptr + oo, tl.sum(x, AXIS))
    elif OP == 3:
        tl.store(idx_ptr + oo, tl.argmax(x, AXIS, tie_break_left=TB))
    else:
        tl.store(idx_ptr + oo, tl.argmin(x, AXIS, tie_break_left=TB))

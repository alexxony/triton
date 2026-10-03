"""Source-based simulator of Triton's compiled argmax/argmin reduction (1D, no cross-warp round).

Spec (frozen before any evaluation): vault triton_00/notes/28-argmax-NaN-소스기반-시뮬레이터-명세.md.
Pure Python / numpy only; the self-test runs before anything else.

Model M1 (triton @ 93ba8872, see spec section 2 for the sources):
  1. in-thread: registers i = 0..R-1 reduced by an adjacent-pair binary tree, combine(earlier, later)
  2. in-warp butterfly: for lane bit b from high to low, acc[l] = combine(acc[l], acc[l ^ (1 << b)])
  3. result read from lane 0
  combine = standard.py _argmax_combine / _argmin_combine (NaN compares are False, so NaN as the first
  operand loses and takes the partner, NaN as the second operand wins).
Coordinate map for a blocked layout: bits low to high = register (sizePerThread), lane, warp, then
extra register (repeat) bits. Only warp_bits == 0 is covered.
"""
import numpy as np


def combine(a, b, op, tb):
    (v1, i1), (v2, i2) = a, b
    if op == "argmax":
        first = (v1 > v2) or (tb and v1 == v2 and i1 < i2)
    else:
        first = (v1 < v2) or (tb and v1 == v2 and i1 < i2)
    return a if first else b


def simulate(col, op, tb, spt, reg_bits, lane_bits, warp_bits=0):
    """Return (index, value) that lane 0 stores. col: 1-D float64 array of length N = 2**nbits."""
    if warp_bits != 0:
        raise ValueError("model does not cover warp_bits > 0")
    N = len(col)
    nbits = N.bit_length() - 1
    assert 1 << nbits == N, N
    r_eff = min(spt.bit_length() - 1, nbits)
    assert reg_bits >= r_eff and reg_bits + lane_bits == nbits, (reg_bits, lane_bits, r_eff, nbits)
    L = 1 << lane_bits
    R = 1 << reg_bits

    def coord(i, lane):
        return (i & ((1 << r_eff) - 1)) | (lane << r_eff) | ((i >> r_eff) << (r_eff + lane_bits))

    acc = []
    for lane in range(L):
        vals = [(float(col[coord(i, lane)]), coord(i, lane)) for i in range(R)]
        while len(vals) > 1:
            vals = [combine(vals[k], vals[k + 1], op, tb) for k in range(0, len(vals), 2)]
        acc.append(vals[0])
    for b in range(lane_bits - 1, -1, -1):
        acc = [combine(acc[l], acc[l ^ (1 << b)], op, tb) for l in range(L)]
    v, i = acc[0]
    return i, v


NAN = float("nan")
NINF = float("-inf")
PINF = float("inf")


def selftest():
    # --- L4 held-out predictions of spec 28 section 5 (BLOCK=4, lanes 0..3, padding -inf for argmax) ---
    blk = dict(op="argmax", tb=True, spt=1, reg_bits=0, lane_bits=2)
    table = {
        "A": ([NAN, 6.0, 8.0, NINF], 2),
        "B": ([3.0, 5.0, NAN, NINF], 1),
        "C": ([9.0, 1.0, NAN, NINF], 1),
        "D": ([1.0, 9.0, 2.0, NAN], 3),
        "E": ([NAN, 9.0, 1.0, 2.0], 1),
        "F": ([1.0, NAN, 9.0, 2.0], 2),
        "G": ([9.0, NAN, 1.0, 2.0], 0),
        "H": ([1.0, 2.0, 9.0, NAN], 3),
    }
    for name, (col, want) in table.items():
        got, _ = simulate(np.array(col), **blk)
        assert got == want, (name, got, want)
    # --- hand-computed N=8 (3 lane bits, no registers), the cases that motivated the model ---
    h = dict(op="argmax", tb=True, spt=1, reg_bits=0, lane_bits=3)
    assert simulate(np.array([9, 1, 2, 3, NAN, 4, 5, 6.0]), **h)[0] == 7      # NaN at 4 swallows the max 9 at 0
    assert simulate(np.array([1, 2, 3, 4, 5, 6, 7, NAN]), **h)[0] == 7        # NaN at 7 survives
    assert simulate(np.array([NAN, 1, 2, 3, 4, 5, 6, 7.0]), **h)[0] == 7      # NaN at 0 harmless
    # --- NaN-free data must equal the true argmax/argmin for every layout (value-distinct and tied) ---
    rng = np.random.default_rng(1)
    layouts = [(8, 1, 0, 3), (32, 1, 0, 5), (128, 1, 2, 5), (1024, 1, 5, 5), (128, 2, 2, 5), (1024, 4, 5, 5), (8, 1, 3, 0)]
    n = 0
    for N, spt, rb, lb in layouts:
        for op in ("argmax", "argmin"):
            for tb in (True, False):
                for _ in range(40):
                    col = rng.choice(np.arange(-5000, 5000), size=N, replace=False).astype(float)
                    want = int(np.argmax(col) if op == "argmax" else np.argmin(col))
                    assert simulate(col, op, tb, spt, rb, lb)[0] == want, (N, spt, rb, lb, op, tb)
                    n += 1
                # ties: tb=True must return the left-most extremum, tb=False must return some extremum
                col = rng.choice(np.arange(-5, 5), size=N).astype(float)
                ext = col.max() if op == "argmax" else col.min()
                got = simulate(col, op, tb, spt, rb, lb)[0]
                assert col[got] == ext
                if tb:
                    assert got == int(np.flatnonzero(col == ext)[0]), (N, spt, rb, lb, op)
    # --- covered-scope guard ---
    try:
        simulate(np.zeros(8), "argmax", True, 1, 0, 1, warp_bits=2)
        raise AssertionError("must refuse warp_bits > 0")
    except ValueError:
        pass
    print("SELFTEST OK (L4 table 8, N=8 hand cases 3, NaN-free property checks %d)" % n)


if __name__ == "__main__":
    selftest()

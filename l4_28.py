"""Out-of-sample L4 test of the spec-28 simulator (run on a Colab L4 session with the main@93ba8872 wheel).

Spec (frozen first): vault triton_00/notes/28-argmax-NaN-소스기반-시뮬레이터-명세.md section 5, test 3.
Part 1: the 8 inputs of the spec table, upstream argmax_kernel (probe_kernels.py), BLOCK=4, nw in {1,4}.
Part 2 (extra, declared here before the run, same frozen model): random inputs, BLOCK=N in {4,8,16,32},
nw in {1,4}, 1-3 NaNs at random positions, distinct values; model prediction vs the GPU result.
Layout for the prediction is read from the TTGIR of the very kernel that is launched.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/w")
os.chdir("/root/w")
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

import probe_kernels as P  # noqa: E402
import reach  # noqa: E402
import sim28  # noqa: E402

sim28.selftest()
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "| triton", triton.__version__)
OUT = "/root/out28"
os.makedirs(OUT, exist_ok=True)
NAN, NINF = float("nan"), float("-inf")
DIV16 = {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]], (2,): [["tt.divisibility", 16]]}

_layouts = {}


def layout_for(block, nw):
    """Compile the kernel for sm_89 and read the reduction layout from its TTGIR."""
    key = (block, nw)
    if key not in _layouts:
        sig = {"x_ptr": "*fp32", "val_ptr": "*fp32", "idx_ptr": "*i32", "N": "constexpr", "BLOCK": "constexpr"}
        src = ASTSource(fn=P.argmax_kernel, signature=sig, constexprs={"N": block, "BLOCK": block}, attrs=DIV16)
        k = triton.compile(src, target=GPUTarget("cuda", 89, 32), options={"num_warps": nw})
        d = reach.analyze_ttgir(k.asm["ttgir"])
        assert d["status"] == "ok" and d["warp_bits"] == 0, ("layout not covered by the model", key, d.get("reason"), d.get("warp_bits"))
        _layouts[key] = d
    return _layouts[key]


def predict(col, block, nw):
    d = layout_for(block, nw)
    pad = list(col) + [NINF] * (block - len(col))     # upstream mask path: other = -inf for argmax
    idx, _ = sim28.simulate(np.array(pad, dtype=np.float64), "argmax", True, d["spt"][0], d["reg_bits"], d["lane_bits"], d["warp_bits"])
    return int(idx)


def run_gpu(col, n, block, nw):
    x = torch.tensor(col, dtype=torch.float32, device="cuda")
    val = torch.empty((), dtype=torch.float32, device="cuda")
    idx = torch.empty((), dtype=torch.int32, device="cuda")
    P.argmax_kernel[(1,)](x, val, idx, N=n, BLOCK=block, num_warps=nw)
    torch.cuda.synchronize()
    return int(idx.item())


def true_idx(col):
    a = np.array(col, dtype=np.float64)
    ok = ~np.isnan(a)
    return int(np.flatnonzero(ok & (a == a[ok].max()))[0])


# ---------------- part 1: the spec table ----------------
SPEC = {   # name: (valid values, N, expected correct index, spec-predicted returned index, spec-predicted outcome)
    "A": ([NAN, 6.0, 8.0], 3, 2, 2, "pass"),
    "B": ([3.0, 5.0, NAN], 3, 1, 1, "pass"),
    "C": ([9.0, 1.0, NAN], 3, 0, 1, "FAIL"),
    "D": ([1.0, 9.0, 2.0, NAN], 4, 1, 3, "FAIL"),
    "E": ([NAN, 9.0, 1.0, 2.0], 4, 1, 1, "pass"),
    "F": ([1.0, NAN, 9.0, 2.0], 4, 2, 2, "pass"),
    "G": ([9.0, NAN, 1.0, 2.0], 4, 0, 0, "pass"),
    "H": ([1.0, 2.0, 9.0, NAN], 4, 2, 3, "FAIL"),
}
rows = []
print("\n== part 1: spec section 5 table (BLOCK=4) ==")
print("%-3s %-3s %-22s %8s %8s %8s %6s  %s" % ("in", "nw", "values", "correct", "spec", "model", "GPU", "match"))
n_ok = n_all = 0
for name, (col, n, correct, spec_pred, outcome) in SPEC.items():
    assert true_idx(col) == correct, name
    for nw in (1, 4):
        model = predict(col, 4, nw)
        assert model == spec_pred, ("model drifted from the frozen spec table", name, nw, model, spec_pred)
        got = run_gpu(col, n, 4, nw)
        good = got == model
        n_all += 1
        n_ok += int(good)
        print("%-3s %-3d %-22s %8d %8d %8d %6d  %s%s" % (name, nw, str(col).replace("nan", "N"), correct, spec_pred, model, got,
                                                       "OK" if good else "MISMATCH", "" if (got == correct) == (outcome == "pass") else "  (outcome differs from spec)"))
        rows.append(dict(part=1, name=name, nw=nw, correct=correct, model=model, gpu=got, match=good))
print("part 1: GPU index == model index in %d of %d" % (n_ok, n_all))

# ---------------- part 2: random inputs ----------------
rng = np.random.default_rng(2828)
rows2 = []
mism = []
for block in (4, 8, 16, 32):
    for nw in (1, 4):
        ok = tot = fails = 0
        for _ in range(100):
            vals = rng.choice(np.arange(-300, 300), size=block, replace=False).astype(float)
            for p in rng.choice(block, size=int(rng.integers(1, 4)) if block > 4 else int(rng.integers(1, 3)), replace=False):
                vals[p] = NAN
            col = vals.tolist()
            model = predict(col, block, nw)
            got = run_gpu(col, block, block, nw)
            tot += 1
            ok += int(got == model)
            fails += int(got != true_idx(col))
            rows2.append(dict(part=2, block=block, nw=nw, col=col, model=model, gpu=got, correct=true_idx(col)))
            if got != model:
                mism.append((block, nw, col, model, got))
        print("part 2: BLOCK=%2d nw=%d  GPU==model %3d/%3d   (compiled result wrong vs true argmax in %3d of them)" % (block, nw, ok, tot, fails))
print("part 2 mismatches: %d" % len(mism))
for m in mism[:20]:
    print("   ", m)
with open(os.path.join(OUT, "l4.jsonl"), "w") as f:
    for r in rows + rows2:
        f.write(json.dumps(r) + "\n")
tot2 = len(rows2)
ok2 = sum(1 for r in rows2 if r["gpu"] == r["model"])
json.dump(dict(part1_ok=n_ok, part1_all=n_all, part2_ok=ok2, part2_all=tot2), open(os.path.join(OUT, "l4_summary.json"), "w"))
print("\nSUMMARY part1 %d/%d, part2 %d/%d" % (n_ok, n_all, ok2, tot2))

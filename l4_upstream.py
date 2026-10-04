"""Does the unmodified upstream NaN test pass on main (no patch)?  Run on a Colab L4 with the main@93ba8872 wheel.

PREDICTION (written before the run; vault note 28 section 11): the frozen model M1 predicts that all 7 asserted
cases x {fp32, bf16} pass, i.e. the upstream tests pass on unpatched main. Asserted cases are copied from the two
tests; kernels in upstream_test_kernels.py; launched exactly as in the tests (BLOCK=4, N=3, default num_warps).
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

import reach  # noqa: E402
import sim28  # noqa: E402
import upstream_test_kernels as U  # noqa: E402

sim28.selftest()
NAN, INF = float("nan"), float("inf")
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "| triton", triton.__version__)
DIV16 = {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]], (2,): [["tt.divisibility", 16]]}
# (kernel, values, expected idx, expected val)   -- copied from the two tests
CASES = [
    ("argmax_kernel", [NAN, 6.0, 8.0], 2, 8.0),
    ("argmin_kernel", [NAN, 6.0, 8.0], 1, 6.0),
    ("argmax_kernel", [3.0, 5.0, NAN], 1, 5.0),
    ("argmax_fast_kernel", [NAN, 6.0, 8.0], 2, 8.0),
    ("argmin_fast_kernel", [NAN, 6.0, 8.0], 1, 6.0),
    ("argmax_fast_kernel", [3.0, 5.0, NAN], 1, 5.0),
    ("argmin_fast_kernel", [3.0, NAN, 1.0], 2, 1.0),
]
PTRS = {"fp32": "*fp32", "bf16": "*bf16"}
TDT = {"fp32": torch.float32, "bf16": torch.bfloat16}


def layout(kname, dt, nw=4):
    sig = {"x_ptr": PTRS[dt], "val_ptr": PTRS[dt], "idx_ptr": "*i32", "N": "constexpr", "BLOCK": "constexpr"}
    src = ASTSource(fn=getattr(U, kname), signature=sig, constexprs={"N": 3, "BLOCK": 4}, attrs=DIV16)
    k = triton.compile(src, target=GPUTarget("cuda", 89, 32), options={"num_warps": nw})
    d = reach.analyze_ttgir(k.asm["ttgir"])
    assert d["status"] == "ok" and d["warp_bits"] == 0, (kname, dt, d.get("reason"), d.get("warp_bits"))
    return d


rows = []
allok = True
print("\n%-20s %-5s %-22s %8s %8s %8s %8s  %s" % ("kernel", "dtype", "values", "expect", "model", "GPU idx", "GPU val", "result"))
for dt in ("fp32", "bf16"):
    for kname, vals, eidx, evalue in CASES:
        argmax = "argmax" in kname
        tb = "fast" not in kname
        d = layout(kname, dt)
        pad = vals + [-INF if argmax else INF]
        model, _ = sim28.simulate(np.array(pad, dtype=np.float64), "argmax" if argmax else "argmin", tb, d["spt"][0], d["reg_bits"], d["lane_bits"], d["warp_bits"])
        x = torch.tensor(vals, dtype=TDT[dt], device="cuda")
        val = torch.empty((), dtype=TDT[dt], device="cuda")
        idx = torch.empty((), dtype=torch.int32, device="cuda")
        getattr(U, kname)[(1,)](x, val, idx, N=3, BLOCK=4)       # default num_warps, as in the tests
        torch.cuda.synchronize()
        gi, gv = int(idx.item()), float(val.item())
        passed = (gi == eidx) and (gv == evalue)
        allok &= passed
        rows.append(dict(kernel=kname, dtype=dt, values=vals, expect_idx=eidx, model_idx=int(model), gpu_idx=gi, gpu_val=gv, test_assert_pass=passed, model_matches_gpu=(int(model) == gi)))
        print("%-20s %-5s %-22s %8d %8d %8d %8s  %s%s" % (kname, dt, str(vals).replace("nan", "N"), eidx, model, gi, gv, "PASS" if passed else "FAIL",
                                                      "" if int(model) == gi else "   (model != GPU)"))
print("\nupstream test assertions on unpatched main: %s" % ("ALL PASS" if allok else "SOME FAIL"))
print("model idx == GPU idx in %d of %d" % (sum(r["model_matches_gpu"] for r in rows), len(rows)))
os.makedirs("/root/out28", exist_ok=True)
json.dump(rows, open("/root/out28/upstream_test.json", "w"))

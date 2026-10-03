"""Negative controls for spec 28 (appendix 9): alternative lowering orders against the same data.

Registered before the run in vault note 28 section 9. Colab CPU. Data: the two Drive tgz files extracted to
/root/data/l4 and /root/data/off. M1 = frozen model (must reproduce the recorded predictions exactly).
"""
import json
import os
import sys
import zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/root/w")
os.chdir("/root/w")
import sim28  # noqa: E402
import sweep  # noqa: E402

NAN, NINF = float("nan"), float("-inf")
combine = sim28.combine


def sim_variant(col, op, tb, spt, reg_bits, lane_bits, v):
    N = len(col)
    nbits = N.bit_length() - 1
    if v == "A6":      # NaN-ignoring symmetric combine == correct argmax/argmin (left-most tie)
        a = np.array(col, dtype=np.float64)
        ok = ~np.isnan(a)
        ext = a[ok].max() if op == "argmax" else a[ok].min()
        return int(np.flatnonzero(ok & (a == ext))[0])
    if v == "A5":      # layout-blind sequential left-to-right fold
        acc = (float(col[0]), 0)
        for c in range(1, N):
            acc = combine(acc, (float(col[c]), c), op, tb)
        return acc[1]
    r_eff = min(spt.bit_length() - 1, nbits)
    L, R = 1 << lane_bits, 1 << reg_bits

    def coord(i, lane):
        return (i & ((1 << r_eff) - 1)) | (lane << r_eff) | ((i >> r_eff) << (r_eff + lane_bits))

    inth = (lambda a, b: combine(b, a, op, tb)) if v == "A3" else (lambda a, b: combine(a, b, op, tb))
    acc = []
    for lane in range(L):
        vals = [(float(col[coord(i, lane)]), coord(i, lane)) for i in range(R)]
        while len(vals) > 1:
            vals = [inth(vals[k], vals[k + 1]) for k in range(0, len(vals), 2)]
        acc.append(vals[0])
    bits = range(lane_bits) if v == "A1" else range(lane_bits - 1, -1, -1)
    for b in bits:
        if v == "A2":
            acc = [combine(acc[l ^ (1 << b)], acc[l], op, tb) for l in range(L)]
        else:
            acc = [combine(acc[l], acc[l ^ (1 << b)], op, tb) for l in range(L)]
    return acc[L - 1 if v == "A4" else 0][1]


VARIANTS = ["M1", "A1", "A2", "A3", "A4", "A5", "A6"]


def run(col, op, tb, lay, v):
    if v == "M1":
        return int(sim28.simulate(np.array(col, dtype=np.float64), op, tb, *lay)[0])
    return int(sim_variant(np.array(col, dtype=np.float64), op, tb, *lay, v))


# ---------------- D1/D2: L4 data (argmax kernel, tb=True, lanes only, spt=1, no registers) ----------------
SPEC = {
    "A": [NAN, 6.0, 8.0, NINF], "B": [3.0, 5.0, NAN, NINF], "C": [9.0, 1.0, NAN, NINF], "D": [1.0, 9.0, 2.0, NAN],
    "E": [NAN, 9.0, 1.0, 2.0], "F": [1.0, NAN, 9.0, 2.0], "G": [9.0, NAN, 1.0, 2.0], "H": [1.0, 2.0, 9.0, NAN],
}
rows = [json.loads(l) for l in open("/root/data/l4/out28/l4.jsonl")]
p1 = [r for r in rows if r["part"] == 1]
p2 = [r for r in rows if r["part"] == 2]
wrong = [r for r in p2 if r["gpu"] != r["correct"]]
print("L4 rows: part1 %d, part2 %d, compiled-wrong part2 %d (expect 263)" % (len(p1), len(p2), len(wrong)))


def lay_of(n):
    return (1, 0, n.bit_length() - 1)     # spt=1, reg_bits=0, lane_bits=log2(block) as read from TTGIR in l4_28.py


# consistency: the frozen model must reproduce every recorded prediction
bad = 0
for r in p2:
    if run(r["col"], "argmax", True, lay_of(r["block"]), "M1") != r["model"]:
        bad += 1
for r in p1:
    if run(SPEC[r["name"]], "argmax", True, lay_of(4), "M1") != r["model"]:
        bad += 1
print("M1 reproduces recorded predictions: %s (%d differences)" % ("YES" if bad == 0 else "NO", bad))
assert bad == 0

res = defaultdict(dict)
for v in VARIANTS:
    res[v]["D1 compiled-wrong random (%d)" % len(wrong)] = sum(run(r["col"], "argmax", True, lay_of(r["block"]), v) != r["gpu"] for r in wrong)
    res[v]["D2 all random (%d)" % len(p2)] = sum(run(r["col"], "argmax", True, lay_of(r["block"]), v) != r["gpu"] for r in p2)
    res[v]["spec table (%d)" % len(p1)] = sum(run(SPEC[r["name"]], "argmax", True, lay_of(4), v) != r["gpu"] for r in p1)

# ---------------- D3: fresh NaN offline cells ----------------
cells = [json.loads(l) for l in open("/root/data/off/out28/cells.jsonl")]
s0 = {}
for l in open("/root/data/off/out/stage0.jsonl"):
    r = json.loads(l)
    s0[r["key"]] = r
fresh = [c for c in cells if c["split"] == "heldout" and c["fam"] in ("nank", "ties_max_nan", "ties_min_nan")]
print("fresh NaN cells:", len(fresh), "| measured violating:", sum(c["meas"] for c in fresh))
d3n = "D3 fresh NaN offline (%d)" % len(fresh)
for v in VARIANTS:
    mm = 0
    for c in fresh:
        cfg = s0[c["key"]]["cfg"]
        rc = s0[c["key"]]["reach"]
        R, L, shape, ax = sweep.geometry(cfg)
        rng = np.random.default_rng(zlib.crc32(("%s|%s|%s|%s" % (c["key"], c["name"], c["pos"], 0)).encode()))
        xs = sweep.gen_pattern(cfg["dtype"], c["name"], R, L, rng, c["pos"])
        tb = True if cfg["tb"] is None else bool(cfg["tb"])
        idx = run(xs[:, 0], cfg["op"], tb, (rc["spt"][0], rc["reg_bits"], rc["lane_bits"]), v)
        pred = any(s == "violation" for _, s, _ in sweep.judge(cfg["op"], tb, xs, np.array([0.0]), np.array([idx])))
        mm += int(pred != c["meas"])
    res[v][d3n] = mm

cols = list(res["M1"].keys())
print("\n== mismatches (lower is better; M1 is the frozen model) ==")
print("%-4s " % "var" + " ".join("%-30s" % c for c in cols))
for v in VARIANTS:
    print("%-4s " % v + " ".join("%-30d" % res[v][c] for c in cols))
print("\nA1 masks low->high | A2 combine(partner, own) | A3 in-thread combine(later, earlier) | A4 read highest lane | A5 sequential fold | A6 correct argmax")
json.dump(res, open("/root/out_ablate28.json", "w"))

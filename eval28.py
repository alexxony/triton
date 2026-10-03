"""Offline evaluation of the spec-28 simulator against v1 measurements (Colab CPU, wheel not needed).

Spec (frozen first): vault triton_00/notes/28-argmax-NaN-소스기반-시뮬레이터-명세.md section 5, tests 1 and 2.
Inputs on the session: /root/out/stage0.jsonl (layouts from TTGIR, made by stage0.py run),
/root/w/dumpA_full.json (v1 DUMP_A with viol_names and viols). Cells whose model does not cover
(warp_bits > 0) are excluded, never guessed.
"""
import json
import os
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "/root/w")
os.chdir("/root/w")
import sim28  # noqa: E402
import sweep  # noqa: E402

sim28.selftest()
OUT = "/root/out28"
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open("/root/out/stage0.jsonl")]
dump = json.load(open("/root/w/dumpA_full.json"))
dcfg = {c["key"]: c for c in dump["cfgs"]}
dviols = defaultdict(list)
for v in dump["viols"]:
    dviols[v["key"]].append(v)


def inputs(c, name, pos):
    R, L, shape, ax = sweep.geometry(c)
    key = sweep.cfg_key(c)
    rng = np.random.default_rng(zlib.crc32(("%s|%s|%s|%s" % (key, name, pos, 0)).encode()))
    return sweep.gen_pattern(c["dtype"], name, R, L, rng, pos)


def tbv(c):
    return True if c["tb"] is None else bool(c["tb"])


cov = {}
excluded = Counter()
for r in rows:
    c, rc = r["cfg"], r["reach"]
    if c["dim"] != 1 or c["dtype"] == "int32" or c["op"] not in sweep.ARG_OPS:
        continue
    if rc.get("status") != "ok":
        excluded["unclassified"] += 1
        continue
    if rc["warp_bits"] != 0:
        excluded["warp_bits>0 (not covered)"] += 1
        continue
    cov[r["key"]] = (c, rc)
print("covered 1D float arg cfgs: %d   excluded: %s" % (len(cov), dict(excluded)))
print("covered by N,nw:", dict(sorted(Counter((c["shape"][0], c["nw"]) for c, _ in cov.values()).items())))
print("layouts seen (spt, reg, lane):", dict(Counter((tuple(rc["spt"]), rc["reg_bits"], rc["lane_bits"]) for _, rc in cov.values())))

# ---------------- test 1: boolean per cell ----------------
cells = []
for key, (c, rc) in cov.items():
    dc = dcfg.get(key)
    assert dc is not None, "cfg missing in DUMP_A: " + key
    measured = set((n, p) for n, p in dc["viol_names"])
    for name, pos in sweep.patterns_for(c):
        xs = inputs(c, name, pos)
        idx, val = sim28.simulate(xs[:, 0], c["op"], tbv(c), rc["spt"][0], rc["reg_bits"], rc["lane_bits"])
        res = sweep.judge(c["op"], tbv(c), xs, np.array([val]), np.array([idx]))
        pred = any(s == "violation" for _, s, _ in res)
        kinds = sorted(k for _, s, k in res if s == "violation")
        fam = sweep.base_name(name)
        N = c["shape"][0]
        split = "fit" if (N == 8 and fam == "nan1_pos") else "heldout"
        cells.append(dict(key=key, name=name, pos=pos, fam=fam, N=N, nw=c["nw"], dtype=c["dtype"], op=c["op"],
                          tb=tbv(c), split=split, pred=pred, meas=(name, pos) in measured, pred_idx=int(idx),
                          pred_kinds=kinds))
with open(os.path.join(OUT, "cells.jsonl"), "w") as f:
    for x in cells:
        f.write(json.dumps(x) + "\n")


def show(title, keyf, subset):
    g = defaultdict(lambda: [0, 0, 0, 0, 0])   # n, agree, TT, FF, mismatch
    for x in subset:
        k = keyf(x)
        g[k][0] += 1
        ok = x["pred"] == x["meas"]
        g[k][1] += int(ok)
        g[k][2] += int(x["pred"] and x["meas"])
        g[k][3] += int((not x["pred"]) and (not x["meas"]))
        g[k][4] += int(not ok)
    print("\n== %s ==" % title)
    print("  %-30s %6s %6s %6s %6s %9s" % ("group", "cells", "agree", "TT", "FF", "mismatch"))
    for k in sorted(g, key=str):
        n, a, tt, ff, mm = g[k]
        print("  %-30s %6d %6d %6d %6d %9d" % (k, n, a, tt, ff, mm))


held = [x for x in cells if x["split"] == "heldout"]
fit = [x for x in cells if x["split"] == "fit"]
print("\ncells total %d | held-out %d | fit (seen) %d" % (len(cells), len(held), len(fit)))
show("HELD-OUT by family", lambda x: x["fam"], held)
show("HELD-OUT by N", lambda x: x["N"], held)
show("HELD-OUT by dtype", lambda x: x["dtype"], held)
show("HELD-OUT by (op, tb)", lambda x: (x["op"], x["tb"]), held)
show("FIT (N=8 nan1_pos; not evidence)", lambda x: x["N"], fit)
mis = [x for x in held if x["pred"] != x["meas"]]
print("\nHELD-OUT MISMATCHES: %d of %d" % (len(mis), len(held)))
for x in mis[:60]:
    print("  ", x["key"], x["name"], x["pos"], "pred=%s meas=%s pred_idx=%d pred_kinds=%s" % (x["pred"], x["meas"], x["pred_idx"], x["pred_kinds"]))

# ---------------- test 2: exact returned index ----------------
tot = ok = 0
bad = []
bykind = defaultdict(lambda: [0, 0])
for key, (c, rc) in cov.items():
    for v in dviols.get(key, []):
        xs = inputs(c, v["pat"], v["pos"])
        if "col" in v:
            assert np.array_equal(np.array(v["col"], dtype=np.float64), xs[:, v["lane"]], equal_nan=True), "regeneration mismatch " + key
        idx, _ = sim28.simulate(xs[:, v["lane"]], c["op"], tbv(c), rc["spt"][0], rc["reg_bits"], rc["lane_bits"])
        tot += 1
        good = int(idx) == int(v["got"])
        ok += int(good)
        bykind[v["kind"]][0] += 1
        bykind[v["kind"]][1] += int(good)
        if not good:
            bad.append((key, v["pat"], v["pos"], v["kind"], int(idx), int(v["got"])))
print("\n== exact returned index on stored violation records (covered 1D) ==")
print("  records %d, predicted index == measured index: %d" % (tot, ok))
for k, (n, g) in sorted(bykind.items()):
    print("  kind %-14s %4d records, %4d exact" % (k, n, g))
for b in bad[:40]:
    print("   MISMATCH", b)
json.dump(dict(cells=len(cells), heldout=len(held), heldout_mismatch=len(mis), exact_records=tot, exact_ok=ok),
          open(os.path.join(OUT, "summary.json"), "w"))

import json, os, re, sys, tempfile
from collections import Counter, defaultdict
sys.path.insert(0, "/root/w"); os.chdir("/root/w")
import reach, stage0
rows = [json.loads(l) for l in open("/root/out/stage0.jsonl")]
# 1) which configs are ptx_mismatch / ok, by dim, axis, dtype
g = defaultdict(Counter)
for r in rows:
    c = r["cfg"]; st = r["reach"]["status"]
    reason = (r["reach"].get("reason") or "")[:34]
    g[(c["dim"], c["axis"])][("ok" if st == "ok" else reason)] += 1
print("== status by (dim, axis) ==")
for k in sorted(g): print(" ", k, dict(g[k]))
# 2) violation rate by dtype among ok configs
d = defaultdict(lambda: [0, 0])
for r in rows:
    if r["reach"]["status"] == "ok":
        d[r["cfg"]["dtype"]][0] += 1; d[r["cfg"]["dtype"]][1] += int(r["violating"])
print("== violating by dtype (ok configs) ==", {k: tuple(v) for k, v in d.items()})
dd = defaultdict(lambda: [0, 0])
for r in rows:
    dd[r["cfg"]["dtype"]][0] += 1; dd[r["cfg"]["dtype"]][1] += int(r["violating"])
print("== violating by dtype (ALL arg configs) ==", {k: tuple(v) for k, v in dd.items()})
# 3) float-only violation rate by stage signature (ok configs)
fs = defaultdict(lambda: [0, 0])
for r in rows:
    if r["reach"]["status"] == "ok" and r["cfg"]["dtype"] != "int32":
        fs[r["reach"]["sig"]][0] += 1; fs[r["reach"]["sig"]][1] += int(r["violating"])
print("== FLOAT-only violating by signature ==", {k: tuple(v) for k, v in sorted(fs.items())})
nb = defaultdict(lambda: [0, 0])
for r in rows:
    if r["reach"]["status"] == "ok" and r["cfg"]["dtype"] != "int32":
        nb[r["reach"]["nbits"]][0] += 1; nb[r["reach"]["nbits"]][1] += int(r["violating"])
print("== FLOAT-only violating by nbits ==", {k: tuple(v) for k, v in sorted(nb.items())})
# 4) look at a few mismatches in detail
def show(c):
    with tempfile.TemporaryDirectory() as td:
        r = stage0.compile_cfg(c, td)
    t, p = r["ttgir"], r["ptx"]
    a = reach.analyze_ttgir(t)
    print("CFG", stage0._layout_key(c), "model:", a.get("sig"), a.get("reg_bits"), a.get("lane_bits"), a.get("warp_bits"),
          "layout", a.get("spt"), a.get("tpw"), a.get("wpc"), "| ttgir convert_layout:", t.count("convert_layout"),
          "| ptx shfl:", p.count("shfl.sync"), "bar:", p.count("bar.sync"), "st.shared:", p.count("st.shared"), "ld.shared:", p.count("ld.shared"))
    return t
mm = [r for r in rows if r["reach"]["status"] != "ok"]
seen = set()
for r in mm:
    c = r["cfg"]; k = (c["dim"], c["axis"], r["reach"]["reason"][:34])
    if k in seen or len(seen) >= 4: continue
    seen.add(k); t = show(c)
    if len(seen) == 1:
        print("---- TTGIR (first mismatch, trimmed) ----"); print("\n".join(l for l in t.split("\n") if any(x in l for x in ("#blocked", "#linear", "convert_layout", "tt.reduce", "tt.store", "tt.load")))[:2500])
for c in [x["cfg"] for x in rows if x["reach"]["status"] == "ok" and x["cfg"]["dim"] == 2][:2] + [x["cfg"] for x in rows if x["reach"]["status"] == "ok" and x["cfg"]["dim"] == 1][:2]:
    show(c)

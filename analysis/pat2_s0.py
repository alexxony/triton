import gzip, json, os, sys
from collections import defaultdict
sys.path.insert(0, "/root/w"); os.chdir("/root/w")
import sweep
dump = json.load(gzip.open("dump_a_cfgs.json.gz", "rt"))
by = {c["key"]: c for c in dump["cfgs"]}
rows = [json.loads(l) for l in open("/root/out/stage0.jsonl")]
def tbl(pat, keyf, title, sel=lambda r: True):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        c = r["cfg"]
        if c["dtype"] == "int32" or r["reach"]["status"] != "ok" or not sel(r): continue
        names = set(sweep.base_name(n[0]) for n in sweep.patterns_for(c))
        if pat not in names: continue
        d = by[r["key"]]
        g[keyf(r)][0] += 1
        g[keyf(r)][1] += int(pat in d["viol_pats"])
    print("\n== %s: pattern %s ==" % (title, pat))
    for k in sorted(g, key=str):
        n, v = g[k]
        print("  %-26s tested %4d  violating %4d  rate %.3f" % (k, n, v, v / n))
for pat in ("nan1_last", "ties_max_nan"):
    tbl(pat, lambda r: r["reach"]["sig"], "by stage signature")
    tbl(pat, lambda r: "reg%d" % min(r["reach"]["reg_bits"], 3), "by in-thread bits (capped 3)")
    tbl(pat, lambda r: "nbits%d" % r["reach"]["nbits"], "by nbits")
tbl("nan1_last", lambda r: (r["cfg"]["axis"], "dtype=" + r["cfg"]["dtype"]), "by axis,dtype")
tbl("nan1_last", lambda r: (r["cfg"]["axis"], r["cfg"]["nw"]), "by axis,nw")

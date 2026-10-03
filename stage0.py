"""Stage 0 of diff-sweep v2: reduction reach of the whole v1 arg-op population, GPU not needed.

Spec: vault triton_00/notes/27-차등스윕v2-명세.md §2 (pre-registered, decision table fixed there).
Runs on a Colab CPU session with the main@93ba8872 wheel installed. Run from a directory that is
NOT an ancestor of the triton source (cwd=/root), see vault note 05-브라우저-노트북-워크플로우.

  python3 stage0.py selftest            # reach.py pure-logic self-test, run FIRST
  python3 stage0.py probe               # compile the upstream test_argmax_argmin_with_nan kernels
  python3 stage0.py run [--workers N] [--out /root/out] [--limit K]

Population = every arg-op config of v1's build_plan() (violating ones come from dump_a_cfgs.json.gz).
tie_break (TB) does not change the layout by assumption; --tbcheck verifies that on a sample.
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reach  # noqa: E402

PTR = {"fp32": "*fp32", "fp16": "*fp16", "bf16": "*bf16", "int32": "*i32"}
OPCODE = {"max": 0, "min": 1, "sum": 2, "argmax": 3, "argmin": 4}
DIV16 = {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]], (2,): [["tt.divisibility", 16]]}


def population():
    """All arg-op configs of v1 build_plan(), deduplicated over TB, plus violation info from DUMP_A."""
    import sweep  # numpy-only logic from v1, unchanged
    plan = sweep.build_plan()
    with gzip.open(os.path.join(HERE, "dump_a_cfgs.json.gz"), "rt") as f:
        dump = json.load(f)
    assert len(plan) == dump["n_cfg"], "plan %d != dump n_cfg %d (plan regenerated with a different sweep.py?)" % (len(plan), dump["n_cfg"])
    viol = {c["key"]: c for c in dump["cfgs"]}
    keys = {sweep.cfg_key(c) for c in plan}
    missing = [k for k in viol if k not in keys]
    assert not missing, "dump keys not in regenerated plan: %s" % missing[:3]
    arg = [c for c in plan if c["op"] in sweep.ARG_OPS]
    return arg, viol, sweep.cfg_key


def _layout_key(c):
    return (c["dim"], tuple(c["shape"]), c["axis"], c["dtype"], c["op"], c["nw"])


def compile_cfg(c, cache_dir):
    """Compile one config for sm_89 without a GPU. Returns dict(ttgir, ptx) or dict(error)."""
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    import sweep_kernels as K
    t = PTR[c["dtype"]]
    tb = True if c["tb"] is None else bool(c["tb"])
    if c["dim"] == 1:
        sig = {"x_ptr": t, "val_ptr": t, "idx_ptr": "*i32", "N": "constexpr", "OP": "constexpr", "TB": "constexpr"}
        cx = {"N": c["shape"][0], "OP": OPCODE[c["op"]], "TB": tb}
        fn = K.kern1d
    else:
        sig = {"x_ptr": t, "val_ptr": t, "idx_ptr": "*i32", "M": "constexpr", "N": "constexpr",
               "AXIS": "constexpr", "OP": "constexpr", "TB": "constexpr"}
        cx = {"M": c["shape"][0], "N": c["shape"][1], "AXIS": c["axis"], "OP": OPCODE[c["op"]], "TB": tb}
        fn = K.kern2d
    try:
        k = triton_compile(fn, sig, cx, c["nw"], ASTSource, GPUTarget, triton)
        return dict(ttgir=k.asm["ttgir"], ptx=k.asm["ptx"])
    except Exception as e:  # ptxas crashes etc.: record, do not guess
        return dict(error="%s: %s" % (type(e).__name__, str(e).replace("\n", " | ")[:300]))


def triton_compile(fn, sig, cx, nw, ASTSource, GPUTarget, triton):
    src = ASTSource(fn=fn, signature=sig, constexprs=cx, attrs=DIV16)
    return triton.compile(src, target=GPUTarget("cuda", 89, 32), options={"num_warps": nw})


def classify(c, cache_dir):
    r = compile_cfg(c, cache_dir)
    rec = dict(layout_key=list(map(str, _layout_key(c))))
    if "error" in r:
        rec.update(status="compile_exc", reason=r["error"])
        return rec
    d = reach.analyze_ttgir(r["ttgir"])
    rec.update(d)
    if d["status"] == "ok":
        bad = reach.ptx_check(d, reach.ptx_features(r["ptx"]))
        if bad:
            rec.update(status="unclassified", reason="ptx_mismatch: " + "; ".join(bad), model_status="ok")
    else:
        rec["ttgir_head"] = r["ttgir"][:1500]
    return rec


def _worker(c):
    with tempfile.TemporaryDirectory() as td:
        return c, classify(c, td)


def cmd_run(a):
    arg, viol, cfg_key = population()
    seen, uniq = set(), []
    for c in arg:
        k = _layout_key(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    if a.limit:
        uniq = uniq[:a.limit]
    print("arg-op configs %d, layout classes (TB-deduped) %d" % (len(arg), len(uniq)), flush=True)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    results = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (c, rec) in enumerate(pool.imap_unordered(_worker, uniq, chunksize=1)):
            results[_layout_key(c)] = rec
            if (i + 1) % 50 == 0 or i + 1 == len(uniq):
                print("  %d/%d  %.0fs" % (i + 1, len(uniq), time.time() - t0), flush=True)
    # TB assumption check on a sample (compile the other TB value, expect an identical split)
    tbcheck = []
    if a.tbcheck:
        rng = random.Random(5)
        sample = [c for c in arg if c["tb"] is not None]
        rng.shuffle(sample)
        for c in sample[:a.tbcheck]:
            c2 = dict(c, tb=not c["tb"])
            with tempfile.TemporaryDirectory() as td:
                r2 = classify(c2, td)
            r1 = results.get(_layout_key(c))
            keys = ("status", "reg_bits", "lane_bits", "warp_bits")
            tbcheck.append((cfg_key(c), all(r1.get(k) == r2.get(k) for k in keys)))
    rows = []
    for c in arg:
        rec = results.get(_layout_key(c))
        if rec is None:
            continue
        v = viol.get(cfg_key(c))
        rows.append(dict(key=cfg_key(c), cfg=c, reach={k: rec.get(k) for k in rec if k != "ttgir_head"},
                         violating=bool(v and v.get("kinds")), viol_launches=(v or {}).get("viol", 0)))
    with open(os.path.join(a.out, "stage0.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "stage0_unclassified_heads.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items() if v.get("status") != "ok"}, f)
    report(rows, tbcheck)


def report(rows, tbcheck):
    st = Counter(r["reach"].get("status") for r in rows)
    print("\n== coverage (arg-op configs, TB not deduped) ==")
    for k, n in st.most_common():
        print("  %-14s %d" % (k, n))
    reasons = Counter(str(r["reach"].get("reason"))[:90] for r in rows if r["reach"].get("status") != "ok")
    for k, n in reasons.most_common(6):
        print("    reason x%-4d %s" % (n, k))
    if tbcheck:
        print("  TB assumption: %d/%d sampled pairs identical split" % (sum(1 for _, ok in tbcheck if ok), len(tbcheck)))
    ok = [r for r in rows if r["reach"].get("status") == "ok"]
    unc = len(rows) - len(ok)
    print("  unclassified fraction: %.3f (spec: >= 0.5 => park D1)" % (unc / max(len(rows), 1)))

    def table(title, keyf):
        g = defaultdict(lambda: [0, 0, 0])
        for r in ok:
            k = keyf(r["reach"])
            g[k][0] += 1
            g[k][1] += int(r["violating"])
            g[k][2] += r["viol_launches"]
        print("\n== %s ==" % title)
        print("  %-28s %6s %9s %7s %12s" % ("class", "cfgs", "violating", "rate", "viol_launch"))
        for k in sorted(g, key=str):
            n, v, vl = g[k]
            print("  %-28s %6d %9d %7.3f %12d" % (k, n, v, v / n, vl))

    table("by stage signature (R in-thread, L lane shuffle, W cross-warp)", lambda d: d["sig"])
    table("by (nbits, reg, lane, warp)", lambda d: (d["nbits"], d["reg_bits"], d["lane_bits"], d["warp_bits"]))
    table("by nbits (log2 of reduced axis)", lambda d: d["nbits"])
    one = [r for r in ok if len(r["reach"]["sig"].replace("-", "")) == 1]
    print("\n== single-stage-kind configs (spec: 'shallow') ==")
    print("  cfgs %d, violating %d" % (len(one), sum(int(r["violating"]) for r in one)))
    for r in one:
        if r["violating"]:
            print("   VIOLATING single-stage:", r["key"], r["reach"]["sig"])
            break


# ----------------------------------------------------------------------------
# probe: upstream test_argmax_argmin_with_nan (3 elements in BLOCK=4, masked load, nw=4 default)
# ----------------------------------------------------------------------------
def cmd_probe(a):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    import probe_kernels as P
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
    out = []
    for name, fn in (("argmax_kernel", P.argmax_kernel), ("argmin_kernel", P.argmin_kernel)):
        for dt in ("fp32", "bf16"):
            t = PTR[dt]
            sig = {"x_ptr": t, "val_ptr": t, "idx_ptr": "*i32", "N": "constexpr", "BLOCK": "constexpr"}
            k = triton_compile(fn, sig, {"N": 3, "BLOCK": 4}, 4, ASTSource, GPUTarget, triton)
            d = reach.analyze_ttgir(k.asm["ttgir"])
            feats = reach.ptx_features(k.asm["ptx"])
            bad = reach.ptx_check(d, feats) if d["status"] == "ok" else ["(unclassified)"]
            out.append((name, dt, d, feats, bad))
            print("PROBE %s %s: status=%s sig=%s reg/lane/warp=%s/%s/%s bcast_lane=%s bcast_warp=%s layout spt=%s tpw=%s wpc=%s ptx=%s check=%s" % (
                name, dt, d["status"], d.get("sig"), d.get("reg_bits"), d.get("lane_bits"), d.get("warp_bits"),
                d.get("bcast_lane_bits"), d.get("bcast_warp_bits"), d.get("spt"), d.get("tpw"), d.get("wpc"), feats, bad))
            if d["status"] != "ok":
                print("   reason:", d["reason"])
                print("   TTGIR head:\n" + k.asm["ttgir"][:1500])
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("probe")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--out", default="/root/out")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--tbcheck", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "selftest":
        reach.selftest()
    elif a.cmd == "probe":
        cmd_probe(a)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()

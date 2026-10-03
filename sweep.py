"""Differential sweep v1: compiled Triton reductions vs a float64 CPU oracle.

Spec: vault triton_00/notes/26-차등스윕v1-파일럿-명세.md (pre-registered).
Pure-logic part (oracle, patterns, plan, clustering) imports only numpy, so the
self-test runs before triton is built. GPU part imports torch/triton lazily and
is always executed in a subprocess (fault isolation).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPCODE = {"max": 0, "min": 1, "sum": 2, "argmax": 3, "argmin": 4}
ARG_OPS = ("argmax", "argmin")
SHAPES_2D = [(8, 8), (16, 32), (32, 16), (64, 64), (128, 32), (32, 128), (4, 256), (256, 4)]
SHAPES_1D = [8, 32, 128, 1024]
NWARPS = (1, 2, 4, 8)
DTYPES = ("fp32", "fp16", "bf16", "int32")


# ----------------------------------------------------------------------------
# dtype rounding and value pools (numpy only)
# ----------------------------------------------------------------------------
def round_to(dtype, a):
    a = np.asarray(a, dtype=np.float64)
    if dtype == "fp32":
        return a.astype(np.float32).astype(np.float64)
    if dtype == "fp16":
        return a.astype(np.float16).astype(np.float64)
    if dtype == "bf16":
        f = np.ascontiguousarray(a.astype(np.float32))
        u = f.view(np.uint32)
        rb = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
        r = ((u + rb) & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)
        return np.where(np.isnan(f), np.nan, r)
    if dtype == "int32":
        return np.rint(a)
    raise ValueError(dtype)


_POOL = {}


def value_pool(dtype):
    if dtype not in _POOL:
        if dtype == "int32":
            p = np.arange(-1500, 1501).astype(np.float64)
        else:
            p = np.unique(round_to(dtype, np.arange(-2048, 2049) * 0.5))
        _POOL[dtype] = p
    return _POOL[dtype]


# ----------------------------------------------------------------------------
# config geometry and patterns
# ----------------------------------------------------------------------------
def cfg_key(c):
    return "%dd|%s|ax%s|%s|%s|tb%s|nw%d" % (c["dim"], "x".join(map(str, c["shape"])), c["axis"], c["dtype"],
                                            c["op"], c["tb"], c["nw"])


def geometry(c):
    """returns (R reduce length, L lanes, actual shape, axis)"""
    if c["dim"] == 1:
        R = c["shape"][0]
        return R, 1, (R,), 0
    M, N = c["shape"]
    if c["axis"] == 0:
        return M, N, (M, N), 0
    return N, M, (M, N), 1


def positions_1d(R):
    if R <= 32:
        return list(range(R))
    r = random.Random(R)
    s = {0, 1, R // 2, R - 2, R - 1}
    while len(s) < 24:
        s.add(r.randrange(R))
    return sorted(s)


def patterns_for(c):
    dtype, op = c["dtype"], c["op"]
    R = geometry(c)[0]
    if op == "sum":
        pats = [("sum_sparse", None)]
        if dtype != "int32":
            pats += [("sum_nan", None), ("sum_posinf", None), ("sum_infmix", None)]
        return pats
    mx = op in ("max", "argmax")
    tie = "ties_max" if mx else "ties_min"
    pats = [("distinct", None), (tie, None), (tie + "#1", None), ("zeros", None)]
    if dtype == "int32":
        return pats
    if c["dim"] == 1:
        pats += [("nan1_pos", p) for p in positions_1d(R)]
    else:
        pats += [("nan1_last", None)] + [("nan1_rand#%d" % i, None) for i in range(3)]
    pats += [("nank#0", None), ("nank#1", None), ("posinf", None), ("neginf", None), ("infmix", None),
             (tie + "_nan", None), ("allnan", None)]
    return pats


def base_name(name):
    return name.split("#")[0]


def gen_pattern(dtype, name, R, L, rng, pos=None):
    """float64 array (R, L), reduction axis first; values exactly representable in dtype."""
    b = base_name(name)
    if b.startswith("sum_"):
        xs = np.zeros((R, L))
        k = min(R, 64)
        vals = np.array([-2.0, -1.0, 1.0, 2.0])
        for l in range(L):
            p = rng.choice(R, size=k, replace=False)
            xs[p, l] = rng.choice(vals, size=k)
        for l in range(L):
            q = rng.choice(R, size=2, replace=False) if R >= 2 else np.array([0, 0])
            if b == "sum_nan":
                xs[q[0], l] = np.nan
            elif b == "sum_posinf":
                xs[q[0], l] = np.inf
            elif b == "sum_infmix":
                xs[q[0], l] = np.inf
                xs[q[1], l] = -np.inf
        return xs
    pool = value_pool(dtype)
    xs = np.stack([rng.choice(pool, size=R, replace=False) for _ in range(L)], axis=1)
    if b == "distinct":
        return xs
    if b == "zeros":
        if dtype == "int32":
            return np.zeros((R, L))
        return rng.choice(np.array([0.0, -0.0]), size=(R, L))
    if b == "allnan":
        return np.full((R, L), np.nan)
    if b == "nan1_last":
        xs[R - 1, :] = np.nan
    elif b == "nan1_pos":
        xs[pos, :] = np.nan
    elif b == "nan1_rand":
        for l in range(L):
            xs[rng.integers(R), l] = np.nan
    elif b == "nank":
        m = rng.random((R, L)) < 0.25
        for l in range(L):
            if m[:, l].all():
                m[rng.integers(R), l] = False
        xs[m] = np.nan
    elif b == "posinf":
        for l in range(L):
            xs[rng.integers(R), l] = np.inf
    elif b == "neginf":
        for l in range(L):
            xs[rng.integers(R), l] = -np.inf
    elif b == "infmix":
        for l in range(L):
            q = rng.choice(R, size=2, replace=False)
            xs[q[0], l] = np.inf
            xs[q[1], l] = -np.inf
    elif b in ("ties_max", "ties_min", "ties_max_nan", "ties_min_nan"):
        use_max = b.startswith("ties_max")
        k = min(3, R)
        for l in range(L):
            m = xs[:, l].max() if use_max else xs[:, l].min()
            p = rng.choice(R, size=k, replace=False)
            xs[p, l] = m
            if b.endswith("_nan") and R > k:
                rest = np.setdiff1d(np.arange(R), p)
                xs[rng.choice(rest), l] = np.nan
    else:
        raise ValueError(name)
    return xs


# ----------------------------------------------------------------------------
# oracle (contract table of spec section 2)
# ----------------------------------------------------------------------------
def judge(op, tb, xs, got_val, got_idx):
    """xs float64 (R, L). returns list of (lane, status, kind); ok lanes are omitted."""
    R, L = xs.shape
    out = []
    with np.errstate(all="ignore"):
        for l in range(L):
            col = xs[:, l]
            nn = ~np.isnan(col)
            if op != "sum" and not nn.any():
                out.append((l, "unspecified", "allnan"))
                continue
            if op == "sum":
                exp = float(np.sum(col))
                g = float(got_val[l])
                if math.isnan(exp):
                    if not math.isnan(g):
                        out.append((l, "violation", "nan_lost"))
                elif math.isnan(g):
                    out.append((l, "violation", "nan_leak"))
                elif g != exp:
                    out.append((l, "violation", "wrong_value"))
                continue
            ext = col[nn].max() if op in ("max", "argmax") else col[nn].min()
            if op in ("max", "min"):
                g = float(got_val[l])
                if math.isnan(g):
                    out.append((l, "violation", "nan_leak"))
                elif g != ext:
                    out.append((l, "violation", "wrong_value"))
                continue
            g = int(got_idx[l])
            cands = np.flatnonzero(nn & (col == ext))
            if g < 0 or g >= R:
                out.append((l, "violation", "wrong_index"))
            elif math.isnan(col[g]):
                out.append((l, "violation", "nan_index"))
            elif col[g] != ext:
                out.append((l, "violation", "wrong_index"))
            elif tb and g != int(cands[0]):
                out.append((l, "violation", "tie_violation"))
    return out


# ----------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------
def _variants(dtype):
    ops = ["max", "min", "sum", "argmax", "argmin"] if dtype in ("fp32", "int32") else ["sum", "argmax", "argmin"]
    for op in ops:
        if op in ARG_OPS:
            yield op, True
            yield op, False
        else:
            yield op, None


def build_plan():
    tiers = {1: [], 2: [], 3: []}

    def add(t, dim, shape, axis, dtype, op, tb, nw):
        tiers[t].append(dict(tier=t, dim=dim, shape=list(shape), axis=axis, dtype=dtype, op=op, tb=tb, nw=nw))

    for dt in ("fp16", "bf16"):
        for sh in SHAPES_2D:
            for ax in (0, 1):
                for nw in NWARPS:
                    for op, tb in _variants(dt):
                        add(1, 2, sh, ax, dt, op, tb, nw)
    for dt in ("fp32", "int32"):
        for sh in SHAPES_2D:
            for ax in (0, 1):
                for nw in NWARPS:
                    for op, tb in _variants(dt):
                        add(2, 2, sh, ax, dt, op, tb, nw)
    for dt in DTYPES:
        for n in SHAPES_1D:
            for nw in NWARPS:
                for op, tb in _variants(dt):
                    add(3, 1, (n,), 0, dt, op, tb, nw)
    for dt in ("fp16", "bf16"):
        for sh in ((8, 8), (64, 64)):
            for ax in (0, 1):
                for op in ("max", "min"):
                    add(3, 2, sh, ax, dt, op, None, 4)
    rr = random.Random(7)
    plan = []
    for t in (1, 2, 3):
        rr.shuffle(tiers[t])
        plan += tiers[t]
    return plan


def smoke_cfgs():
    def m(dim, shape, axis, dtype, op, tb, nw):
        return dict(tier=0, dim=dim, shape=list(shape), axis=axis, dtype=dtype, op=op, tb=tb, nw=nw)
    return [
        m(1, (32,), 0, "fp32", "argmax", True, 4),   # positive control
        m(1, (32,), 0, "fp32", "argmin", True, 4),   # positive control
        m(1, (128,), 0, "fp32", "argmax", True, 1),
        m(1, (32,), 0, "fp32", "max", None, 4),
        m(1, (32,), 0, "fp32", "sum", None, 4),
        m(1, (32,), 0, "int32", "argmax", True, 4),
        m(2, (16, 32), 0, "fp16", "argmax", True, 4),
        m(2, (32, 16), 1, "bf16", "argmin", True, 4),
        m(2, (16, 32), 1, "fp16", "sum", None, 2),
        m(2, (64, 64), 0, "int32", "argmax", True, 4),
    ]


# ----------------------------------------------------------------------------
# GPU part (subprocess only)
# ----------------------------------------------------------------------------
def run_config(c, seed_base, fh, stage, only=None):
    import torch
    sys.path.insert(0, HERE)
    import sweep_kernels as K
    key = cfg_key(c)
    R, L, shape, ax = geometry(c)
    tdt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "int32": torch.int32}[c["dtype"]]
    is_int = c["dtype"] == "int32"
    tb = True if c["tb"] is None else bool(c["tb"])
    pats = patterns_for(c)
    npat_total = len(pats)
    only_set = None
    if only is not None:
        only_set = {(only[0], only[1])}
    elif c.get("only_pats"):
        only_set = {(p[0], p[1]) for p in c["only_pats"]}
    if only_set is not None:
        pats = [p for p in pats if (p[0], p[1]) in only_set]
    st = dict(launches=0, ok=0, viol=0, unspec=0)
    viol_names = []
    kinds = Counter()
    pk = defaultdict(Counter)
    viol_pats = set()
    seen_rec = set()
    t0 = time.time()
    compile_s = None
    crash = None
    fatal = False
    for name, pos in pats:
        rng = np.random.default_rng(zlib.crc32(("%s|%s|%s|%s" % (key, name, pos, seed_base)).encode()))
        xs = gen_pattern(c["dtype"], name, R, L, rng, pos)
        actual = xs[:, 0] if c["dim"] == 1 else (xs if ax == 0 else xs.T)
        actual = np.ascontiguousarray(actual)
        src = actual.astype(np.int64) if is_int else actual
        xt = torch.from_numpy(src).to(tdt).to("cuda")
        back = xt.to(torch.float64).cpu().numpy()
        if not np.array_equal(back, actual, equal_nan=True):
            kinds["harness_conv"] += 1
            continue
        val = torch.full((L,), 1234567 if is_int else -12345.0, dtype=tdt, device="cuda")
        idx = torch.full((L,), -1, dtype=torch.int32, device="cuda")
        t1 = time.time()
        try:
            if c["dim"] == 1:
                K.kern1d[(1,)](xt, val, idx, N=R, OP=OPCODE[c["op"]], TB=tb, num_warps=c["nw"])
            else:
                K.kern2d[(1,)](xt, val, idx, M=shape[0], N=shape[1], AXIS=ax, OP=OPCODE[c["op"]], TB=tb,
                               num_warps=c["nw"])
            torch.cuda.synchronize()
        except Exception as e:  # noqa
            crash = (type(e).__name__ + ": " + str(e))[:400]
            fatal = ("CUDA" in str(e)) or ("illegal" in str(e).lower())
            break
        if compile_s is None:
            compile_s = time.time() - t1
        gv = val.to(torch.float64).cpu().numpy()
        gi = idx.cpu().numpy()
        res = judge(c["op"], tb, xs, gv, gi)
        st["launches"] += 1
        viol = [r for r in res if r[1] == "violation"]
        if viol:
            st["viol"] += 1
            bn = base_name(name)
            viol_pats.add(bn)
            viol_names.append([name, pos])
            for r in viol:
                kinds[r[2]] += 1
                pk[bn][r[2]] += 1
            for lane, _s, kind in viol:
                if (bn, kind) not in seen_rec and len(seen_rec) < 24:
                    seen_rec.add((bn, kind))
                    got = int(gi[lane]) if c["op"] in ARG_OPS else float(gv[lane])
                    fh.write(json.dumps(dict(t="viol", stage=stage, key=key, cfg=c, pat=name, pos=pos, lane=lane,
                                             kind=kind, col=xs[:, lane].tolist(), got=got)) + "\n")
        elif res:
            st["unspec"] += 1
        else:
            st["ok"] += 1
    fh.write(json.dumps(dict(t="cfg", stage=stage, key=key, cfg=c, tier=c.get("tier"), compile_s=compile_s,
                             crash=crash, sec=time.time() - t0, kinds=dict(kinds), viol_pats=sorted(viol_pats),
                             pat_kinds={p: dict(v) for p, v in pk.items()}, viol_names=viol_names,
                             npat=len(pats), npat_total=npat_total, **st)) + "\n")
    fh.flush()
    return "fatal" if fatal else "ok"


def cli_run(a):
    cfgs = json.load(open(a.cfgs))
    fh = open(a.out, "a")
    for c in cfgs:
        if run_config(c, a.seed, fh, a.stage) == "fatal":
            sys.exit(3)
    print("DONE")


def cli_repro(a):
    c = json.loads(a.cfg)
    pos = json.loads(a.pos)
    tmp = a.out
    fh = open(tmp, "w")
    run_config(c, a.seed, fh, a.stage, only=(a.pattern, pos))
    fh.close()
    recs = [json.loads(x) for x in open(tmp)]
    cfg_rec = [r for r in recs if r["t"] == "cfg"][0]
    print("REPRO " + json.dumps(dict(viol=cfg_rec["viol"], kinds=cfg_rec["kinds"], crash=cfg_rec["crash"])))


# ----------------------------------------------------------------------------
# driver (runs in the notebook kernel; never imports torch/triton)
# ----------------------------------------------------------------------------
def _spawn(cfg_path, out_path, stage, seed, timeout, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [sys.executable, os.path.join(HERE, "sweep.py"), "run", "--cfgs", cfg_path, "--out", out_path,
           "--stage", stage, "--seed", str(seed)]
    try:
        r = subprocess.run(cmd, cwd="/tmp", env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stderr[-500:]
    except subprocess.TimeoutExpired:
        return -9, "timeout"


def _read(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def load_stage(stage, out_dir="/content/out"):
    recs = []
    if os.path.isdir(out_dir):
        for fn in sorted(os.listdir(out_dir)):
            if fn.startswith(stage + "_chunk") and fn.endswith(".jsonl"):
                recs += _read(os.path.join(out_dir, fn))
    return [r for r in recs if r["t"] == "cfg"], [r for r in recs if r["t"] == "viol"]


def run_stage(cfgs, stage, budget_s, chunk_size=30, seed=0, out_dir="/content/out", env_extra=None):
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    chunks = [cfgs[i:i + chunk_size] for i in range(0, len(cfgs), chunk_size)]
    bisect_budget = 600.0
    tot = Counter()
    kinds_tot = Counter()
    for ci, ch in enumerate(chunks):
        if time.time() - t_start > budget_s:
            print("[%s] budget exhausted; stopping before chunk %d/%d" % (stage, ci + 1, len(chunks)), flush=True)
            break
        cfg_path = "%s/%s_chunk%d.cfgs.json" % (out_dir, stage, ci)
        out_path = "%s/%s_chunk%d.jsonl" % (out_dir, stage, ci)
        json.dump(ch, open(cfg_path, "w"))
        open(out_path, "w").close()
        remaining = budget_s - (time.time() - t_start)
        rc, err = _spawn(cfg_path, out_path, stage, seed, max(300, min(1200, remaining + 300)), env_extra)
        recs = _read(out_path)
        seen = {r["key"] for r in recs if r["t"] == "cfg"}
        missing = [c for c in ch if cfg_key(c) not in seen]
        if missing:
            print("[%s] chunk %d rc=%s missing=%d err=%s" % (stage, ci + 1, rc, len(missing), err.replace("\n", " ")[-200:]),
                  flush=True)
            for c in missing:
                if bisect_budget <= 0:
                    with open(out_path, "a") as fh:
                        fh.write(json.dumps(dict(t="cfg", stage=stage, key=cfg_key(c), cfg=c, tier=c.get("tier"),
                                                 crash="unlocalized_process_failure", launches=0, ok=0, viol=0,
                                                 unspec=0, kinds={}, viol_pats=[], pat_kinds={}, sec=0,
                                                 compile_s=None)) + "\n")
                    continue
                one = "%s/%s_chunk%d_one.cfgs.json" % (out_dir, stage, ci)
                json.dump([c], open(one, "w"))
                tb0 = time.time()
                rc1, err1 = _spawn(one, out_path, stage, seed, 300, env_extra)
                bisect_budget -= time.time() - tb0
                if cfg_key(c) not in {r["key"] for r in _read(out_path) if r["t"] == "cfg"}:
                    with open(out_path, "a") as fh:
                        fh.write(json.dumps(dict(t="cfg", stage=stage, key=cfg_key(c), cfg=c, tier=c.get("tier"),
                                                 crash="process_died rc=%s %s" % (rc1, err1.replace("\n", " ")[-160:]),
                                                 launches=0, ok=0, viol=0, unspec=0, kinds={}, viol_pats=[],
                                                 pat_kinds={}, sec=0, compile_s=None)) + "\n")
            recs = _read(out_path)
        crecs = [r for r in recs if r["t"] == "cfg"]
        ck = Counter()
        for r in crecs:
            tot["cfgs"] += 1
            tot["launches"] += r.get("launches", 0)
            tot["ok"] += r.get("ok", 0)
            tot["viol"] += r.get("viol", 0)
            tot["unspec"] += r.get("unspec", 0)
            tot["crash"] += 1 if r.get("crash") else 0
            for k, v in (r.get("kinds") or {}).items():
                ck[k] += v
                kinds_tot[k] += v
        print("[%s %d/%d] t=%dm cfgs=%d launches=%d ok=%d viol=%d unspec=%d crash=%d | chunk kinds=%s" % (
            stage, ci + 1, len(chunks), (time.time() - t_start) / 60, tot["cfgs"], tot["launches"], tot["ok"],
            tot["viol"], tot["unspec"], tot["crash"], dict(ck)), flush=True)
    return dict(stats=dict(tot), kinds=dict(kinds_tot), elapsed_s=time.time() - t_start, planned=len(cfgs))


def coverage_report(plan, cfg_recs):
    done = {r["key"] for r in cfg_recs}
    by = defaultdict(lambda: [0, 0])
    for c in plan:
        by[c["tier"]][0] += 1
        if cfg_key(c) in done:
            by[c["tier"]][1] += 1
    lines = ["tier %d: executed %d / planned %d" % (t, by[t][1], by[t][0]) for t in sorted(by)]
    crashes = [r for r in cfg_recs if r.get("crash")]
    lines.append("crash configs: %d" % len(crashes))
    return "\n".join(lines)


NAN_PATS = {"nan1_pos", "nan1_last", "nan1_rand", "nank", "ties_max_nan", "ties_min_nan"}


def _is_11993_like(c, pat, kind):
    return c["op"] in ARG_OPS and pat in NAN_PATS and kind in ("nan_index", "wrong_index")


def select_stage_b(plan, cfg_recs, frac=0.1, seed=3):
    """Priority: crashes, then configs with violations that do NOT look like #11993, then a clean-config sample
    (patch regression check), then #11993-looking configs (tier-1 first). Violating configs rerun only the
    patterns that violated in A."""
    by_key = {cfg_key(c): c for c in plan}
    rr = random.Random(seed)
    crash, other, clean, rest = [], [], [], []
    for r in cfg_recs:
        c = by_key.get(r["key"])
        if c is None:
            continue
        cc = dict(c)
        if r.get("crash"):
            crash.append(cc)
        elif r.get("viol", 0) > 0:
            cc["only_pats"] = r.get("viol_names") or None
            like_only = all(_is_11993_like(c, p, k) for p, kd in (r.get("pat_kinds") or {}).items() for k in kd)
            (rest if like_only else other).append(cc)
        else:
            clean.append(cc)
    for lst in (crash, other, clean, rest):
        rr.shuffle(lst)
    clean = clean[: int(len(clean) * frac)]
    rest.sort(key=lambda c: c["tier"])
    return crash + other + clean + rest


def stage_b_eta(planB, cfg_recs):
    A = {r["key"]: r for r in cfg_recs}
    tot = 0.0
    for c in planB:
        r = A.get(cfg_key(c))
        if not r:
            continue
        npat = max(1, r.get("npat_total") or 1)
        nrun = len(c["only_pats"]) if c.get("only_pats") else npat
        comp = r.get("compile_s") or 0.0
        tot += comp + max(0.0, r.get("sec", 0.0) - comp) * nrun / npat
    return tot


def _shape_class(c):
    return "1D" if c["dim"] == 1 else "2D-ax%d" % c["axis"]


def _triples(rec):
    t = {(p, k) for p, kd in (rec.get("pat_kinds") or {}).items() for k in kd}
    if rec.get("crash"):
        t.add(("*", "crash"))
    return t


def cluster_report(cfg_a, cfg_b, viol_a, viol_b):
    """cfg-level comparison of stage A (unpatched) and B (patched): per (config, pattern, kind)."""
    A = {r["key"]: r for r in cfg_a}
    B = {r["key"]: r for r in cfg_b}
    table = defaultdict(Counter)
    members = defaultdict(list)
    for key in set(A) | set(B):
        a, b = A.get(key), B.get(key)
        c = (a or b)["cfg"]
        ta = _triples(a) if a else set()
        tb_ = _triples(b) if b else set()
        for (p, k) in ta | tb_:
            ck = (c["op"], c["dtype"], p, k, _shape_class(c))
            if (p, k) in ta and (p, k) in tb_:
                st = "persistent"
            elif (p, k) in ta:
                st = "cleared_by_patch" if b else "not_rerun"
            else:
                st = "introduced_by_patch"
            table[ck][st] += 1
            if st in ("persistent", "introduced_by_patch"):
                members[ck].append((math.prod(c["shape"]), c["nw"], key, p, k, c))
    rows = sorted(table.items(), key=lambda kv: -(kv[1]["persistent"] + kv[1]["introduced_by_patch"]))
    rep = {}
    for ck, _cnt in rows:
        if ck not in members:
            continue
        mem = members[ck]
        size, nw, key, p, k, c = min(mem, key=lambda m: (m[0], m[1], m[2]))
        v = None
        for pool in (viol_b, viol_a):
            for x in pool:
                if x["key"] == key and base_name(x["pat"]) == p and x["kind"] == k:
                    v = x
                    break
            if v:
                break
        if v is None and p == "*":
            name, pos = patterns_for(c)[0]
            v = dict(cfg=c, pat=name, pos=pos, key=key, kind=k)
        rep[ck] = (size, nw, v)
    return rows, rep


def print_clusters(rows):
    print("%-8s %-6s %-14s %-14s %-8s | %s" % ("op", "dtype", "pattern", "kind", "shape", "cleared / persistent / introduced / not_rerun"))
    for ck, c in rows:
        print("%-8s %-6s %-14s %-14s %-8s | %d / %d / %d / %d" % (ck + (c["cleared_by_patch"], c["persistent"],
                                                                     c["introduced_by_patch"], c["not_rerun"])))


def repro_check(rep, stage, n=3, out_dir="/content/out", env_extra=None, max_clusters=15, budget_s=600):
    results = {}
    t_start = time.time()
    for ci, (ck, (_size, _nw, v)) in enumerate(rep.items()):
        if ci >= max_clusters or time.time() - t_start > budget_s:
            results[ck] = (False, ["skipped (cluster cap or repro budget)"])
            continue
        if v is None:
            results[ck] = (False, ["no representative record"])
            continue
        outs = []
        for i in range(n):
            env = dict(os.environ)
            env.update(env_extra or {})
            cmd = [sys.executable, os.path.join(HERE, "sweep.py"), "repro", "--cfg", json.dumps(v["cfg"]),
                   "--pattern", v["pat"], "--pos", json.dumps(v["pos"]), "--seed", "0", "--stage", stage,
                   "--out", "%s/repro_%d.jsonl" % (out_dir, i)]
            try:
                r = subprocess.run(cmd, cwd="/tmp", env=env, capture_output=True, text=True, timeout=300)
                line = [x for x in r.stdout.splitlines() if x.startswith("REPRO ")]
                outs.append(line[0][6:] if line else "NOOUT rc=%s %s" % (r.returncode, r.stderr[-120:]))
            except subprocess.TimeoutExpired:
                outs.append("timeout")
        results[ck] = (len(set(outs)) == 1, outs)
    return results


# ----------------------------------------------------------------------------
# self-test (pure logic; runs before the triton build)
# ----------------------------------------------------------------------------
def selftest():
    nan, inf = float("nan"), float("inf")
    for dt in DTYPES:
        p = value_pool(dt)
        assert len(p) >= 1024, (dt, len(p))
        assert np.array_equal(round_to(dt, p), p), dt
        print("pool", dt, len(p))
    assert round_to("bf16", [257.0])[0] in (256.0, 258.0)
    assert round_to("bf16", [1.0, -3.0]).tolist() == [1.0, -3.0]

    def J(op, tb, col, got_val=None, got_idx=None):
        xs = np.array(col, dtype=float).reshape(-1, 1)
        gv = np.array([got_val], dtype=float) if got_val is not None else np.zeros(1)
        gi = np.array([got_idx]) if got_idx is not None else np.zeros(1, dtype=int)
        return judge(op, tb, xs, gv, gi)

    assert J("argmax", True, [3, nan, 5, 2], got_idx=2) == []
    assert J("argmax", True, [3, nan, 5, 2], got_idx=1)[0][2] == "nan_index"
    assert J("argmax", True, [3, nan, 5, 2], got_idx=0)[0][2] == "wrong_index"
    assert J("argmax", True, [5, 1, 5], got_idx=2)[0][2] == "tie_violation"
    assert J("argmax", False, [5, 1, 5], got_idx=2) == []
    assert J("argmin", True, [nan, 2, 1, 1], got_idx=2) == []
    assert J("argmin", True, [nan, 2, 1, 1], got_idx=3)[0][2] == "tie_violation"
    assert J("argmax", True, [-0.0, 0.0], got_idx=0) == []
    assert J("argmax", True, [-0.0, 0.0], got_idx=1)[0][2] == "tie_violation"
    assert J("argmax", True, [1, 2, 3], got_idx=-1)[0][2] == "wrong_index"
    assert J("max", None, [1, nan, 4], got_val=4.0) == []
    assert J("max", None, [1, nan, 4], got_val=nan)[0][2] == "nan_leak"
    assert J("min", None, [1, nan, 4], got_val=4.0)[0][2] == "wrong_value"
    assert J("sum", None, [1, inf], got_val=inf) == []
    assert J("sum", None, [inf, -inf], got_val=nan) == []
    assert J("sum", None, [inf, -inf], got_val=0.0)[0][2] == "nan_lost"
    assert J("sum", None, [1, nan], got_val=nan) == []
    assert J("sum", None, [1, 2], got_val=nan)[0][2] == "nan_leak"
    assert J("argmax", True, [nan, nan], got_idx=1)[0][1] == "unspecified"
    print("judge unit tests ok")

    rng = np.random.default_rng(0)
    names = ["distinct", "zeros", "allnan", "nan1_last", "nan1_pos", "nan1_rand#0", "nank#0", "posinf", "neginf",
             "infmix", "ties_max", "ties_min", "ties_max_nan", "ties_min_nan", "sum_sparse", "sum_nan",
             "sum_posinf", "sum_infmix"]
    for dt in DTYPES:
        for nm in names:
            if dt == "int32" and nm not in ("distinct", "zeros", "ties_max", "ties_min", "sum_sparse"):
                continue
            xs = gen_pattern(dt, nm, 64, 8, rng, 3)
            assert xs.shape == (64, 8), (dt, nm)
            fin = xs[np.isfinite(xs)]
            assert np.array_equal(round_to(dt, fin), fin), (dt, nm, "not representable")
            if nm == "distinct":
                assert all(len(np.unique(xs[:, l])) == 64 for l in range(8)), (dt, nm)
            if nm.startswith("nank"):
                assert all((~np.isnan(xs[:, l])).any() for l in range(8)), (dt, nm)
            if nm in ("ties_max", "ties_max_nan"):
                assert all((xs[:, l] == np.nanmax(xs[:, l])).sum() >= 2 for l in range(8)), (dt, nm)
            if nm in ("ties_min", "ties_min_nan"):
                assert all((xs[:, l] == np.nanmin(xs[:, l])).sum() >= 2 for l in range(8)), (dt, nm)
            if nm == "sum_sparse":
                assert all(np.abs(np.cumsum(xs[:, l])).max() <= 128 for l in range(8)), (dt, nm)
    plan = build_plan()
    cnt = Counter(c["tier"] for c in plan)
    assert dict(cnt) == {1: 640, 2: 896, 3: 400}, dict(cnt)
    assert len({cfg_key(c) for c in plan}) == len(plan), "duplicate config keys"
    for c in plan + smoke_cfgs():
        pats = patterns_for(c)
        assert pats and len({(p[0], p[1]) for p in pats}) == len(pats), cfg_key(c)
    print("plan ok:", dict(cnt), "total", len(plan))
    selftest_driver()
    print("SELFTEST OK")




def _trim_viol(v):
    col = v.get("col")
    if col is not None and len(col) > 128:
        a = np.array(col, dtype=float)
        fin = a[np.isfinite(a)]
        v = dict(v)
        v.pop("col")
        v["colsum"] = dict(n=len(col), nan_pos=np.flatnonzero(np.isnan(a))[:64].tolist(),
                           inf_pos=np.flatnonzero(np.isinf(a))[:16].tolist(),
                           max_pos=(np.flatnonzero(a == fin.max())[:8].tolist() if fin.size else []),
                           min_pos=(np.flatnonzero(a == fin.min())[:8].tolist() if fin.size else []),
                           head=list(col[:16]))
    return v


def dump_compact(stage, cfg_recs, viol_recs):
    """gzip+base64 of violating/crash cfg records and one representative violation per cluster/shape (printable)."""
    import base64
    import gzip
    keep_cfg = [r for r in cfg_recs if r.get("viol", 0) or r.get("crash")]
    seen = set()
    keep_viol = []
    for v in sorted(viol_recs, key=lambda x: x["cfg"]["nw"]):
        c = v["cfg"]
        k = (c["op"], c["dtype"], base_name(v["pat"]), v["kind"], tuple(c["shape"]), c["axis"])
        if k in seen:
            continue
        seen.add(k)
        keep_viol.append(_trim_viol(v))
    payload = dict(stage=stage, n_cfg=len(cfg_recs), cfgs=keep_cfg, viols=keep_viol)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    b = base64.b64encode(gzip.compress(raw, 9)).decode()
    lines = [b[i:i + 120] for i in range(0, len(b), 120)]
    return "DUMP_%s_BEGIN raw_bytes=%d cfgs_kept=%d viols_kept=%d\n%s\nDUMP_%s_END" % (
        stage, len(raw), len(keep_cfg), len(keep_viol), "\n".join(lines), stage)


def distinct_report(cfg_recs):
    """Violations on distinct finite inputs: 1D = control failure warning, 2D = layout-bug candidates."""
    d1 = [r for r in cfg_recs if "distinct" in (r.get("viol_pats") or []) and r["cfg"]["dim"] == 1]
    d2 = [r for r in cfg_recs if "distinct" in (r.get("viol_pats") or []) and r["cfg"]["dim"] == 2]
    out = ["distinct-input violations: 1D=%d (CONTROL FAILURE if >0), 2D=%d (layout-bug candidates)" % (len(d1), len(d2))]
    for r in (d1 + d2)[:10]:
        out.append("  %s kinds=%s" % (r["key"], r["pat_kinds"].get("distinct")))
    return "\n".join(out)


def selftest_driver():
    """Exercises the driver/report functions with synthetic records (no GPU, no triton)."""
    import base64
    import gzip
    import tempfile
    from types import SimpleNamespace
    plan = build_plan()

    def pick(pred):
        return next(c for c in plan if pred(c))
    c1 = pick(lambda c: c["dim"] == 2 and c["dtype"] == "fp16" and c["op"] == "argmax" and c["tb"] is True)
    c2 = pick(lambda c: c["dim"] == 2 and c["dtype"] == "bf16" and c["op"] == "sum")
    c3 = pick(lambda c: c["dim"] == 1 and c["dtype"] == "fp32" and c["op"] == "max")
    c4 = pick(lambda c: c["dim"] == 2 and c["dtype"] == "fp32" and c["op"] == "argmin" and c["tb"] is True)
    c5 = pick(lambda c: c["dim"] == 2 and c["dtype"] == "int32" and c["op"] == "sum")
    c6 = pick(lambda c: c["dim"] == 1 and c["dtype"] == "fp16" and c["op"] == "argmin" and c["tb"] is False)

    def rec(c, pk=None, crash=None, sec=4.0):
        pk = pk or {}
        return dict(t="cfg", stage="A", key=cfg_key(c), cfg=c, tier=c["tier"], compile_s=1.0, crash=crash, sec=sec,
                    kinds={}, viol_pats=sorted(pk), pat_kinds={p: {k: 1} for p, k in pk.items()},
                    viol_names=[[p, None] for p in pk], npat=10, npat_total=10,
                    launches=10, ok=10 - len(pk), viol=len(pk), unspec=0)
    A = [rec(c1, {"nan1_last": "nan_index"}), rec(c2, {"sum_nan": "nan_lost"}), rec(c3), rec(c4, {"ties_min": "tie_violation"}),
         rec(c5, crash="CompilationError: boom"), rec(c6, {"nan1_pos": "wrong_index"})]
    order = select_stage_b(plan, A, frac=1.0)
    keys = [cfg_key(c) for c in order]
    assert keys.index(cfg_key(c5)) < keys.index(cfg_key(c1)), "crash must precede #11993-like"
    assert keys.index(cfg_key(c2)) < keys.index(cfg_key(c1)), "non-#11993 violation must precede #11993-like"
    assert keys.index(cfg_key(c4)) < keys.index(cfg_key(c6)), "tie violation must precede #11993-like"
    assert keys.index(cfg_key(c3)) < keys.index(cfg_key(c1)), "clean sample precedes #11993-like"
    assert keys.index(cfg_key(c1)) < keys.index(cfg_key(c6)), "tier-1 (2D) precedes tier-3 (1D) among #11993-like"
    by = {cfg_key(c): c for c in order}
    assert by[cfg_key(c1)]["only_pats"] == [["nan1_last", None]]
    assert by[cfg_key(c5)].get("only_pats") is None
    assert stage_b_eta(order, A) > 0
    B = [rec(c1), rec(c2, {"sum_nan": "nan_lost"}), rec(c3, {"distinct": "wrong_value"}), rec(c5, crash="CompilationError: boom")]
    for r in B:
        r["stage"] = "B"
    vA = [dict(t="viol", key=cfg_key(c2), cfg=c2, pat="sum_nan", pos=None, kind="nan_lost", lane=0, col=[1.0, float("nan")], got=0.0),
          dict(t="viol", key=cfg_key(c6), cfg=c6, pat="nan1_pos", pos=3, kind="wrong_index", lane=0, col=[float(i) for i in range(300)], got=1)]
    vB = [dict(t="viol", key=cfg_key(c3), cfg=c3, pat="distinct", pos=None, kind="wrong_value", lane=0, col=[1.0, 2.0], got=0.0)]
    rows, repd = cluster_report(A, B, vA, vB)
    tab = {ck: dict(v) for ck, v in rows}
    k_cleared = (c1["op"], c1["dtype"], "nan1_last", "nan_index", "2D-ax%d" % c1["axis"])
    assert tab[k_cleared].get("cleared_by_patch") == 1, tab[k_cleared]
    k_pers = (c2["op"], c2["dtype"], "sum_nan", "nan_lost", "2D-ax%d" % c2["axis"])
    assert tab[k_pers].get("persistent") == 1, tab[k_pers]
    k_intro = (c3["op"], c3["dtype"], "distinct", "wrong_value", "1D")
    assert tab[k_intro].get("introduced_by_patch") == 1, tab[k_intro]
    k_notrerun = (c6["op"], c6["dtype"], "nan1_pos", "wrong_index", "1D")
    assert tab[k_notrerun].get("not_rerun") == 1, tab[k_notrerun]
    assert ("*", "crash") in {(ck[2], ck[3]) for ck in tab}, "crash triple missing"
    assert k_pers in repd and k_intro in repd and k_cleared not in repd and k_notrerun not in repd
    print_clusters(rows)
    print(coverage_report(plan, A))
    print(distinct_report(A + B))
    dump = dump_compact("A", A, vA)
    body = "".join(dump.splitlines()[1:-1])
    data = json.loads(gzip.decompress(base64.b64decode(body)))
    assert data["stage"] == "A" and len(data["cfgs"]) == 5 and len(data["viols"]) == 2
    assert any("colsum" in v for v in data["viols"]), "long column not trimmed"
    g = globals()
    orig = g["_spawn"]

    def fake(cfg_path, out_path, stage, seed, timeout, env_extra=None):
        cfgs = json.load(open(cfg_path))
        write = cfgs[:-1] if len(cfgs) > 1 else []
        with open(out_path, "a") as fh:
            for c in write:
                fh.write(json.dumps(rec(c)) + "\n")
        return 3, "fake death"
    g["_spawn"] = fake
    try:
        with tempfile.TemporaryDirectory() as td:
            res = run_stage(plan[:5], "T", 600, chunk_size=3, out_dir=td)
            crecs, _v = load_stage("T", td)
            assert res["stats"]["cfgs"] == 5 and res["stats"]["crash"] == 2, res
            assert len(crecs) == 5
    finally:
        g["_spawn"] = orig
    orig_run = subprocess.run
    subprocess.run = lambda *a, **k: SimpleNamespace(stdout='REPRO {"viol": 1, "kinds": {}, "crash": null}\n', stderr="", returncode=0)
    try:
        with tempfile.TemporaryDirectory() as td:
            rc = repro_check(repd, "R", n=3, out_dir=td)
            assert all(det for det, _o in rc.values()), rc
    finally:
        subprocess.run = orig_run
    print("driver self-test ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--cfgs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--stage", default="A")
    r.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("repro")
    q.add_argument("--cfg", required=True)
    q.add_argument("--pattern", required=True)
    q.add_argument("--pos", default="null")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--stage", default="R")
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        cli_run(a)
    elif a.cmd == "repro":
        cli_repro(a)


if __name__ == "__main__":
    main()

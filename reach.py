"""Reduction reach: how a tt.reduce axis is split across register / lane / warp, from TTGIR text.

Spec: vault triton_00/notes/27-차등스윕v2-명세.md §2 (pre-registered).
Pure Python, no triton import, so the self-test runs before any wheel is installed.

Model, read from triton @ 93ba8872b6d783665af1cf662ebd48bda49e60b9:
  - lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp: reduce register bases first
    (reduceWithinThreads), then lane bases (reduceWithinWarps, shuffles), then loop over
    warp bases through shared memory (at most 2 rounds).
  - lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:881 BlockedEncodingAttr::toLinearLayout
    = identityStandardND(register) * identityStandardND(lane) * identityStandardND(warp),
    then combineCtaCgaWithShape(shape), then removeZeroBasesAlongDim(register).
  Per dimension the register bits are the lowest, then lane bits, then warp bits. Bits at or above
  log2(shape) are broadcast (zero bases, no work). If the shape is larger than the layout covers,
  the extra bits are appended as register bits. So for an axis of size 2**nbits:
      reg_bits + lane_bits + warp_bits == nbits      (always; used as a self-test invariant)
  Only blocked layouts are modelled. Anything else is reported as 'unclassified', never guessed.

All of this is a model of the lowering (hypothesis); ptx_check() cross-checks it against the PTX.
"""
import re


def _log2_exact(n):
    n = int(n)
    if n < 1 or (n & (n - 1)):
        raise ValueError("not a power of 2: %d" % n)
    return n.bit_length() - 1


def depth_blocked(axis_size, spt, tpw, wpc):
    """Effective split of one reduction axis for a blocked layout (per-axis sizePerThread etc.)."""
    nbits = _log2_exact(axis_size)
    r0, l0, w0 = _log2_exact(spt), _log2_exact(tpw), _log2_exact(wpc)
    reg = min(r0, nbits)
    lane = min(max(nbits - r0, 0), l0)
    warp = min(max(nbits - r0 - l0, 0), w0)
    rep = max(nbits - r0 - l0 - w0, 0)
    d = dict(nbits=nbits, reg_bits=reg + rep, lane_bits=lane, warp_bits=warp,
             bcast_lane_bits=l0 - lane, bcast_warp_bits=w0 - warp)
    assert d["reg_bits"] + d["lane_bits"] + d["warp_bits"] == nbits, d
    return d


def stage_signature(d):
    """'R' in-thread, 'L' lane shuffle, 'W' cross-warp shared memory; '-' if the axis has size 1."""
    s = ("R" if d["reg_bits"] else "") + ("L" if d["lane_bits"] else "") + ("W" if d["warp_bits"] else "")
    return s or "-"


# ----------------------------------------------------------------------------
# TTGIR text parsing
# ----------------------------------------------------------------------------
_ALIAS = re.compile(r"^(#[\w$]+)\s*=\s*(.+?)\s*$", re.M)
_BLOCKED = re.compile(r"#(?:ttg|triton_gpu)\.blocked<\{(.*)\}>", re.S)


def _match_angle(s, i):
    """s[i] == '<'. Return the index just after the matching '>'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "<":
            depth += 1
        elif s[j] == ">":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced <>")


def _split_top(s, sep=","):
    out, depth, cur = [], 0, []
    for c in s:
        if c in "<{[(":
            depth += 1
        elif c in ">}])":
            depth -= 1
        if c == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _int_list(body, key):
    m = re.search(r"\b" + key + r"\s*=\s*\[([^\]]*)\]", body)
    if not m:
        raise ValueError("missing " + key)
    return [int(x) for x in m.group(1).split(",") if x.strip()]


def _parse_tensor_type(t):
    """'tensor<8x8xf16, #blocked>' -> (shape list, dtype str, encoding str or None)."""
    t = t.strip()
    if not t.startswith("tensor<"):
        raise ValueError("non-tensor operand: " + t[:30])
    end = _match_angle(t, len("tensor"))
    inner = t[len("tensor<"):end - 1]
    parts = _split_top(inner)
    m = re.match(r"^((?:\d+x)+)(\w+)$", parts[0])
    if not m:
        raise ValueError("dynamic or odd shape: " + parts[0])
    shape = [int(x) for x in m.group(1).rstrip("x").split("x")]
    return shape, m.group(2), (parts[1] if len(parts) > 1 else None)


def _resolve_blocked(enc, aliases):
    if enc is None:
        raise ValueError("no encoding")
    full = enc
    if enc.startswith("#") and "<" not in enc:
        if enc not in aliases:
            raise ValueError("alias not found: " + enc)
        full = aliases[enc]
    m = _BLOCKED.search(full)
    if not m:
        raise ValueError("not a blocked layout: " + full[:60])
    body = m.group(1)
    if re.search(r"CGALayout\s*=\s*\[\s*\[", body) or any(v > 1 for v in _try_list(body, "CTAsPerCGA")):
        raise ValueError("multi-CTA layout")
    return dict(spt=_int_list(body, "sizePerThread"), tpw=_int_list(body, "threadsPerWarp"),
                wpc=_int_list(body, "warpsPerCTA"), order=_int_list(body, "order"))


def _try_list(body, key):
    try:
        return _int_list(body, key)
    except ValueError:
        return []


def find_reduces(ttgir):
    """Yield dicts for every tt.reduce: axis, nargs, list of operand type strings."""
    out = []
    for m in re.finditer(r'"tt\.reduce"\(([^)]*)\)\s*<\{([^}]*)\}>', ttgir):
        am = re.search(r"axis\s*=\s*(\d+)", m.group(2))
        nargs = len([x for x in m.group(1).split(",") if x.strip()])
        p = ttgir.find("({", m.end())
        if am is None or p < 0:
            out.append(dict(error="malformed tt.reduce", nargs=nargs))
            continue
        depth, j = 0, p + 1
        while j < len(ttgir):
            if ttgir[j] == "{":
                depth += 1
            elif ttgir[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        sm = re.compile(r"\)\s*:\s*\(").match(ttgir, j + 1)
        if not sm:
            out.append(dict(error="no type signature after region", nargs=nargs))
            continue
        k, depth, start, types = sm.end(), 1, sm.end(), []
        while k < len(ttgir) and depth:
            c = ttgir[k]
            if c in "<{[(":
                depth += 1
            elif c in ">}])":
                depth -= 1
            k += 1
        types = _split_top(ttgir[start:k - 1])
        out.append(dict(axis=int(am.group(1)), nargs=nargs, types=types))
    return out


def analyze_ttgir(ttgir):
    """Classify the reduce axis of a compiled kernel. Prefers the multi-operand (argmax/argmin) reduce.

    Returns dict(status='ok'|'unclassified', reason, ...). Never raises.
    """
    try:
        aliases = {m.group(1): m.group(2) for m in _ALIAS.finditer(ttgir)}
        reds = find_reduces(ttgir)
        if not reds:
            return dict(status="unclassified", reason="no tt.reduce")
        if any("error" in r for r in reds):
            return dict(status="unclassified", reason=[r["error"] for r in reds if "error" in r][0])
        reds.sort(key=lambda r: -r["nargs"])
        r = reds[0]
        shape, dtype, enc = _parse_tensor_type(r["types"][0])
        lay = _resolve_blocked(enc, aliases)
        ax = r["axis"]
        if not (len(shape) == len(lay["spt"]) == len(lay["tpw"]) == len(lay["wpc"])) or ax >= len(shape):
            raise ValueError("rank mismatch")
        d = depth_blocked(shape[ax], lay["spt"][ax], lay["tpw"][ax], lay["wpc"][ax])
        d.update(status="ok", reason="", axis=ax, shape=shape, nargs=r["nargs"], n_reduces=len(reds),
                 spt=lay["spt"], tpw=lay["tpw"], wpc=lay["wpc"], order=lay["order"], sig=stage_signature(d))
        return d
    except Exception as e:  # unclassified is a legitimate outcome, not a crash
        return dict(status="unclassified", reason="%s: %s" % (type(e).__name__, e))


def ptx_features(ptx):
    return dict(shfl=("shfl.sync" in ptx) or ("redux.sync" in ptx),
                xwarp=("bar.sync" in ptx) or ("st.shared" in ptx) or ("ld.shared" in ptx))


def ptx_check(d, feats, confounded=False):
    """Presence-only cross-check of the model against the PTX. Returns list of disagreements.

    Model predicts a stage that the PTX lacks: always a contradiction.
    PTX has a feature the model does not predict: a contradiction only if the kernel has no
    ttg.convert_layout (a layout conversion can itself emit shuffles / shared-memory traffic, so with
    one present the PTX presence does not isolate the reduction). `confounded` = TTGIR has convert_layout.
    (Added after the first stage-0 run, where 576 2D configs were flagged only because of the store-side
    conversion; see vault 27 §2 'result'.)
    """
    bad = []
    for name, model, ptx in (("shfl", d["lane_bits"] > 0, feats["shfl"]), ("xwarp", d["warp_bits"] > 0, feats["xwarp"])):
        if model and not ptx:
            bad.append("%s: model=True ptx=False" % name)
        elif ptx and not model and not confounded:
            bad.append("%s: model=False ptx=True" % name)
    return bad


# ----------------------------------------------------------------------------
# self-test (pure logic; must print SELFTEST OK before any compile is trusted)
# ----------------------------------------------------------------------------
_SYN_ARGMAX = '''#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#loc = loc("k.py":1:0)
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @k(%arg0: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : tensor<8x!tt.ptr<f32>, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %c = arith.maxnumf %a, %b : f32
      tt.reduce.return %c : f32
    }) : (tensor<8xf32, #blocked>) -> f32
    %2:2 = "tt.reduce"(%0, %0) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %i: i32, %b: f32, %j: i32):
      %c = arith.cmpf ogt, %a, %b : f32
      %d = arith.select %c, %i, %j : i32
      tt.reduce.return %a, %d : f32, i32
    }) : (tensor<8xf32, #blocked>, tensor<8xi32, #blocked>) -> (f32, i32)
    tt.return
  }
}
'''


def selftest():
    # --- depth formula, hand-computed (reg, lane, warp, bcast_lane, bcast_warp) ---
    cases = [
        # (axis_size, spt, tpw, wpc) -> expected
        ((8, 1, 32, 1), (0, 3, 0, 2, 0)),      # N=8 on 32 lanes: 3 shuffle levels, 2 broadcast lane bits
        ((8, 1, 32, 4), (0, 3, 0, 2, 2)),      # replicated across all 4 warps: NO cross-warp work (advisor case)
        ((4, 1, 32, 4), (0, 2, 0, 3, 2)),      # upstream test shape (BLOCK=4, nw=4)
        ((32, 1, 32, 4), (0, 5, 0, 0, 2)),     # exactly one warp of data
        ((128, 1, 32, 4), (0, 5, 2, 0, 0)),    # lanes + warps
        ((1024, 1, 32, 4), (3, 5, 2, 0, 0)),   # layout smaller than shape: 3 extra register bits
        ((64, 4, 8, 2), (2, 3, 1, 0, 0)),      # 2 reg + 3 lane + 1 warp
        ((2, 4, 8, 2), (1, 0, 0, 3, 1)),       # spt larger than the axis: clipped, all else broadcast
        ((256, 1, 32, 1), (3, 5, 0, 0, 0)),
    ]
    for args, exp in cases:
        d = depth_blocked(*args)
        got = (d["reg_bits"], d["lane_bits"], d["warp_bits"], d["bcast_lane_bits"], d["bcast_warp_bits"])
        assert got == exp, (args, got, exp)
    # invariant over a grid of power-of-two layouts (reg + lane + warp == log2 axis size)
    n = 0
    for nb in range(0, 12):
        for r0 in range(0, 4):
            for l0 in range(0, 6):
                for w0 in range(0, 4):
                    d = depth_blocked(1 << nb, 1 << r0, 1 << l0, 1 << w0)
                    assert d["reg_bits"] + d["lane_bits"] + d["warp_bits"] == nb
                    n += 1
    assert stage_signature(depth_blocked(8, 1, 32, 1)) == "L"
    assert stage_signature(depth_blocked(1024, 1, 32, 4)) == "RLW"
    assert stage_signature(depth_blocked(1, 1, 32, 1)) == "-"
    # --- TTGIR parsing ---
    a = analyze_ttgir(_SYN_ARGMAX)
    assert a["status"] == "ok" and a["nargs"] == 2 and a["n_reduces"] == 2 and a["sig"] == "L", a
    assert (a["reg_bits"], a["lane_bits"], a["warp_bits"]) == (0, 3, 0), a
    inline = _SYN_ARGMAX.replace("#blocked>", "#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>")
    assert analyze_ttgir(inline)["status"] == "ok", analyze_ttgir(inline)
    # --- negatives: must be 'unclassified', never silently classified ---
    sl = _SYN_ARGMAX.replace("#blocked>", "#ttg.slice<{dim = 1, parent = #blocked}>>")
    assert analyze_ttgir(sl)["status"] == "unclassified", analyze_ttgir(sl)
    assert analyze_ttgir("module {}")["status"] == "unclassified"
    npow2 = _SYN_ARGMAX.replace("tensor<8x", "tensor<6x")
    assert analyze_ttgir(npow2)["status"] == "unclassified"
    cga = _SYN_ARGMAX.replace("order = [0]}>", "order = [0], CGALayout = [[1]]}>", 1)
    assert analyze_ttgir(cga)["status"] == "unclassified", analyze_ttgir(cga)
    # --- PTX cross-check ---
    ok = analyze_ttgir(_SYN_ARGMAX)
    assert ptx_check(ok, ptx_features("shfl.sync.bfly.b32 ...")) == []
    assert len(ptx_check(ok, ptx_features("bar.sync 0; st.shared.b32"))) == 2   # no shfl, unexpected xwarp
    assert len(ptx_check(ok, ptx_features("mov.u32 %r1, 0;"))) == 1             # missing shfl
    # confounded by a layout conversion: unexpected PTX features are tolerated, missing predicted ones are not
    assert ptx_check(ok, ptx_features("shfl.sync.bfly.b32; bar.sync 0; st.shared.b32"), confounded=True) == []
    assert len(ptx_check(ok, ptx_features("shfl.sync.bfly.b32; bar.sync 0; st.shared.b32"), confounded=False)) == 1
    assert len(ptx_check(ok, ptx_features("bar.sync 0; st.shared.b32"), confounded=True)) == 1   # shfl still missing
    print("SELFTEST OK (depth cases %d, invariant grid %d)" % (len(cases), n))


if __name__ == "__main__":
    selftest()

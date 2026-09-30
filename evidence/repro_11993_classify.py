"""Follow-up for #11993: classify wrong argmax results, print combine source, try nightly."""
import os, subprocess, textwrap, json

WORKER = textwrap.dedent('''
    import os, json, inspect
    import torch, triton, triton.language as tl
    import triton.language.standard as st
    interp = os.environ.get("TRITON_INTERPRET") == "1"
    dev = "cpu" if interp else "cuda"
    fn = st._argmax_combine
    src = inspect.getsource(getattr(fn, "fn", fn))
    print("TRITON", triton.__version__, "interp", interp)
    print("COMBINE_SRC\\n" + src)

    @triton.jit
    def k(x_ptr, i_ptr, v_ptr, N: tl.constexpr, TB: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, N))
        tl.store(i_ptr + 0, tl.argmax(x, 0, tie_break_left=TB))
        tl.store(i_ptr + 1, tl.argmin(x, 0, tie_break_left=TB))
        tl.store(v_ptr + 0, tl.max(x, 0))
        tl.store(v_ptr + 1, tl.min(x, 0))

    g = torch.Generator().manual_seed(1)
    summary = {}
    examples = []
    for N in (8, 32, 128):
        for nw in (1, 4):
            for TB in (True, False):
                cnt = dict(ok=0, nan_idx=0, wrong_finite=0, total=0)
                for trial in range(4):
                    base = torch.randperm(N, generator=g).float() - N / 2
                    for p in range(N):
                        x = base.clone(); x[p] = float("nan")
                        xf = base.clone(); xf[p] = -1e30
                        true_i = int(xf.argmax()); true_v = float(xf[true_i])
                        i = torch.zeros(2, dtype=torch.int32, device=dev)
                        v = torch.zeros(2, dtype=torch.float32, device=dev)
                        k[(1,)](x.to(dev), i, v, N=N, TB=TB, num_warps=nw)
                        gi = int(i[0]); gv = float(v[0])
                        cnt["total"] += 1
                        if gi == true_i: cnt["ok"] += 1
                        elif gi == p: cnt["nan_idx"] += 1
                        else:
                            cnt["wrong_finite"] += 1
                            if len(examples) < 6:
                                examples.append(dict(N=N, nw=nw, TB=TB, nan_pos=p, got=gi,
                                    x_got=float(x[gi]), true_idx=true_i, true_val=true_v, tl_max_val=gv))
                summary[f"N{N}_nw{nw}_TB{TB}"] = cnt
    print("SUMMARY", json.dumps(summary))
    print("EXAMPLES", json.dumps(examples))
''')
os.makedirs("/tmp/w", exist_ok=True)
open("/tmp/w/worker2.py", "w").write(WORKER)

def sh(cmd, env=None, timeout=900):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, shell=True, cwd="/tmp", env=e, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr

def run_all(tag):
    for mode, env in (("interp_cpu", {"TRITON_INTERPRET": "1"}), ("compiled_gpu", {})):
        rc, o, e = sh("python /tmp/w/worker2.py", env=env, timeout=900)
        print(f"##### {tag} {mode} rc={rc}")
        if rc != 0:
            print("STDERR tail:", e[-1200:]); continue
        for line in o.splitlines():
            if line.startswith("SUMMARY") and mode == "compiled_gpu":
                s = json.loads(line[8:])
                agg = dict(ok=0, nan_idx=0, wrong_finite=0, total=0)
                for c in s.values():
                    for kk in agg: agg[kk] += c[kk]
                print("AGG", agg)
                for name, c in s.items(): print(" ", name, c)
            elif line.startswith("SUMMARY"):
                s = json.loads(line[8:]); agg = dict(ok=0, nan_idx=0, wrong_finite=0, total=0)
                for c in s.values():
                    for kk in agg: agg[kk] += c[kk]
                print("AGG", agg)
            else:
                print(line[:400])

run_all("installed")
rc, o, e = sh("pip install -q -U --pre triton-nightly --index-url "
              "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ 2>&1 | tail -5", timeout=600)
print("NIGHTLY_INSTALL rc", rc, (o + e)[-600:])
rc, o, e = sh("python -c \"import triton;print(triton.__version__, triton.__file__)\"")
print("AFTER_INSTALL", o.strip(), e.strip()[-200:])
run_all("after_nightly_attempt")

"""Repro for triton-lang/triton#11993: argmin/argmax NaN-lane order dependence.
Runs inside a Colab python kernel. Worker runs in subprocess (cwd=/tmp) so that
TRITON_INTERPRET can be toggled per mode and no import shadowing occurs.
"""
import json, os, subprocess, sys, textwrap

WORKER = textwrap.dedent('''
    import os, sys, json
    import torch, triton, triton.language as tl

    interp = os.environ.get("TRITON_INTERPRET") == "1"
    dev = "cpu" if interp else "cuda"

    @triton.jit
    def k(x_ptr, i_ptr, v_ptr, N: tl.constexpr, TB: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, N))
        tl.store(i_ptr + 0, tl.argmax(x, 0, tie_break_left=TB))
        tl.store(i_ptr + 1, tl.argmin(x, 0, tie_break_left=TB))
        tl.store(v_ptr + 0, tl.max(x, 0))
        tl.store(v_ptr + 1, tl.min(x, 0))

    g = torch.Generator().manual_seed(0)
    out = {"triton": triton.__version__, "interp": interp, "rows": []}
    for N in (4, 8, 32, 128, 1024):
        for nw in (1, 4):
            for TB in (True, False):
                base = torch.randperm(N, generator=g).float() - N / 2  # distinct finite
                positions = list(range(N)) if N <= 32 else sorted(set(
                    [0, 1, 2, N // 2, N - 2, N - 1] + torch.randint(0, N, (26,), generator=g).tolist()))
                bad_max, bad_min, val_nan = [], [], 0
                for p in positions:
                    x = base.clone(); x[p] = float("nan")
                    fin = base.clone(); fin[p] = float("nan")
                    exp_max = int(torch.nanmean(fin) * 0 + 0)  # placeholder, overwritten below
                    xf = base.clone(); xf[p] = -1e30
                    exp_max = int(xf.argmax())
                    xf = base.clone(); xf[p] = 1e30
                    exp_min = int(xf.argmin())
                    xd = x.to(dev)
                    i = torch.zeros(2, dtype=torch.int32, device=dev)
                    v = torch.zeros(2, dtype=torch.float32, device=dev)
                    k[(1,)](xd, i, v, N=N, TB=TB, num_warps=nw)
                    if not interp:
                        torch.cuda.synchronize()
                    i = i.cpu().tolist(); v = v.cpu().tolist()
                    if i[0] != exp_max: bad_max.append((p, i[0]))
                    if i[1] != exp_min: bad_min.append((p, i[1]))
                    if v[0] != v[0] or v[1] != v[1]: val_nan += 1
                out["rows"].append(dict(N=N, nw=nw, TB=TB, n=len(positions),
                    bad_argmax=len(bad_max), bad_argmin=len(bad_min), value_nan=val_nan,
                    ex_argmax=bad_max[:6], ex_argmin=bad_min[:6]))
    print("JSON:" + json.dumps(out))
''')

os.makedirs("/tmp/w", exist_ok=True)
open("/tmp/w/worker.py", "w").write(WORKER)

def sh(cmd, env=None, timeout=900):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, shell=True, cwd="/tmp", env=e, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr

rc, o, e = sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader; python -c \"import triton,torch;print('triton',triton.__version__,'torch',torch.__version__,triton.__file__)\"")
print("ENV rc", rc, o.strip(), e.strip()[-300:])
rc, o, e = sh("python - <<'EOF'\nimport inspect, triton.language.standard as s\nprint(inspect.getsource(s._argmax_combine))\nEOF")
print("COMBINE rc", rc, "\n" + o, e.strip()[-300:])

results = {}
for mode, env in (("interp_cpu", {"TRITON_INTERPRET": "1"}), ("compiled_gpu", {})):
    rc, o, e = sh("python /tmp/w/worker.py", env=env, timeout=1500)
    line = [l for l in o.splitlines() if l.startswith("JSON:")]
    if rc != 0 or not line:
        print(mode, "FAILED rc", rc, "\nSTDOUT tail:", o[-800:], "\nSTDERR tail:", e[-1500:])
        continue
    results[mode] = json.loads(line[0][5:])

for mode, r in results.items():
    print(f"=== {mode} triton={r['triton']}")
    print("N    nw TB    n   bad_argmax bad_argmin value_nan  examples(argmax: pos->got)")
    for x in r["rows"]:
        print(f"{x['N']:<5}{x['nw']:<3}{str(x['TB']):<6}{x['n']:<4}{x['bad_argmax']:<11}{x['bad_argmin']:<11}{x['value_nan']:<10} {x['ex_argmax']}")
json.dump(results, open("/tmp/w/results.json", "w"))
print("saved /tmp/w/results.json")

import subprocess
code = r'''
import sys, numpy as np, torch
sys.path.insert(0, "/content")
import sweep_kernels as K
torch.manual_seed(0)
N = 1024
def run_argmax(x):
    xt = torch.tensor(x, dtype=torch.float32, device="cuda")
    val = torch.zeros(1, dtype=torch.float32, device="cuda"); idx = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    K.kern1d[(1,)](xt, val, idx, N=N, OP=3, TB=True, num_warps=4)
    torch.cuda.synchronize(); return int(idx.item())
base = np.random.RandomState(1).permutation(N).astype(np.float32)  # distinct finite
exp_fin = int(np.argmax(base))
# negative control: no NaN
print("NEG no-NaN:", "ok" if run_argmax(base) == exp_fin else "MISMATCH")
# positive control: single NaN at each position; contract = NaN ignored
bad = 0; nanidx = 0; total = 0
for p in list(range(0, N, 37)) + [exp_fin, N-1, 0]:
    x = base.copy(); x[p] = np.nan
    e = int(np.nanargmax(x)); g = run_argmax(x); total += 1
    if g != e:
        bad += 1; nanidx += (g == p)
print(f"POS NaN-case: total={total} wrong={bad} returned_nan_index={nanidx}")
'''
open("/content/ctl_run.py", "w").write(code)
r = subprocess.run("cd /tmp && python3 /content/ctl_run.py 2>&1 | tail -6", shell=True, capture_output=True, text=True); print(r.stdout + r.stderr)
code2 = r'''
import sys, torch
sys.path.insert(0, "/content")
import sweep_kernels as K
for dt in (torch.bfloat16, torch.float16, torch.float32):
    for nw in (1, 4):
        x = torch.randn(128, 32, device="cuda").to(dt)
        val = torch.zeros(32, dtype=dt, device="cuda"); idx = torch.zeros(32, dtype=torch.int32, device="cuda")
        try:
            K.kern2d[(1,)](x, val, idx, M=128, N=32, AXIS=0, OP=2, TB=True, num_warps=nw)
            torch.cuda.synchronize(); print(dt, "nw", nw, "OK")
        except Exception as e:
            print(dt, "nw", nw, "EXC", type(e).__name__, str(e).replace(chr(10), " | ")[:110])
'''
open("/content/crash_run.py", "w").write(code2)
r = subprocess.run("cd /tmp && python3 /content/crash_run.py 2>&1 | grep -v nvdisasm | tail -8", shell=True, capture_output=True, text=True); print(r.stdout + r.stderr)

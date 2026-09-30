import subprocess, textwrap, os
def sh(c, t=600):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    return (r.stdout + r.stderr)
print(sh("pip install -q --no-deps /content/wheelhouse/triton-3.9.0+git93ba8872-cp313-cp313-linux_x86_64.whl 2>&1 | tail -3; python3 -c \"import triton; print('triton', triton.__version__, triton.__file__)\""))
os.makedirs('/content/w', exist_ok=True)
open('/content/w/kern.py','w').write(open('/content/w/kern.py').read() if os.path.exists('/content/w/kern.py') else '')
open('/content/w/kern.py','w').write('''
import triton, triton.language as tl
@triton.jit
def kern2d(x_ptr, val_ptr, idx_ptr, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, OP: tl.constexpr, TB: tl.constexpr):
    om = tl.arange(0, M)
    on = tl.arange(0, N)
    x = tl.load(x_ptr + om[:, None] * N + on[None, :])
    if AXIS == 0:
        oo = on
    else:
        oo = om
    tl.store(val_ptr + oo, tl.sum(x, AXIS))
''')
open('/content/w/comp.py','w').write('''
import sys, triton, traceback
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import kern
dt, M, N, nw, axis = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sig = {"x_ptr": "*"+dt, "val_ptr": "*"+dt, "idx_ptr": "*i32", "M": "constexpr", "N": "constexpr", "AXIS": "constexpr", "OP": "constexpr", "TB": "constexpr"}
src = ASTSource(fn=kern.kern2d, signature=sig, constexprs={"M": M, "N": N, "AXIS": axis, "OP": 2, "TB": False})
try:
    k = triton.compile(src, target=GPUTarget("cuda", 89, 32), options={"num_warps": nw})
    print("COMPILED_OK", dt, M, N, nw, "ptx_len", len(k.asm["ptx"]))
except Exception as e:
    print("COMPILE_EXC", type(e).__name__, str(e).replace(chr(10), " | ")[:400])
''')
for cfg in ["bf16 128 32 1 0", "fp16 128 32 1 0", "fp32 128 32 1 0", "bf16 128 32 4 0", "bf16 128 32 1 1"]:
    print("== ", cfg)
    for i in range(3 if cfg == "bf16 128 32 1 0" else 1):
        out = sh(f"cd /content/w && rm -rf ~/.triton/cache && TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/content/w/dump_{cfg.replace(' ','_')}_{i} python3 comp.py {cfg} 2>&1 | tail -4")
        print(f"[run{i}]", out.strip()[:500])

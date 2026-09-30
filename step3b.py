import subprocess, os
def sh(c, t=600):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return r.stdout + r.stderr
src = open('/content/w/comp.py').read()
src = src.replace('src = ASTSource(fn=kern.kern2d, signature=sig, constexprs={"M": M, "N": N, "AXIS": axis, "OP": 2, "TB": False})',
 'div = {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]], (2,): [["tt.divisibility", 16]]}\nsrc = ASTSource(fn=kern.kern2d, signature=sig, constexprs={"M": M, "N": N, "AXIS": axis, "OP": 2, "TB": False}, attrs=div)')
assert 'div =' in src
open('/content/w/comp_div.py','w').write(src)
for cfg in ["bf16 128 32 1 0", "fp16 128 32 1 0", "bf16 64 64 1 0", "fp16 64 64 1 0", "fp32 128 32 1 0", "bf16 128 32 4 0"]:
    for i in range(2 if cfg.startswith(("bf16 128 32 1","fp16 128 32 1")) else 1):
        out = sh(f"cd /content/w && rm -rf ~/.triton/cache && python3 comp_div.py {cfg} 2>&1 | grep -v nvdisasm | tail -3")
        print(cfg, f"[run{i}]", out.strip()[:600])

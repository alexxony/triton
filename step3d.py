import subprocess, glob, os, zipfile
def sh(c, t=600):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return (r.stdout + r.stderr).strip()
whl = glob.glob("/content/w/nv/*.whl")[0]
z = zipfile.ZipFile(whl)
names = [n for n in z.namelist() if n.endswith("/ptxas")]
print("ptxas in wheel:", names)
z.extractall("/content/w/nv/x")
P13 = glob.glob("/content/w/nv/x/**/bin/ptxas", recursive=True)[0]
os.chmod(P13, 0o755)
print("v13:", sh(f"{P13} --version | tail -2"))
p = "/content/w/repro/kern2d_bf16_128x32_ax0_nw1.ptx"
for i in range(3):
    r = subprocess.run(f"{P13} -lineinfo --fmad=false -v --regAllocOptLevel=2 --gpu-name=sm_89 {p} -o /content/w/repro/o13.cubin", shell=True, capture_output=True, text=True)
    print(f"ptxas13 run{i}: rc={r.returncode}", (r.stderr or '')[-160:].replace("\n", " | "))
# pack repro artifacts
print(sh("cd /content/w && tar czf repro.tgz repro/kern2d_bf16_128x32_ax0_nw1.ptx kern.py comp_div.py && ls -la repro.tgz && sha256sum repro/kern2d_bf16_128x32_ax0_nw1.ptx"))

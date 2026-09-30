import subprocess, glob, os, shutil
def sh(c, t=600):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return (r.stdout + r.stderr).strip()
PT = "/usr/local/lib/python3.13/dist-packages/triton/backends/nvidia/bin/ptxas"
print("ptxas version:", sh(f"{PT} --version | tail -2"))
sh("rm -rf /content/w/dumpX ~/.triton/cache")
print(sh("cd /content/w && TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/content/w/dumpX python3 comp_div.py bf16 128 32 1 0 2>&1 | grep -c PTXASError"))
ptx = glob.glob("/content/w/dumpX/**/*.ptx", recursive=True)
print("ptx files:", ptx)
os.makedirs("/content/w/repro", exist_ok=True)
if ptx:
    shutil.copy(ptx[0], "/content/w/repro/kern2d_bf16_128x32_ax0_nw1.ptx")
    p = "/content/w/repro/kern2d_bf16_128x32_ax0_nw1.ptx"
    print("ptx size", os.path.getsize(p), " head:", sh(f"head -5 {p}"))
    for i in range(3):
        r = subprocess.run(f"{PT} -lineinfo --fmad=false -v --regAllocOptLevel=2 --gpu-name=sm_89 {p} -o /content/w/repro/o.cubin", shell=True, capture_output=True, text=True)
        print(f"standalone run{i}: rc={r.returncode}", (r.stderr or '')[-150:])
    # variants: without -lineinfo / regAllocOptLevel / opt level
    for name, extra in [("no-lineinfo", f"--fmad=false --gpu-name=sm_89"), ("O0", f"-O0 --fmad=false --gpu-name=sm_89"), ("fmad-true", f"--gpu-name=sm_89"), ("sm_90", f"--fmad=false --gpu-name=sm_90")]:
        r = subprocess.run(f"{PT} {extra} {p} -o /content/w/repro/o2.cubin", shell=True, capture_output=True, text=True)
        print(f"variant {name}: rc={r.returncode}")
# newer ptxas from pip
print(sh("pip download nvidia-cuda-nvcc --no-deps -d /content/w/nv 2>&1 | tail -2"))
print(sh("ls -la /content/w/nv 2>&1 | head"))

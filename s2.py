import subprocess, os
def sh(c, t=900):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return (r.stdout + r.stderr).strip()
W = "/content/triton-3.9.0+git93ba8872-cp313-cp313-linux_x86_64.whl"
print(sh(f"cat /content/part_0* > {W} && sha256sum {W} && ls -l {W}"))
print("EXPECT 60b7f57504d0e28f1aa84e6e3f7331454ecfa2d2d4f0c17ea9b11ae5f494619b")
print(sh(f"pip uninstall -y triton 2>&1 | tail -1; pip install -q --no-deps {W} 2>&1 | tail -3"))
print(sh("cd /tmp && python3 -c \"import triton, torch; print('triton', triton.__version__, triton.__file__); print('gpu', torch.cuda.get_device_name(0))\""))

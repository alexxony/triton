# colab-split-v1 — CPU build / GPU run split (2026-09-30)

Target: triton-lang/triton @ 93ba8872b6d783665af1cf662ebd48bda49e60b9

- build_launch.py : S1, CPU high-RAM (8 vCPU/50GB) build -> wheel. 2261 s. wheel 460,795,269 B
  sha256 60b7f57504d0e28f1aa84e6e3f7331454ecfa2d2d4f0c17ea9b11ae5f494619b
- step3*.py       : GPU-free compile of the ptxas -11 crash config (needs tt.divisibility=16 attrs on the 3 pointers)
- s2.py / ctl.py  : S2, L4: chunked-upload reassembly + sha check + install + positive/negative control + crash launch
- repro/*.ptx     : PTX that segfaults bundled ptxas 12.9.86 (rc=139) but compiles with ptxas 13.4.92 (rc=0)
  sha256 bd3a322f1882a9fff52cb0da5b3d002427575dc0e62f76f1a3f8970a9c7a39f6
  run: ptxas -lineinfo --fmad=false -v --regAllocOptLevel=2 --gpu-name=sm_89 kern2d_bf16_128x32_ax0_nw1.ptx -o o.cubin

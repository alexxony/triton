# colab-diffsweep-v2 — diff-sweep v2 code (stage 0 first)

Spec (pre-registered): vault `triton_00/notes/27-차등스윕v2-명세.md`. Target triton-lang/triton @ 93ba8872b6d783665af1cf662ebd48bda49e60b9.

- `sweep.py`, `sweep_kernels.py` : v1 snapshot from `colab-diffsweep-v1` @ bf94a93, unchanged.
- `dump_a_cfgs.json.gz` : v1 DUMP_A reduced to config records (964 of 1936 configs = all violating + 4 crash).
- `reach.py` : TTGIR reduction-reach parser (pure Python) + self-test. `python3 reach.py`.
- `probe_kernels.py` : upstream `test_argmax_argmin_with_nan` kernels, classified only.
- `stage0.py` : `selftest | probe | run` on a Colab CPU session with the main wheel installed.

Run everything on Colab (never locally). cwd must not be an ancestor of the triton source.

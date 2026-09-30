
import sys, triton, traceback
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import kern
dt, M, N, nw, axis = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sig = {"x_ptr": "*"+dt, "val_ptr": "*"+dt, "idx_ptr": "*i32", "M": "constexpr", "N": "constexpr", "AXIS": "constexpr", "OP": "constexpr", "TB": "constexpr"}
div = {(0,): [["tt.divisibility", 16]], (1,): [["tt.divisibility", 16]], (2,): [["tt.divisibility", 16]]}
src = ASTSource(fn=kern.kern2d, signature=sig, constexprs={"M": M, "N": N, "AXIS": axis, "OP": 2, "TB": False}, attrs=div)
try:
    k = triton.compile(src, target=GPUTarget("cuda", 89, 32), options={"num_warps": nw})
    print("COMPILED_OK", dt, M, N, nw, "ptx_len", len(k.asm["ptx"]))
except Exception as e:
    print("COMPILE_EXC", type(e).__name__, str(e).replace(chr(10), " | ")[:400])

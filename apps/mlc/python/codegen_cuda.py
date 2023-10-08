import tvm

from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        # for i_0 in T.thread_binding(8, thread="blockIdx.x"):
        #     for i_1 in T.thread_binding(128, thread="threadIdx.x"):
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    C[vi] = A[vi] + B[vi]


mod: tvm.runtime.Module = tvm.build(MyModule, target='cuda')
print(mod.imported_modules[0].get_source())
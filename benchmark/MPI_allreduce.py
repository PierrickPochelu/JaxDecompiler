from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax
from src.JaxDecompiler import decompiler

import numpy as np
import time

EXP = 10
ROUND = 4

# print(MPI.COMM_WORLD.allreduce(MPI.COMM_WORLD.Get_rank(), op=MPI.SUM))

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def BENCH(f, arg, message):
    _=f(arg)  # just in case warmup is important (JIT case)
    scores = []
    for i in range(EXP):
        st = time.time()
        y = f(arg)
        scores.append(time.time() - st)
    if rank == 0:
        print(f"{message} size:{size} mean:{round(np.mean(scores), ROUND)}, std:{round(np.std(scores), ROUND)}")


def unjitted_decompiled():
    def foo(arr):
        arr = arr + comm.Get_rank()
        arr_sum, _ = mpi4jax.allreduce(arr, op=MPI.SUM, comm=comm)
        return arr_sum

    a = jnp.zeros((3, 3))

    BENCH(foo, a, "unjitted_orig")

    python_code = decompiler.python_jaxpr_python(
        foo, (a,), is_python_returned=False, module_name="decompiled_module" + str(comm.Get_rank())
    )
    BENCH(python_code, a, "unjitted_decompiled")


def jitted_decompiled():
    @jax.jit
    def foo(arr):
        arr = arr + comm.Get_rank()
        arr_sum, _ = mpi4jax.allreduce(arr, op=MPI.SUM, comm=comm)
        return arr_sum

    a = jnp.zeros((3, 3))

    BENCH(foo, a, "jitted_orig")

    python_code = decompiler.python_jaxpr_python(
        foo, (a,), is_python_returned=False, module_name="decompiled_module" + str(comm.Get_rank())
    )

    BENCH(python_code, a, "jitted_decompiled")


unjitted_decompiled()
jitted_decompiled()

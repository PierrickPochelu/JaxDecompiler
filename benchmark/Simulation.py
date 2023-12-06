import sys
import time
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from src.JaxDecompiler import decompiler

ITERATIONS = int(sys.argv[1])
MOLECULES = int(sys.argv[2])  # Number of molecules

def BENCH(is_reconstructed, is_jitted):
    # Define the potential energy function and its gradient
    def fun(ra, rb, r0, k):
        rab = ra - rb
        rab = jnp.sqrt(jnp.dot(rab, rab))
        return 0.5 * k * (rab - r0) ** 2

    df = value_and_grad(fun)

    # Initial positions and parameters for each molecule
    mols = [jax.random.normal(shape=(3,), key=jax.random.PRNGKey(42), dtype=float) for j in range(MOLECULES)]

    r0 = 0.5
    k = 20.0



    # Gradient descent optimization loop
    learning_rate = 1e-3

    # Compile the jitted version for faster execution
    if is_reconstructed:
        df = decompiler.jaxpr2python(df, mols[0], mols[1], r0, k)

    if is_jitted:
        df = jit(df)

    st = time.time()

    for i in range(ITERATIONS):
        # Interaction loop - compute potential energy contributions for each pair of molecules
        grads=[None for j in range(MOLECULES)]
        for j in range(MOLECULES):

            for k in range(j + 1, MOLECULES):
                interaction_energy, grad_y = df(mols[j] , mols[k], r0, k)  # Assuming x1[0] is ra and x2[1] is rb

                # update the grads
                for gradid, molid in enumerate([j, k]):
                    if grads[molid] is None:
                        grads[molid]=grad_y[gradid]
                    else:
                        grads[molid]+=grad_y[gradid]

        for j,g in enumerate(grads):
            mols[j] = mols[j] - learning_rate * g



    elapsed_time = time.time() - st

    return elapsed_time

times = 1
for is_jitted in [True, False]:
    for is_reconstructed in [True, False]:
        cum = 0.
        for i in range(times):
            elapsed_time = BENCH(is_reconstructed, is_jitted)
            cum += elapsed_time
        print(f"is_reconstructed: {is_reconstructed}, is_jitted:{is_jitted} BENCH TOOK {round(cum/times, 2)}")

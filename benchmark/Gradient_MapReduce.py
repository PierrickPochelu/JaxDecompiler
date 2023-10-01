import sys


CPU = int(sys.argv[1])
NUM_DATA = int(sys.argv[2])
NUM_MIN = 3

print(f"Bench MapReduce: CPU={CPU} NUM_DATA={NUM_DATA} NUM_MIN={NUM_MIN}")

assert NUM_DATA % CPU == 0
import os

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CPU}"

from src.JaxDecompiler import decompiler
import time
import jax
import jax.numpy as jnp


print(jax.devices())
num_batches = len(jax.devices())

data = jnp.arange(0.0, 1.0, 1.0 / NUM_DATA)
data_batches = data.reshape(num_batches, len(data) // num_batches)

solution = jnp.arange(0.0, 1.0, 1.0 / NUM_DATA)
solution_batches = data.reshape(num_batches, len(data) // num_batches)

def L2(batch_data):
    return jnp.sort(batch_data)[:NUM_MIN]

def python_f(data_batches):
    local_results = jax.pmap(L2)(data_batches)
    result = jnp.concatenate(local_results, axis=0)
    return jnp.sum(jnp.sort(result)[:NUM_MIN])


start_time = time.time()
y_expected = jax.grad(python_f)(data_batches)
print(y_expected)
print("jaxpr time: ", time.time() - start_time)

start_time = time.time()
y_expected = jax.jit(jax.grad(python_f))(data_batches)
print("jitted jaxpr time: ", time.time() - start_time)

reconstructed_f = decompiler.python_jaxpr_python(python_f, (data_batches,))

start_time = time.time()
y = jax.grad(reconstructed_f)(data_batches)
print("JaxDecompile time: ", time.time() - start_time)

start_time = time.time()
y = jax.jit(jax.grad(reconstructed_f))(data_batches)
print("jitted JaxDecompile time: ", time.time() - start_time)

print("y_expected-y=", sum(y_expected) - sum(y))
print("out: ", y)

##################################
# CODE SAMPLES USED IN README.md #
##################################

# This code simulates the user code after instalation with `pip3 install JaxDecompiler`

import jax


def f(x, smooth_rate):
    local_minimums = (1 - smooth_rate) * jax.numpy.cos(x)
    global_minimum = smooth_rate * x**2
    return global_minimum + local_minimums


df = jax.grad(f, (0,))

from JaxDecompiler import decompiler  # <--- My contribution

# Let's display the jaxpr code
decompiler.display_wrapped_jaxpr(df, (1.0, 1.0))

# Let's decompile it
decompiled_df, python_code = decompiler.python_jaxpr_python(
    df, (1.0, 1.0), is_python_returned=True
)

# Let's compare the output
print("df: ", df(4.0, 0.99))  # jaxpr wrapped code
print("decompiled df: ", decompiled_df(4.0, 0.99))  # raw python code

# Now let's display the python code of the derivative:
print(python_code)

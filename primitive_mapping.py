from jax import numpy as jnp

def add(input_var,output_var, params):
    rvalue=" + ".join(input_var) # 1 or 2
    lvalue=output_var[0]
    return f"{lvalue} = {rvalue}"

def mul(input_var,output_var, params):
    rvalue=" * ".join(input_var) # 1 or 2
    lvalue=output_var[0]
    return f"{lvalue} = {rvalue}"

def div(input_var, output_var, params):
    rvalue=" / ".join(input_var) # 1 or 2
    lvalue=output_var[0]
    return f"{lvalue} = {rvalue}"

def log(input_var, output_var, params):
    rvalue=input_var[0]
    lvalue=output_var[0]
    return f"{lvalue} = log({rvalue})"

def exp(input_var, output_var, params):
    rvalue=input_var[0]
    lvalue=output_var[0]
    return f"{lvalue} = exp({rvalue})"

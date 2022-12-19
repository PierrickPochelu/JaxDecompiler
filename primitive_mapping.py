from typing import *
import jax
from jax import numpy as jnp
import os

_LOCAL_F_COUNT = 0


def _recurive_op(params, python_call, local_f_name):
    # Recursive calls
    K = params["decompiler_K"]
    _line_body = params["decompiler_line_body"]
    _line_input = params["decompiler_line_input"]
    _line_return = params["decompiler_line_return"]
    _tab = params["decompiler_tab"]

    # collect input/output vars
    input_vars = [str(v) for v in params["call_jaxpr"].invars]
    output_vars = [str(v) for v in params["call_jaxpr"].outvars]

    args = ", ".join(input_vars)
    l = f"def {local_f_name}({args}):"
    input_local_f_lines = [l]

    return_vars = ", ".join(list(output_vars))
    l = f"return {return_vars}"
    tabbed_l = _tab(l, 1)
    output_local_f_lines = [tabbed_l]

    # build body of the global function and the local functions
    body_local_f_lines = []
    for eqn in params["call_jaxpr"].eqns:
        python_lambda_body = _line_body(eqn, K, 1)  # 'ex: "    b = a + 1.0"
        body_local_f_lines.append(python_lambda_body)

    # call line
    call_local_f_lines = [python_call]

    # MERGE THE LOCAL FUNCTION CODES AND THE CALL CODE
    out = (
        input_local_f_lines
        + body_local_f_lines
        + output_local_f_lines
        + call_local_f_lines
    )

    return out


def add(input_var, output_var, params):
    rvalue = " + ".join(input_var)  # 2
    lvalue = output_var[0]
    return f"{lvalue} = {rvalue}"


def add_any(input_var, output_var, params):
    return add(input_var, output_var, params)


def mul(input_var, output_var, params):
    rvalue = " * ".join(input_var)  # 2
    lvalue = output_var[0]
    return f"{lvalue} = {rvalue}"


def sub(input_var, output_var, params):
    rvalue = " - ".join(input_var)  # 2
    lvalue = output_var[0]
    return f"{lvalue} = {rvalue}"


def neg(input_var, output_var, params):
    return f"{output_var[0]} = -{input_var[0]}"


def div(input_var, output_var, params):
    rvalue = " / ".join(input_var)  # 2
    lvalue = output_var[0]
    return f"{lvalue} = {rvalue}"


def integer_pow(input_var, output_var, params):
    return f"{output_var[0]} = {input_var[0]} ** {params['y']}"


def pow(input_var, output_var, params):
    return f"{output_var[0]} = {input_var[0]} ** {input_var[1]}"


def sqrt(input_var, output_var, params):
    return f"{output_var[0]} = sqrt({input_var[0]})"


def log(input_var, output_var, params):
    rvalue = input_var[0]
    lvalue = output_var[0]
    return f"{lvalue} = log({rvalue})"


def exp(input_var, output_var, params):
    rvalue = input_var[0]
    lvalue = output_var[0]
    return f"{lvalue} = exp({rvalue})"


def dot_general(input_var, output_var, params):
    rvalue = ", ".join(input_var)
    lvalue = output_var[0]
    return f"{lvalue} = dot({rvalue})"


def cos(input_var, output_var, params):
    return f"{output_var[0]} = cos({input_var[0]})"


def sin(input_var, output_var, params):
    return f"{output_var[0]} = sin({input_var[0]})"


def tan(input_var, output_var, params):
    return f"{output_var[0]} = tan({input_var[0]})"


def tanh(input_var, output_var, params):
    return f"{output_var[0]} = tanh({input_var[0]})"


def acos(input_var, output_var, params):
    return f"{output_var[0]} = arccos({input_var[0]})"


def asin(input_var, output_var, params):
    return f"{output_var[0]} = arcsin({input_var[0]})"


def atan(input_var, output_var, params):
    return f"{output_var[0]} = arctan({input_var[0]})"


def copy(input_var, output_var, params):
    return f"{output_var[0]} = jax.numpy.copy({input_var[0]})"


def convert_element_type(input_var, output_var, params):
    t = params["new_dtype"]
    return f"{output_var[0]} = array({input_var[0]}).astype({t})"


def xla_pmap(input_var, output_var, params) -> List[Union[List, str]]:
    global _LOCAL_F_COUNT
    local_f_name = "local_f" + str(_LOCAL_F_COUNT)
    _LOCAL_F_COUNT += 1

    lvalue = ", ".join(output_var)
    rvalue = ", ".join(input_var)
    l = f"{lvalue} = jax.pmap({local_f_name})({rvalue})"

    lines = _recurive_op(params, l, local_f_name)
    return lines


def xla_call(input_var, output_var, params) -> List[Union[List, str]]:
    global _LOCAL_F_COUNT
    local_f_name = "local_f" + str(_LOCAL_F_COUNT)
    _LOCAL_F_COUNT += 1

    lvalue = ", ".join(output_var)
    rvalue = ", ".join(input_var)
    l = f"{lvalue} = {local_f_name}({rvalue})"

    lines = _recurive_op(params, l, local_f_name)
    return lines

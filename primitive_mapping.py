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


def floor(input_var, output_var, params):
    return f"{output_var[0]} = floor({input_var[0]})"


def ceil(input_var, output_var, params):
    return f"{output_var[0]} = ceil({input_var[0]})"


def round(input_var, output_var, params):
    return f"{output_var[0]} = round({input_var[0]})"


def clamp(input_var, output_var, params):
    v = input_var[1]
    minv = input_var[0]
    maxv = input_var[2]
    return f"{output_var[0]} = clip({v}, {minv}, {maxv})"


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


def reshape(input_var, output_var, params):
    new_sizes = params["new_sizes"]
    return f"{output_var[0]} = array({input_var[0]}).reshape({new_sizes})"


def gather(input_var, output_var, params):
    # np.take_along_axis(a, ai, axis=1)
    rvalue = ", ".join(input_var)
    dim_num_obj = params["dimension_numbers"]
    dims = dim_num_obj.collapsed_slice_dims[0]
    return f"{output_var[0]} = take_along_axis({input_var[0]}, {input_var[1]}, {dims})"


def squeeze(input_var, output_var, params):
    return f"{output_var[0]} = squeeze({input_var[0]})"


def argmin(input_var, output_var, params):
    return f"{output_var[0]} = argmin({input_var[0]})"


def argmax(input_var, output_var, params):
    return f"{output_var[0]} = argmax({input_var[0]})"


def reduce_min(input_var, output_var, params):
    return f"{output_var[0]} = min({input_var[0]})"


def reduce_max(input_var, output_var, params):
    return f"{output_var[0]} = max({input_var[0]})"


def reduce_sum(input_var, output_var, params):
    return f"{output_var[0]} = sum({input_var[0]})"


def broadcast_in_dim(input_var, output_var, params):
    rvalue = ",".join(input_var)
    shape = params["shape"]
    return f"{output_var[0]} = array(broadcast_to({rvalue}, {shape}))"


def select_n(input_var, output_var, params):
    # jaxpr: (pred, on_false, on_true) nump: (condlist, choicelist, default)
    pred = input_var[0]
    on_false = input_var[1]
    on_true = input_var[2]

    condlist = f"[{pred},invert({pred})]"
    choicelist = f"[{on_true},{on_false}]"
    line = f"{output_var[0]} = select({condlist}, {choicelist})"
    # condlist=f"repeat({on_true},len({on_true})//len({pred}))"
    # choicelist=f"{pred}"
    # default=f"repeat({on_false},len({on_false})//len({pred}))"
    # line=f"{output_var[0]} = select({condlist}, {choicelist},  {default})"
    return line


def ne(input_var, output_var, params):  # element wise not equal
    rvalue = "!=".join(input_var)
    return f"{output_var[0]} = {rvalue}"


def eq(input_var, output_var, params):  # element wise not equal
    rvalue = "==".join(input_var)
    return f"{output_var[0]} = {rvalue}"


def sort(input_var, output_var, params):
    rvalue = ", ".join(input_var)
    axis = params["dimension"]
    return f"{output_var[0]} = sort({rvalue},axis={axis})"


def reduce_or(input_var, output_var, params):
    rvalue = ", ".join(input_var)
    return f"{output_var[0]} = any({rvalue})"


def reduce_and(input_var, output_var, params):
    rvalue = ", ".join(input_var)
    return f"{output_var[0]} = all({rvalue})"


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

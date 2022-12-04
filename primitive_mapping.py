import jax
from jax import numpy as jnp
import os


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


def xla_pmap(input_var, output_var, params):
    out = ""
    # out+="import multiprocessing"
    # out+="pool = multiprocessing.Pool()"

    # Recursive calls
    K = params["decompiler_K"]
    _line_body = params["decompiler_line_body"]
    _line_input = params["decompiler_line_input"]
    _line_return = params["decompiler_line_return"]
    _tab = params["decompiler_tab"]

    # comput input line
    local_f_name = "g"
    input_vars = set([])
    for eqn in params["call_jaxpr"].eqns:
        for v in eqn.invars:
            if isinstance(v, jax.core.Var):
                input_vars.add(str(v))
    args = ", ".join(input_vars)
    l = f"def {local_f_name}({args}):"
    input_local_f_lines = [l]

    # build body of the function
    body_local_f_lines = []
    for eqn in params["call_jaxpr"].eqns:
        python_lambda_body = _line_body(eqn, K, 1)  # 'ex: "    b = a + 1.0"
        body_local_f_lines.append(python_lambda_body)

    # compute output line
    output_vars = set([])
    for eqn in params["call_jaxpr"].eqns:
        for v in eqn.outvars:
            output_vars.add(str(v))
    return_vars = ", ".join(list(output_vars))
    l = f"return {return_vars}"
    tabbed_l = _tab(l, 1)
    output_local_f_lines = [tabbed_l]

    # call line
    lvalue = ", ".join(output_var)
    rvalue = ", ".join(input_var)
    l = f"{lvalue} = jax.pmap({local_f_name})({rvalue})"
    call_local_f_lines = [l]

    # MERGE THE LOCAL FUNCTION CODES AND THE CALL CODE
    out = (
        input_local_f_lines
        + body_local_f_lines
        + output_local_f_lines
        + call_local_f_lines
    )

    return out

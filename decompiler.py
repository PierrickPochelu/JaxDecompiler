import gc
import importlib
import os.path
import sys
import time
from typing import *
import jax
from jax import numpy as jnp
from os import path, linesep


def from_jaxpr_object_to_python(
        jaxpr_obj, module_name="decompiled_module", dir_path="out", is_python_returned=False
) -> Union[Callable, Tuple[Callable, str]]:
    """from jaxpr code to python code"""
    python_lines = decompiler(jaxpr_obj)

    f = from_strings_to_callable(
        python_lines, module_name=module_name, dir_path=dir_path
    )
    if is_python_returned:
        return f, (os.linesep).join(python_lines)
    else:
        return f


def python_jaxpr_python(
        python_f, moc_inputs, **kwargs
) -> Union[Callable, Tuple[Callable, str]]:
    """Compilation followed by Decompilation allows to check if the decompilation is correct
    We assume Compilation is always correct here.
    Therefore, input program should be similar to the output program"""
    jaxpr_obj = jax.make_jaxpr(python_f)(*moc_inputs)

    out = from_jaxpr_object_to_python(jaxpr_obj, **kwargs)

    return out  # can be either (f:Callable,code:str) or f:Callable


def display_wrapped_jaxpr(python_f, x) -> None:
    jaxpr_obj = jax.make_jaxpr(python_f)(*x)
    jaxpr_code = jaxpr_obj.jaxpr
    """used in development phase to build the instruction table named 'primitive_mapping' """
    print("===== HEADER =======")
    print("invars:", jaxpr_code.invars)
    print("outvars:", jaxpr_code.outvars)
    print("constvars:", jaxpr_code.constvars)
    # for eqn in jaxpr_code.eqns:
    #    print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
    print("===== CODE =======")
    print(jaxpr_obj)


def get_primitive_mapping() -> Dict[str, Callable]:
    import primitive_mapping

    K = {}
    for i in dir(primitive_mapping):
        att = getattr(primitive_mapping, i)
        if callable(att):
            K[i] = att
    return K


def _tab(python_line, tab_level) -> str:
    tab_prefix = " " * 4 * tab_level
    return tab_prefix + python_line


def _tab_recursively(lines, tab_level) -> None:  # lines is I/O
    for i, l in enumerate(lines):
        if isinstance(l, str):
            lines[i] = _tab(l, tab_level)
        elif isinstance(l, list):
            _tab_recursively(l, tab_level)
        else:
            raise ValueError("Unexpected type in _tab_recursively()")


def _line_input(jaxpr, tab_level=0, python_func_name="f") -> str:
    if not hasattr(jaxpr, "invars"):
        return "def f():"
    str_input = [str(var) for var in jaxpr.invars]
    args = ", ".join(str_input)
    python_body_line = f"def {python_func_name}({args}):"
    line = _tab(python_body_line, tab_level)
    return line


def _line_return(jaxpr, tab_level=1) -> str:
    if not hasattr(jaxpr, "outvars"):
        python_body_line = "pass #no output"
    else:
        str_output = [str(var) for var in jaxpr.outvars]
        args = ", ".join(str_output)
        python_body_line = f"return {args}"
    line = _tab(python_body_line, tab_level)
    return line


def decompile_type_convert(
        v,
):  # TODO for instance types are partially ignored (default python behaviour)
    """From jaxpr object token to Python string token"""
    """type(v) in {jax.core.Var, jax.core.Literal}."""

    if isinstance(v, jax.core.Literal):
        internal_type = v.aval.dtype
        dimensions = v.aval.ndim
        shape = v.aval.shape
        v2 = str(v)
    elif isinstance(v, jax.core.Var):
        internal_type = v.aval.dtype
        dimensions = v.aval.ndim
        shape = v.aval.shape
        v2 = str(v)
    else:
        print(f"WARNING jaxpr token type not understood: {type(v)}")
        v2 = str(v)
    return v2


def _line_body(eqn, K, tab_level) -> List[Union[List, str]]:
    python_op_name = str(eqn.primitive.name)
    jaxpr_line = str(eqn)

    if python_op_name not in K:
        raise TypeError(f'Instruction: "{python_op_name}" not yet supported')

    # Check if we need it. It is usefull for pmap
    # if "call_jaxprs" in eqn.params:
    eqn.params["decompiler_line_input"] = _line_input
    eqn.params["decompiler_line_body"] = _line_body
    eqn.params["decompiler_line_return"] = _line_return
    eqn.params["decompiler_K"] = K
    eqn.params["decompiler_tab"] = _tab

    # input_var of the function (function inputs, or variable inside)
    # lvars, rvars=_lvar_rvar(jaxpr_line)
    # input_var = [str(var) for var in rvars]
    # output_var = [str(var) for var in lvars]

    # before:
    input_var = [decompile_type_convert(var) for var in eqn.invars]
    output_var = [decompile_type_convert(var) for var in eqn.outvars]

    # build the line as string
    python_body_line_builder = K[python_op_name]
    params = {}
    lines = python_body_line_builder(input_var, output_var, eqn.params)
    if isinstance(lines, str):
        lines = [lines]

    _tab_recursively(lines, tab_level)  # python_body_line is updated
    return lines


def import_statements(tabbed_python_lines) -> None:
    tabbed_python_lines.append("import jax")
    tabbed_python_lines.append("from jax.numpy import *")


def decompiler(
        jaxpr_obj, starting_tab_level=0, python_func_name="f"
) -> List[Union[List, str]]:
    jaxpr_code = jaxpr_obj.jaxpr
    K = get_primitive_mapping()
    tabbed_python_lines = []

    import_statements(tabbed_python_lines)

    p = _line_input(jaxpr_code, starting_tab_level, python_func_name=python_func_name)
    tabbed_python_lines.append(p)

    # Constants
    for var_name, var_val in zip(jaxpr_code.constvars, jaxpr_obj.literals):
        var_val_literal = repr(var_val)  # from jaxpr object to string literal
        p = _tab(f"{var_name} = {var_val_literal}", starting_tab_level + 1)
        tabbed_python_lines.append(p)

    # body of the function
    for eqn in jaxpr_code.eqns:
        list_of_python_lines = _line_body(eqn, K, starting_tab_level + 1)
        tabbed_python_lines.extend(list_of_python_lines)

    # return instruction
    p = _line_return(jaxpr_code, starting_tab_level + 1)
    tabbed_python_lines.append(p)

    return tabbed_python_lines


def _recursively_write_python_program(
        file, lines
) -> None:  # warning lines may contains lines
    for line in lines:
        if isinstance(line, list):
            _recursively_write_python_program(file, line)
        elif isinstance(line, str):
            file.write(line + os.linesep)
        else:
            raise ValueError("unexpected event occurs")


def from_strings_to_callable(
        python_lines, module_name="decompiled_module", dir_path="out"
) -> Callable:
    """warning this function create a file named `tmp_file` in the directory `dir_path`"""

    # Write it
    file_path = path.join(dir_path, module_name + ".py")
    with open(file_path, "w") as file:
        _recursively_write_python_program(file, python_lines)

    # Import it
    if dir_path not in sys.path:  # TODO can we do better ? O(n) linear scan
        sys.path.append(dir_path)
    # refresh
    if module_name in sys.modules:
        del sys.modules[module_name]
        time.sleep(1)  # TODO: Why is it mandatory ? Urgent patch needed here
    module = __import__(module_name)
    # importlib.reload(importlib.import_module(module_name))

    callable_f = module.f

    return callable_f

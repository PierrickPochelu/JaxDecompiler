import gc
import importlib
import os.path
import sys
import time

import jax
from jax import numpy as jnp
from os import path, linesep


def _examine_jaxpr(python_f, x):
    jaxpr_object = jax.make_jaxpr(python_f)(x).jaxpr
    """used in development phase to build the instruction table named 'primitive_mapping' """
    print("invars:", jaxpr_object.invars)
    print("outvars:", jaxpr_object.outvars)
    print("constvars:", jaxpr_object.constvars)
    for eqn in jaxpr_object.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
    print("=====")


def get_primitive_mapping():
    import primitive_mapping
    K = {}
    for i in dir(primitive_mapping):
        att = getattr(primitive_mapping, i)
        if callable(att):
            K[i] = att
    return K


def _tab(python_line, tab_level):
    tab_prefix = " " * 4 * tab_level
    return tab_prefix + python_line


def _line_input(jaxpr, tab_level=0):
    if not hasattr(jaxpr, 'invars'):
        return "def f():"
    str_input = [str(var) for var in jaxpr.invars]
    args = ', '.join(str_input)
    python_body_line = f"def f({args}):"
    line = _tab(python_body_line, tab_level)
    return line


def _line_return(jaxpr, tab_level=1):
    if not hasattr(jaxpr, 'outvars'):
        python_body_line = "pass #no output"
    else:
        str_output = [str(var) for var in jaxpr.outvars]
        args = ', '.join(str_output)
        python_body_line = f"return {args}"
    line = _tab(python_body_line, tab_level)
    return line


def _lvar_rvar(jaxpr_line):
    """Not used anymore"""
    # typically jaxpr_line=="a:f32[3,3] = add b c"
    sides = jaxpr_line.split("=")
    sides = [s.strip() for s in sides]

    left_vars = []
    for pair in sides[0].split(" "):
        tokens = pair.split(":")
        if len(tokens) == 2:
            var_name, var_type = tokens
        elif len(tokens) == 1:
            var_name = tokens
        else:
            raise ValueError("Unexpected number of tokens")

        left_vars.append(var_name)
    # left_vars==["a"]

    right = sides[1].split(" ")  # ["add", "b", "c"]
    right_vars = right[1:]  # ["b", "c"]

    return left_vars, right_vars


def decompile_type_convert(v):
    """From jaxpr object token to Python string token"""
    """type(v) in {jax.core.Var, jax.core.Literal}."""

    if isinstance(v, jax.core.Literal):
        internal_type = v.aval.dtype
        dimensions = v.aval.ndim
        shape = v.aval.shape

        if dimensions == 0:
            v2 = str(v)
        else:
            pass  # we need to convert v into a numpy array right now
    elif isinstance(v, jax.core.Var):
        internal_type = v.aval.dtype
        dimensions = v.aval.ndim
        shape = v.aval.shape
        v2 = str(v)
    else:
        print(f"WARNING jaxpr token type not understood: {type(v)}")
        v2 = str(v)
    return v2


def _line(eqn, K, tab_level=1):
    python_op_name = str(eqn.primitive.name)
    jaxpr_line = str(eqn)

    if python_op_name not in K:
        raise NotImplemented(f"Instruction: \"{python_op_name}\" not yet supported")

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
    python_body_line = python_body_line_builder(input_var, output_var, params)
    line = _tab(python_body_line, tab_level)
    return line

def import_statements(tabbed_python_lines):
    tabbed_python_lines.append("import jax")
    tabbed_python_lines.append("from jax.numpy import *")

def decompiler(jaxpr_obj, K):
    jaxpr_code=jaxpr_obj.jaxpr


    tabbed_python_lines = []

    import_statements(tabbed_python_lines)

    p = _line_input(jaxpr_code, 0)
    tabbed_python_lines.append(p)

    # Constants
    for var_name, var_val in zip(jaxpr_code.constvars, jaxpr_obj.literals):
        var_val_literal=repr(var_val) #from jaxpr object to string literal
        p=_tab(f"{var_name} = {var_val_literal}",1)
        tabbed_python_lines.append(p)

    # body of the function
    for eqn in jaxpr_code.eqns:
        p = _line(eqn, K, 1)
        tabbed_python_lines.append(p)

    p = _line_return(jaxpr_code, 1)
    tabbed_python_lines.append(p)
    return tabbed_python_lines


def from_strings_to_callable(python_lines, module_name="decompiled_module", dir_path="out"):
    """warning this function create a file named `tmp_file` in the directory `dir_path` """

    # Write it
    file_path = path.join(dir_path, module_name + ".py")
    with open(file_path, "w") as file:
        for line in python_lines:
            file.write(line + linesep)

    # Import it
    if dir_path not in sys.path:  # TODO can we do better ? O(n) linear scan
        sys.path.append(dir_path)
    # refresh
    if module_name in sys.modules:
        del sys.modules[module_name]
        time.sleep(1)  # TODO: Why is it mandatory ?
    module = __import__(module_name)
    # importlib.reload(importlib.import_module(module_name))

    callable_f = module.f

    return callable_f

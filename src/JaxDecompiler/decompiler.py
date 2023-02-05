import os.path
import sys
from typing import *
import jax
from os import path

# Keep the alphabetical order below for readability purpose.
PYTHON_KEY_WORDS = {"id", "if", "in", "or", "is", "def", "del", "for", "not", "set", "try", "elif", "else", "from"}


def from_jaxpr_object_to_python(
        jaxpr_obj, module_name="decompiled_module", dir_path="out", is_python_returned=False
) -> Union[Callable, Tuple[Callable, str]]:
    """from jaxpr code to python code"""
    python_lines = decompiler(jaxpr_obj)

    f = _from_strings_to_callable(
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


def _filter_var_name(var_name):
    """
    All jaxpr vars should be given to this function
    Automatic variable names are: a, b, c, ... y, z, aa, ab, ac, ... ic, ID, ie, IF, ig ...
    """
    if var_name in PYTHON_KEY_WORDS:
        return var_name.upper()
    return var_name


def _get_primitive_mapping() -> Dict[str, Callable]:
    from JaxDecompiler import primitive_mapping

    K = {}
    for i in dir(primitive_mapping):
        att = getattr(primitive_mapping, i)
        if callable(att):
            K[i] = att
    return K


def _tab(python_line, tab_level) -> str:
    tab_prefix = " " * 4 * tab_level
    return tab_prefix + python_line


def _tab_recursively(lines, tab_level) -> List[str]:  # lines is I/O
    strings = []
    for i, l in enumerate(lines):
        if isinstance(l, str):
            tabbed_str = _tab(l, tab_level)
            lines[i] = tabbed_str
            strings.append(tabbed_str)
        elif isinstance(l, list):
            nested_strings = _tab_recursively(l, tab_level)
            strings.extend(nested_strings)
        else:
            raise ValueError("Unexpected type in _tab_recursively()")
    return strings


def _lines_constant(jaxpr_constvars, jaxpr_literals, tab_level) -> List[str]:
    list_python_lines = []
    for var_name, var_val in zip(jaxpr_constvars, jaxpr_literals):
        var_name = _filter_var_name(str(var_name))
        var_val_literal = repr(var_val)  # from jaxpr object to string literal

        # note the whitespace, "Array"->"array" "DeviceArray"->"DeviceArray"
        var_val_literal = var_val_literal.replace("DeviceArray", "array")
        var_val_literal = var_val_literal.replace("Array", "array")

        line = f"{var_name} = {var_val_literal}"
        tabbed_line = _tab(line, tab_level)
        list_python_lines.append(tabbed_line)
    return list_python_lines


def _line_input(jaxpr, tab_level=0, python_func_name="f") -> str:
    if not hasattr(jaxpr, "invars"):
        return "def f():"
    str_input = [_filter_var_name(str(var)) for var in jaxpr.invars]
    args = ", ".join(str_input)
    python_body_line = f"def {python_func_name}({args}):"
    line = _tab(python_body_line, tab_level)
    return line


def _line_return(jaxpr, tab_level=1) -> str:
    if not hasattr(jaxpr, "outvars"):
        python_body_line = "pass #no output"
    else:
        str_output = [_filter_var_name(str(var)) for var in jaxpr.outvars]
        args = ", ".join(str_output)
        python_body_line = f"return {args}"
    line = _tab(python_body_line, tab_level)
    return line


def _line_body(eqn, K, tab_level) -> List[Union[List, str]]:
    jaxpr_op_name = str(eqn.primitive.name)

    # Due to some python naming constraints, we create some exceptions to the mapping
    # jaxpr function name -> python function name
    python_op_name = jaxpr_op_name.replace("-", "_")  # e.g., jaxpr "scatter-add" become python "scatter_add"
    if python_op_name in PYTHON_KEY_WORDS:
        python_op_name = python_op_name + "__"  # e.g., jaxpr "or" become python "or__"

    if python_op_name not in K:
        raise TypeError(f'Instruction: "{python_op_name}" not yet supported')

    # Below code is usefull for pmap, more generally if "call_jaxprs" in eqn.params:
    eqn.params["decompiler_line_input"] = _line_input
    eqn.params["decompiler_line_body"] = _line_body
    eqn.params["decompiler_line_return"] = _line_return
    eqn.params["decompiler_K"] = K
    eqn.params["decompiler_tab"] = _tab
    eqn.params["decompiler_filter_var_name"] = _filter_var_name

    # process var names
    input_var = [_filter_var_name(str(var)) for var in eqn.invars]
    output_var = [_filter_var_name(str(var)) for var in eqn.outvars]

    # build the line as string
    python_body_line_builder = K[python_op_name]
    lines = python_body_line_builder(input_var, output_var, eqn.params)
    # lines can be a str or a list of [str,list]
    if isinstance(lines, str):
        lines = [lines]

    list_python_lines = _tab_recursively(
        lines, tab_level
    )  # python_body_line is updated

    return list_python_lines


def _import_statements(tabbed_python_lines) -> None:
    tabbed_python_lines.append("import jax")
    tabbed_python_lines.append("from jax.numpy import *")
    tabbed_python_lines.append("from jax._src import prng")


def decompiler(
        jaxpr_obj, starting_tab_level=0, python_func_name="f"
) -> List[Union[List, str]]:
    jaxpr_code = jaxpr_obj.jaxpr
    K = _get_primitive_mapping()
    tabbed_python_lines = []

    _import_statements(tabbed_python_lines)

    p = _line_input(jaxpr_code, starting_tab_level, python_func_name=python_func_name)
    tabbed_python_lines.append(p)

    # Constants
    l = _lines_constant(jaxpr_code.constvars, jaxpr_obj.literals, starting_tab_level + 1)
    tabbed_python_lines.extend(l)

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


def _from_strings_to_callable(
        python_lines, module_name="decompiled_module", dir_path="out"
) -> Callable:
    """warning this function create a file named "`module_name`.py" in the directory `dir_path`"""

    # Write folder out/ if not present yet
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Write it
    file_path = path.join(dir_path, module_name + ".py")
    with open(file_path, "w") as file:
        _recursively_write_python_program(file, python_lines)

    # Import it
    if dir_path not in sys.path:
        sys.path.append(dir_path)

    # refresh
    if module_name in sys.modules:
        # https://stackoverflow.com/questions/74891109/write-import-call-same-python-module-multiple-times-runs-outdated-code?noredirect=1#comment132184787_74891109
        os.remove(sys.modules[module_name].__cached__)  # remove cached bytecode
        del sys.modules[module_name]

    module = __import__(module_name)

    callable_f = module.f

    return callable_f

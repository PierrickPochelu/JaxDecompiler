from typing import *
import builtins

_LOCAL_F_COUNT = 0

def _recurive_op(params, python_call, local_f_name):
    # Recursive calls
    K = params["decompiler_K"]
    _line_body = params["decompiler_line_body"]
    _line_input = params["decompiler_line_input"]
    _line_return = params["decompiler_line_return"]
    _tab = params["decompiler_tab"]
    filter_var_name = params["decompiler_filter_var_name"]

    # collect input/output vars
    input_vars = [filter_var_name(str(v)) for v in params["call_jaxpr"].invars]
    output_vars = [filter_var_name(str(v)) for v in params["call_jaxpr"].outvars]

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
    arr = input_var[0]

    start_indices = input_var[1]  # ex: [0,0]
    slice_sizes = list(params["slice_sizes"])  # [1,1,7]

    # while(len(start_indices) < len(slice_sizes)):
    #    start_indices.append(0)

    slicing_code = "["
    d = 0
    for end in slice_sizes:
        start = f"{start_indices}[{d}]"
        dim_slice = f"{start} if len({start_indices})>{d} else 0:{start}+{end},"
        slicing_code += dim_slice
        d += 1
    slicing_code = slicing_code[:-1]
    slicing_code += "]"

    dim_num_obj = params["dimension_numbers"]
    collapsed_dims = dim_num_obj.collapsed_slice_dims

    return f"{output_var[0]} = squeeze( array({arr}{slicing_code}) , axis={collapsed_dims})"

def random_seed(input_var, output_var, params):
    #return f"{output_var[0]} = random.seed({input_var[0]})"
    impl_obj=params["impl"]
    PRNG_IMPLS = {
        'threefry2x32': "prng.threefry_prng_impl",
        'rbg': "prng.rbg_prng_impl",
        'unsafe_rbg': "prng.unsafe_rbg_prng_impl",
        "fry":  "prng.threefry_prng_impl",
    }
    impl=PRNG_IMPLS[impl_obj.tag]
    return f"{output_var[0]} = jax.random.PRNGKeyArray(key_data=jax.random.PRNGKey({input_var[0]}),impl={impl})"

def random_unwrap(input_var, output_var, params):
    #return f"{output_var[0]} = {input_var[0]}.unwrap()"
    return f"{output_var[0]} = prng.random_unwrap({input_var[0]})"
def random_wrap(input_var, output_var, params):
    impl_obj=params["impl"]
    PRNG_IMPLS = {
        'threefry2x32': "prng.threefry_prng_impl",
        'rbg': "prng.rbg_prng_impl",
        'unsafe_rbg': "prng.unsafe_rbg_prng_impl",
        "fry":  "prng.threefry_prng_impl",
    }
    impl=PRNG_IMPLS[impl_obj.tag]

    return f"{output_var[0]} = prng.random_wrap({input_var[0]}, impl={impl})"

def random_bits(input_var, output_var, params):
    options=f"bit_width={params['bit_width']}, shape={params['shape']}"
    return f"{output_var[0]} = prng.random_bits({input_var[0]}, {options})"

def shift_right_logical(input_var, output_var, params):
    return f"{output_var[0]} = right_shift({input_var[0]}, {input_var[1]})"

def shift_left_logical(input_var, output_var, params):
    return f"{output_var[0]} = left_shift({input_var[0]}, {input_var[1]})"

def concatenate(input_var, output_var, params):
    dim = params["dimension"]
    rvalue = ", ".join(input_var)
    return f"{output_var[0]} = concatenate(({rvalue}), axis={dim})"


def squeeze(input_var, output_var, params):
    return f"{output_var[0]} = squeeze(array({input_var[0]}))"


def argmin(input_var, output_var, params):
    return f"{output_var[0]} = argmin({input_var[0]})"


def argmax(input_var, output_var, params):
    return f"{output_var[0]} = argmax({input_var[0]})"

def min(input_var, output_var, params):
    return f"{output_var[0]} = array([min({input_var[0]})])"

def reduce_min(input_var, output_var, params):
    return f"{output_var[0]} = min({input_var[0]})"

def max(input_var, output_var, params):
    return f"{output_var[0]} = array([max({input_var[0]})])"

def reduce_max(input_var, output_var, params):
    return f"{output_var[0]} = max({input_var[0]})"

def abs(input_var, output_var, params):
    return f"{output_var[0]} = abs({input_var[0]})"

def sign(input_var, output_var, params):
    return f"{output_var[0]} = sign({input_var[0]})"

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
    return line


def ne(input_var, output_var, params):  # element wise not equal
    rvalue = "!=".join(input_var)
    return f"{output_var[0]} = {rvalue}"


def eq(input_var, output_var, params):  # element wise not equal
    rvalue = "==".join(input_var)
    return f"{output_var[0]} = {rvalue}"

def ge(input_var, output_var, params):
    rvalue = ">=".join(input_var)
    return f"{output_var[0]} = {rvalue}"
def gt(input_var, output_var, params):
    rvalue = ">=".join(input_var)
    return f"{output_var[0]} = {rvalue}"
def le(input_var, output_var, params):
    rvalue = "<=".join(input_var)
    return f"{output_var[0]} = {rvalue}"

def lt(input_var, output_var, params):
    rvalue = "<=".join(input_var)
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


def rev(input_var, output_var, params):
    axis = params["dimensions"]
    return f"{output_var[0]} = flip({input_var[0]},axis={axis})"


def conv_general_dilated(input_var, output_var, params):
    shape = params["lhs_shape"]
    return f"{output_var[0]} = convolve(squeeze({input_var[0]}), squeeze({input_var[1]}), mode='same').reshape({shape})"


def dynamic_slice(input_var, output_var, params):  # TODO: unit test
    a, b = input_var
    ss = params["slice_sizes"]
    #return f"{output_var[0]} = {a}[{b}:{b}+{ss}[0]] # dynamic slice"
    return f"{output_var[0]} = jax.lax.dynamic_slice_in_dim({a}, {b}, {ss}[0], axis=0)"

def slice(input_var, output_var, params):  # TODO: unit test
    start = params['start_indices']  # e.g. "(1, 2)", "(0, )", "a"
    limit = params['limit_indices']
    strides = params['strides']

    # Example 1: ["(1, 2)", "(3, 4)", "(5, 6)"] -> we want "[1:3:5][2:4:6]"
    # Example 2: ["(1, 2)", "(3, 4)", None] -> we want "[1:3:][2:4:]"
    inputs = [start, limit, strides]

    # splits:
    # Example 1: ["(1, 2)", "(3, 4)", "(5, 6)"] -> [['1','3','5']['2','4','6']]
    # Example 2: ["(1, 2)", "(3, 4)"] -> [['1','2']['3','4'],[]]
    splitted_inputs = []
    for i, v in enumerate(inputs):
        if v is not None:
            v = str(v)
            for symbol in ["(", ")", " "]:
                v = v.replace(symbol, "")
            v_splited = v.split(",")
            if v_splited[-1]=="": #(1,) -> ["1"] not ["1", ""]
                v_splited.pop(-1)
            splitted_inputs.append(v_splited)
        else:
            splitted_inputs.append([])

    # balancing dimension insructions:
    # Example: [['1','2']['3','4'],[]] -> [['1','2']['3','4'],['','']]
    max_depth = builtins.max([len(v) for v in splitted_inputs])
    balanced_splitted_inputs = []
    for v in splitted_inputs:
        balanced_v = v[:]
        for i in range(len(balanced_v), max_depth):
            balanced_v.append("")
        balanced_splitted_inputs.append(balanced_v)

    # create slicing
    # [['1', '2']['3', '4'], ['', '']] -> "[1:3:][2:4:]"
    slicing_code = []
    for i in range(max_depth):
        slicing_code.append("[")
        dim_info = [v[i] for v in balanced_splitted_inputs]
        slicing_code.append(":".join(dim_info))
        slicing_code.append("]")
    slicing_code = "".join(slicing_code)

    return f"{output_var[0]} = {input_var[0]}{slicing_code} if len({input_var[0]}.shape)>={max_depth} else {input_var[0]} # static slice inputs:{inputs}"


def dynamic_update_slice(input_var, output_var, params):  # TODO: unit test
    a, b, c = input_var
    #return f"{output_var[0]} = concatenate([ {b}[{c}:] , {a}]) # dynamic update slice"
    return f"{output_var[0]} = jax.lax.dynamic_update_slice({a}, {b}, ({c},))"

def scatter_add(input_var, output_var, params):  # TODO: unit test
    a, b, c = input_var
    return f"{output_var[0]} = add.ad({a}, {b}, {c})"

def or__(input_var, output_var, params):
    a, b = input_var
    return f"{output_var[0]} = {a} | {b}"

def and__(input_var, output_var, params):
    a, b = input_var
    return f"{output_var[0]} = {a} & {b}"

def bitcast_convert_type(input_var, output_var, params):
    dt=params["new_dtype"]
    return f"{output_var[0]} = jax.lax.bitcast_convert_type({input_var[0]}, new_dtype={dt})"

def erf_inv(input_var, output_var, params):
    return f"{output_var[0]} = jax.lax.erf_inv({input_var[0]})"

def stop_gradient(input_var, output_var, params):
    return f"{output_var[0]} = {input_var[0]} # stop grad"
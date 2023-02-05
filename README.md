# JaxDecompiler
Jax Decompiler

The JAX decompiler takes jaxpr code and produces a more readable Python code. Even if some information about the original function is lost (obfuscated code) like variable names being lost, it is an important tool for reverse-engineering.

Associated pr:
https://github.com/google/jax/issues/13398

## Installation

```bash
pip3 install JaxDecompiler
```

## Usage example

Given any jaxpr function, here "df", we want to generate the associated Python code.

```python
import jax

def f(x, smooth_rate):
    local_minimums = (1 - smooth_rate) * jax.numpy.cos(x)
    global_minimum = smooth_rate * x**2
    return global_minimum + local_minimums


df = jax.grad(f, (0,))
```

Function df is implemented with jaxpr code. You can display it with:

```python

from JaxDecompiler import decompiler

decompiler.display_wrapped_jaxpr(df, (1.0, 1.0))
```
returns:
```
===== HEADER =======
invars: [a, b]
outvars: [p]
constvars: []
===== CODE =======
{ lambda ; a:f32[] b:f32[]. let
    c:f32[] = sub 1.0 b
    d:f32[] = cos a
    e:f32[] = sin a
[...]
```

The below code decompiles it automatically. It generates the python function and its python code as text.

```python

from JaxDecompiler import decompiler

decompiled_df, python_code = decompiler.python_jaxpr_python(
    df, (1.0, 1.0), is_python_returned=True
)
```

Let's check df and decompiled_df behave the same:
```python
print("df: ", df(4.0, 0.99)) # ~7.927568
print("decompiled df: ", decompiled_df(4.0, 0.99))  # ~7.927568
```
They produce the same result in spite to be written in different languages!

Now Let's display what is inside decompiled_df:
```python
print(python_code)
```
Display:
```python
def f(a, b):
    c = 1.0 - b
    d = cos(a)
    e = sin(a)
    f = c * d
    g = a ** 2
    h = a ** 1
    i = 2.0 * h
    j = b * g
    _ = j + f
    k = c * 1.0
    l = -k
    m = l * e
    n = b * 1.0
    o = n * i
    p = m + o
    return p
```
Now, the user owns its derivative code and may easily refactor/edit it! This is a reverse-engineering tool, for example, we can now improving arithemtic stability, manually optimize the code, ...

Notice: python_jaxpr_python create out/ folder in the current directory.

## Next steps

There are the next steps:
* **More operators**. Today ~60 jaxpr operators are implemented ('add', 'mul', 'cos', ...). The exhaustive list of the implemented operators is in the file "primitive_mapping.py". This python file aims to map jaxpr operator (the name of the functions) into python code (string returned by the function).

* **Automatic refactoring**. There is room for improvement to make the automatically produced Python code easier to read/maintain. 
An automatic refactoring tool should be able to translate this low-level Python style into a more readable one for humans.

* **Automatic detection of useless codes**. In the example above, "j" variable is useless.

import unittest
import jax
from jax.numpy import *
import decompiler

K = decompiler.get_primitive_mapping()
DELTA=0.001

def python_jaxpr_python(python_f, moc_inputs):
    """Compilation followed by Decompilation allows to check if the decompilation is correct
    We assume Compilation is always correct here.
    Therefore, input program should be similar to the output program"""
    jaxpr_obj = jax.make_jaxpr(python_f)(*moc_inputs)

    python_lines = decompiler.decompiler(jaxpr_obj, K)

    f = decompiler.from_strings_to_callable(python_lines)
    return f


class MyTestCase(unittest.TestCase):
    def test_exp(self):
        f = lambda x: log(1 + exp(x))
        df = jax.grad(f)

        decompiled_df = python_jaxpr_python(df, (10.0,))
        y_produced = decompiled_df(10.0)
        y_expected = df(10.0)

        self.assertAlmostEqual(y_expected, y_produced)

    def test_trigonometry(self):
        def f(x):
            return cos(x), sin(x), tanh(x),\
                arctan(x), arccos(x), arcsin(x),\
                tanh(x)


        decompiler.info_jaxpr(f, (2.1,))

        #return
        decompiled_f=python_jaxpr_python(f, (0.1,))
        y_expected=f(0.2)
        y=decompiled_f(0.2)

        gap=sum(array(y_expected)-array(y))

        self.assertAlmostEqual(0.,gap,delta=DELTA)
if __name__ == "__main__":
    unittest.main()

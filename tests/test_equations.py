import unittest
import jax
from jax.numpy import *
import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):
    def test_neg(self):
        f = lambda x: -x

        decompiled_df = decompiler.python_jaxpr_python(f, (10.0,))
        y_produced = decompiled_df(10.0)
        y_expected = f(10.0)

        self.assertAlmostEqual(y_expected, y_produced)

    def test_log_exp(self):
        f = lambda x: log(1 + exp(x))
        df = jax.grad(f)

        decompiled_df = decompiler.python_jaxpr_python(df, (10.0,))
        y_produced = decompiled_df(10.0)
        y_expected = df(10.0)

        self.assertAlmostEqual(y_expected, y_produced)

    def test_pow_sqrt(self):
        def python_f(x):
            return sqrt(x), x**-0.1, x**2

        decompiled_df = decompiler.python_jaxpr_python(python_f, (1.0,))
        y = decompiled_df(2.0)
        y_expected = python_f(2.0)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_trigonometry(self):
        def f(x):
            return cos(x), sin(x), tanh(x), arctan(x), arccos(x), arcsin(x), tan(x)

        # return
        decompiled_f = decompiler.python_jaxpr_python(f, (0.1,))
        y_expected = f(0.2)
        y = decompiled_f(0.2)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_type(self):
        def f(x):
            xint = jax.numpy.array(x, dtype=int32)
            xfloatint = array(xint, dtype=float32)
            return xfloatint

        # return
        decompiled_f = decompiler.python_jaxpr_python(f, (0.1,))
        y_expected = f(0.2)
        y = decompiled_f(0.2)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

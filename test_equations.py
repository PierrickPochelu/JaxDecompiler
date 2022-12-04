import unittest
import jax
from jax.numpy import *
import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):
    def test_exp(self):
        f = lambda x: log(1 + exp(x))
        df = jax.grad(f)

        decompiled_df = decompiler.from_wrapped_jaxpr_to_python(df, (10.0,))
        y_produced = decompiled_df(10.0)
        y_expected = df(10.0)

        self.assertAlmostEqual(y_expected, y_produced)

    def test_trigonometry(self):
        def f(x):
            return cos(x), sin(x), tanh(x), arctan(x), arccos(x), arcsin(x), tanh(x)

        decompiler.display_wrapped_jaxpr(f, (2.1,))

        # return
        decompiled_f = decompiler.from_wrapped_jaxpr_to_python(f, (0.1,))
        y_expected = f(0.2)
        y = decompiled_f(0.2)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

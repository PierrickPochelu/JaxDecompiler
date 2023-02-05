import unittest
import jax
from jax.numpy import *
from src.JaxDecompiler import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):
    def test_jit_2pmap_A(self):
        @jax.jit
        def python_f(x):
            x1 = x + 1
            return x1

        x = array([[1, 2, 3]], dtype=float32)

        y_expected = python_f(x)
        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))
        y = decompiled_f(x)
        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_jit_2pmap_B(self):
        @jax.jit
        def python_f(x):
            def one_input_two_output(x):
                b = x + 1
                c = b * 2
                return b, c

            def two_input_one_output(x1, x2):
                return cos(sin(x1) + x2)

            x1 = x + 1
            b, c = jax.pmap(one_input_two_output)(x1)
            y2 = jax.pmap(two_input_one_output)(b, c)
            y3 = y2**2
            return y3

        x = array([[1, 2, 3]], dtype=float32)

        y_expected = python_f(x)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)

        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

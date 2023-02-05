import unittest
import jax
from jax.numpy import *
from src.JaxDecompiler import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):
    def test_vmap_1line(self):
        def python_f(x):
            g = lambda element: element + 1
            y = jax.vmap(g)(x)
            return y

        x = array([1, 2, 3], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected = array([[2, 3, 4]], dtype=float32)

        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_pmap_1line(self):
        def python_f(x):
            g = lambda element: element + 1
            y = jax.pmap(g)(x)
            return y

        x = array([3], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected = python_f(x)

        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_pmap_2lines(self):
        def python_f(x):
            def g(x):
                a = x + 1
                b = a * 2
                return b

            y = jax.pmap(g)(x)
            return y

        x = array([[1, 2, 3]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y_expected = python_f(x)
        y = decompiled_f(x)

        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_pmap_nlines(self):
        def python_f(x):
            def g(x):
                a = sin(x)  # useless line to check if it a problem
                b = x + 1
                c = b * 2
                d = cos(c + x)  # useless line
                return c

            y = jax.pmap(g)(x)
            y2 = y - 1
            y3 = y2 * 2
            return y3

        x = array([[1, 2, 3]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y_expected = python_f(x)
        y = decompiled_f(x)

        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_2pmap(self):
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

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y_expected = python_f(x)
        y = decompiled_f(x)

        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

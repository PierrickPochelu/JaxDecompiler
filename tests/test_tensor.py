import unittest
import jax
from jax.numpy import *
import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):

    def test_add_tensors(self):
        def python_f(x):
            return x + array([[-1, 0, -1.1], [0, -1, 0], [0, 0, 0]], dtype=float32)

        x = array([[1, 0, 1.1], [0, 1, 0], [0, 0, 0]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))
        y = decompiled_f(x)

        gap = sum(y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_dot_tensors(self):
        def python_f(x, w):
            return dot(x, w)

        x = array([[1, 1, 0]], dtype=float32)
        w = array([[9, 8, 7, 6], [5, 4, 3, 2], [1, 0.9, 0.8, 0.7]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x, w))

        y = decompiled_f(x, w)
        y_expected = array([[9 + 8 + 7 + 6, 5 + 4 + 3 + 2, 0, 0]], dtype=float32)

        gap = sum(y_expected - y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_min_max(self):
        def python_f(x):
            return argmin(x), min(x), max(x), argmax(x)

        x = array([[1, 2, 3, 4, 0]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected = python_f(x)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_avg_mean(self):
        def python_f(x):
            return average(x), mean(x)  # TODO average can contain weights

        x = array([[1, 2, 3, 4, 0]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected = python_f(x)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_std_var(self):
        def python_f(x):
            return std(x), var(x)

        x = array([[1, 2, 3, 4, 0]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected = python_f(x)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_med(self):
        def python_f(x):
            return median(x)

        x = array([[1, 2, 3, 4, 0]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected = python_f(x)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_perc_quant(self):
        def python_f(x):
            return percentile(x, 0.8), quantile(x, 0.6)

        x = array([[1, 2, 3, 4, 0]], dtype=float32)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected = python_f(x)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_conv_1D(self):
        def python_f(x):
            k = array([1., 1., 1.])
            return convolve(x, k, mode="same")

        x = array([0, 1, 0, 1, 0, 0, 0], dtype=float32)

        y_expected = python_f(x)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

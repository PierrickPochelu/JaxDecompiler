import unittest
import jax
from jax.numpy import *
from src.JaxDecompiler import decompiler
from src.JaxDecompiler.primitive_mapping import reduce_sum

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

    def test_outer_tensors(self):
        """
        invert dot tensor Y=X.W requires the derivative code contains the outer dot product
        """

        def predict(W, inputs):
            outputs = dot(inputs, W)
            return sum(outputs)

        X = array([1, 2, 3])
        W = array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=float)

        dpred = jax.grad(predict, argnums=(0,))

        decomp = decompiler.python_jaxpr_python(dpred, (W, X))

        y_exp = dpred(W, X)
        y = decomp(W, X)

        self.assertAlmostEqual(0.0, sum(array(y) - array(y_exp)), delta=DELTA)

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
            k = array([1.0, 1.0, 1.0])
            return convolve(x, k, mode="same")

        x = array([0, 1, 0, 1, 0, 0, 0], dtype=float32)

        y_expected = python_f(x)

        decompiled_f = decompiler.python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_dynamic_slicing(self):
        # this function will use dynamic_slicing jaxpr operator
        def f(X, w):
            return array([w]) ** 2 + X

        w = 0.1
        X = array([0.7])

        def circuit_to_opt(x, w):
            y_ = f(x, w)
            return y_[0]

        deriv_circuit_to_opt = jax.grad(circuit_to_opt, argnums=(-1,))

        deriv_circuit_to_opt_reconstructed = decompiler.python_jaxpr_python(
            deriv_circuit_to_opt, (X, w)
        )

        dw_expected = deriv_circuit_to_opt(X, w)
        dw = deriv_circuit_to_opt_reconstructed(X, w)

        gap = sum(array(dw_expected) - array(dw))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_reduce_sum(self):
        def f(X):
            # return sum(X,axis=axis) #TODO checking: , mean(X, axis=axis)  min(X, axis=axis), max(X, axis=axis)
            return sum(X, axis=(0,)), sum(X, axis=(1,)), sum(X, axis=(0, 1))

        X = array([[0.1], [0.2]])
        f_reconstructed = decompiler.python_jaxpr_python(f, (X,))

        potential_shapes = [(1, 2), (2, 1), (1, 1, 2), (2, 1, 1)]
        for X_shape in potential_shapes:
            X = X.reshape(X_shape)
            y_expected_list = f_reconstructed(X)
            y_list = f(X)

            for y_expected, y in zip(y_expected_list, y_list):
                # test same dims
                self.assertAlmostEqual(
                    len(array(y_expected).shape), len(array(y).shape), delta=DELTA
                )

                # test same shape
                self.assertAlmostEqual(
                    array(y_expected).shape, array(y).shape, delta=DELTA
                )

                # test sum values
                gap = sum(array(y_expected) - array(y))
                self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

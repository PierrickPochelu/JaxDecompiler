import unittest
import jax
from jax.numpy import *
from src.JaxDecompiler import decompiler
from jax.experimental import sparse

DELTA = 0.001

M = jax.numpy.array([[0.0, 1.0, 0.0, 2.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]])
V = jax.numpy.array([1.0, 0.0, 0.0, 0.0])


class MyTestCase(unittest.TestCase):
    def test_sparse_tensors(self):
        def python_f(M):
            return sparse.COO.fromdense(M, nse=4)

        decompiled_f = decompiler.jaxpr2python(python_f, M)
        y = decompiled_f(M)
        y_expected = python_f(M)

        gap = sum(array(y_expected.data) - array(y[0]))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

        gap = sum(array(y_expected.row) - array(y[1]))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

        gap = sum(array(y_expected.col) - array(y[2]))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_sparse_dot(self):
        def python_f(M, V):
            a = sparse.COO.fromdense(M, nse=4)
            return a @ V

        decompiled_f = decompiler.jaxpr2python(python_f, M, V)

        y = decompiled_f(M, V)
        y_expected = python_f(M, V)

        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

import unittest
import jax
from jax.numpy import *
import decompiler


K = decompiler.get_primitive_mapping()
DELTA = 0.001


def python_jaxpr_python(python_f, moc_inputs):
    """Compilation followed by Decompilation allows to check if the decompilation is correct
    We assume Compilation is always correct here.
    Therefore, input program should be similar to the output program"""
    jaxpr_obj = jax.make_jaxpr(python_f)(*moc_inputs)

    python_lines = decompiler.decompiler(jaxpr_obj, K)

    f = decompiler.from_strings_to_callable(python_lines)
    return f


class MyTestCase(unittest.TestCase):
    def test_add_tensors(self):
        def python_f(x):
            return x + array(
                [[-1, 0, -1.1], [0, -1, 0], [0, 0, 0]], dtype=float32
            )

        x = array([[1, 0, 1.1], [0, 1, 0], [0, 0, 0]], dtype=float32)

        decompiled_f = python_jaxpr_python(python_f, (x,))
        y = decompiled_f(x)

        gap = sum(y)
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_dot_tensors(self):
        def python_f(x, w):
            return dot(x,w)


        x = array([[1, 1, 0]], dtype=float32)
        w = array([[9, 8, 7, 6],
                   [5, 4, 3, 2],
                   [1, 0.9, 0.8, 0.7]], dtype=float32)

        decompiler.info_jaxpr(python_f,(x,w))

        decompiled_f = python_jaxpr_python(python_f, (x,w))

        y = decompiled_f(x,w)
        y_expected=array([[9+8+7+6, 5+4+3+2, 0, 0]], dtype=float32)

        print(y)
        gap = sum(y_expected-y)
        self.assertAlmostEqual(0., gap, delta=DELTA)

if __name__ == "__main__":
    unittest.main()

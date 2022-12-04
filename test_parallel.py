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
    """
    def test_vmap_1line(self):
        def python_f(x):
            g=lambda element:element+1
            y=jax.vmap(g)(x)
            return y

        x = array([1, 2, 3], dtype=float32)

        #decompiler.info_jaxpr(python_f,(x,))

        decompiled_f = python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected=array([[2, 3, 4]], dtype=float32)

        gap = sum(y_expected-y)
        self.assertAlmostEqual(0., gap, delta=DELTA)
    """
    def test_pmap_1line(self):
        def python_f(x):
            g=lambda element:element+1
            y=jax.pmap(g)(x)
            return y

        x = array([3], dtype=float32)

        #decompiler.info_jaxpr(python_f,(x,))

        decompiled_f = python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected=python_f(x)

        gap = sum(y_expected-y)
        self.assertAlmostEqual(0., gap, delta=DELTA)

    """
    def test_pmap_2lines(self):
        def python_f(x):
            def g(x):
                a=x+1
                b=a*2
                return b
            y=jax.pmap(g)(x)
            return y

        x = array([1, 2, 3], dtype=float32)

        #decompiler.info_jaxpr(python_f,(x,))

        decompiled_f = python_jaxpr_python(python_f, (x,))

        y = decompiled_f(x)
        y_expected=array([[2, 3, 4]], dtype=float32)

        gap = sum(y_expected-y)
        self.assertAlmostEqual(0., gap, delta=DELTA)
    """



if __name__ == "__main__":
    unittest.main()

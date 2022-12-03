import unittest
import jax
from jax import numpy as jnp
import decompiler


K=decompiler.get_primitive_mapping()
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


    def test_add_tensors(self):
        def python_f(x):
            return x+jnp.array([[-1,0,-1.1],[0,-1,0],[0,0,0]],dtype=jnp.float32)
        x=jnp.array([[1,0,1.1],[0,1,0],[0,0,0]],dtype=jnp.float32)

        decompiler._examine_jaxpr(python_f, x)

        decompiled_f=python_jaxpr_python(python_f, (x,))
        y=decompiled_f(x)

        gap=jnp.sum(y)
        self.assertAlmostEqual(0.,gap,delta=DELTA)


if __name__ == '__main__':
    unittest.main()

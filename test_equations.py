import unittest
import jax
import decompiler

K=decompiler.get_primitive_mapping()

def python_jaxpr_python(python_f, moc_inputs):
    """Compilation followed by Decompilation allows to check if the decompilation is correct
    We assume Compilation is always correct here.
    Therefore, input program should be similar to the output program"""
    jaxpr_obj = jax.make_jaxpr(python_f)(*moc_inputs)

    python_lines = decompiler.decompiler(jaxpr_obj, K)

    f = decompiler.from_strings_to_callable(python_lines)
    return f

class MyTestCase(unittest.TestCase):

    def test_eq(self):
        from jax import numpy as jnp
        f = lambda x: jnp.log(1 + jnp.exp(x))
        df=jax.grad(f)

        decompiled_df = python_jaxpr_python(df, (10.,))
        y_produced = decompiled_df(10.)
        y_expected = df(10.)

        self.assertAlmostEqual(y_expected, y_produced)


if __name__ == '__main__':
    unittest.main()

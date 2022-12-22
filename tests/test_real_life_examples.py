import unittest
import jax
from jax.numpy import *
import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):
    def test_lun4m_equation(self):
        import jax.numpy as jnp
        from jax import value_and_grad
        import decompiler

        def fun(ra, rb, r0, k):
            rab = ra - rb
            rab = jnp.sqrt(jnp.dot(rab, rab))
            return 0.5 * k * (rab - r0) ** 2

        df = value_and_grad(fun)
        ra = jnp.ones(3)
        rb = 2 * ra

        x = (ra, rb, 0.5, 20.0)

        y_expected = df(*x)

        decomp, pycode = decompiler.python_jaxpr_python(df, x, is_python_returned=True)

        y = decomp(*x)

        gap = sum(array(y_expected[0]) - array(y[0]))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

        gap = sum(array(y_expected[1]) - array(y[1]))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_lun4m_equation_jit(self):
        import jax.numpy as jnp
        from jax import value_and_grad
        import decompiler

        @jax.jit
        def fun(ra, rb, r0, k):
            rab = ra - rb
            rab = jnp.sqrt(jnp.dot(rab, rab))
            return 0.5 * k * (rab - r0) ** 2

        df = value_and_grad(fun)

        ra = jnp.ones(3)
        rb = 2 * ra

        x = (ra, rb, 0.5, 20.0)

        df = jax.jit(df)
        y_expected = df(*x)

        decomp, pycode = decompiler.python_jaxpr_python(df, x, is_python_returned=True)

        y = decomp(*x)

        gap = sum(array(y_expected[0]) - array(y[0]))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

        gap = sum(array(y_expected[1]) - array(y[1]))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

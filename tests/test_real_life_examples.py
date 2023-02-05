import unittest
import jax
from jax.numpy import *
from src.JaxDecompiler import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):

    def test_lun4m_equation(self):
        import jax.numpy as jnp
        from jax import value_and_grad

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

    #TODO correct below code
    """
    def test_MZI(self):
        def MZI(X, teta):
            R = array([
                [cos(teta), -sin(teta)],
                [sin(teta), cos(teta)]
            ])
            #return array([teta]) + X  # OK
            #return array([cos(0.222), -sin(0.222)]) + X # OK
            #return array([cos(teta), -sin(teta)])+X # ok
            return dot(R, X)

        weights=0.07

        X=array([0.1, .9])

        def circuit_to_opt(x,w):
            y_=MZI(x,w)
            return y_[0]
            #return array(y_[0])
            #return mean((Y-y_)**2)

        deriv_circuit_to_opt = jax.grad(circuit_to_opt, argnums=(-1,))
        dw_expected = deriv_circuit_to_opt(X, weights)  # 0.9938458...

        deriv_circuit_to_opt_reconstructed = decompiler.python_jaxpr_python(deriv_circuit_to_opt, (X, weights))
        dw=deriv_circuit_to_opt_reconstructed(X,weights)

        gap = sum(array(dw_expected) - array(dw))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)
    """
if __name__ == "__main__":
    unittest.main()

import unittest
import jax
from jax.numpy import *
from src.JaxDecompiler import decompiler

DELTA = 0.001


class MyTestCase(unittest.TestCase):
    def test_neg(self):
        f = lambda x: -x

        decompiled_df = decompiler.python_jaxpr_python(f, (10.0,))
        y_produced = decompiled_df(10.0)
        y_expected = f(10.0)

        self.assertAlmostEqual(y_expected, y_produced)

    def test_log_exp(self):
        f = lambda x: log(1 + exp(x))
        df = jax.grad(f)

        decompiled_df = decompiler.python_jaxpr_python(df, (10.0,))
        y_produced = decompiled_df(10.0)
        y_expected = df(10.0)

        self.assertAlmostEqual(y_expected, y_produced)

    def test_pow_sqrt(self):
        def python_f(x):
            return sqrt(x), x**-0.1, x**2

        decompiled_df = decompiler.python_jaxpr_python(python_f, (1.0,))
        y = decompiled_df(2.0)
        y_expected = python_f(2.0)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_trigonometry(self):
        def f(x):
            return cos(x), sin(x), tanh(x), arctan(x), arccos(x), arcsin(x), tan(x)

        # return
        decompiled_f = decompiler.python_jaxpr_python(f, (0.1,))
        y_expected = f(0.2)
        y = decompiled_f(0.2)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_type(self):
        def f(x):
            xint = jax.numpy.array(x, dtype=int32)
            xfloatint = array(xint, dtype=float32)
            return xfloatint

        # return
        decompiled_f = decompiler.python_jaxpr_python(f, (0.1,))
        y_expected = f(0.2)
        y = decompiled_f(0.2)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_non_derivable(self):
        def f(x):
            return (
                max(array((x, 0))),
                min(array((x, 0))),
                round(x),
                ceil(x),
                floor(x),
                abs(x),
            )

        # return
        decompiled_f = decompiler.python_jaxpr_python(f, (0.1,))
        y_expected = f(0.2)
        y = decompiled_f(0.2)

        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_gaussian_noise(self):
        def f(x):
            random_seed = 1
            key = jax.random.PRNGKey(random_seed)

            std = 0.0001
            gaussian_noise = (
                jax.random.normal(shape=(1,), key=key, dtype=float32)[0] * std
            )
            return x + gaussian_noise

        # return
        decompiled_f = decompiler.python_jaxpr_python(f, (0.1,))
        y_expected = f(0.2)
        y = decompiled_f(0.2)
        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_uniform_noise(self):
        def f(x):
            random_seed = 1
            key = jax.random.PRNGKey(random_seed)

            interval = 0.0001
            uniform_noise = jax.random.uniform(
                shape=(1,),
                key=key,
                dtype=float32,
                minval=-1 * interval,
                maxval=interval,
            )[0]

            return x + uniform_noise

        decompiled_f = decompiler.python_jaxpr_python(f, (0.1,))

        y_expected = f(0.2)
        y = decompiled_f(0.2)
        gap = sum(array(y_expected) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_no_gradient(self):
        def forward(x):
            return round(x)

        decompiled_f = decompiler.python_jaxpr_python(forward, (0.1,))

        # FORWARD
        y_expected = forward(0.2)
        y = decompiled_f(0.2)
        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

        # NAIVE BACKWARD
        backward = jax.grad(forward)
        decompiled_b = decompiler.python_jaxpr_python(backward, (0.1,))
        y_expected = backward(0.2)
        y = decompiled_b(0.2)
        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

        # CORRECTED BACKWARD USING NO_GRADS
        def no_grad_f(x):
            zero = x - jax.lax.stop_gradient(x)
            return zero + jax.lax.stop_gradient(forward(x))

        no_grad_b = jax.grad(no_grad_f)
        decompiled_no_grad_b = decompiler.python_jaxpr_python(no_grad_b, (0.1,))
        y_expected = no_grad_b(0.2)
        y = decompiled_no_grad_b(0.2)
        gap = sum(array(y_expected) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

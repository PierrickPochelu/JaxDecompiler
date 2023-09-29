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

    def test_linear_regression(self):
        def predict(W, inputs):
            outputs = dot(inputs, W)
            return outputs

        def loss(params, inputs, targets):
            preds = predict(params, inputs)
            return mean((preds - targets) ** 2)

        dloss = jax.grad(loss, argnums=(0,))

        X = array([1, 0, -1], dtype=float)
        Y = array([1, 0, -1], dtype=float)
        W = array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=float)

        for i in range(4):
            first_mse = loss(W, X, Y)
            delta = dloss(W, X, Y)
            W = W - 0.01 * delta[0]  # weight

        #########

        from src.JaxDecompiler import decompiler

        decomp = decompiler.python_jaxpr_python(dloss, (W, X, Y))
        y_exp = dloss(W, X, Y)
        y = decomp(W, X, Y)

        gap = sum(array(y_exp) - array(y))

        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_chain_rule(self):
        def predict(params, inputs):
            for W in params:
                outputs = dot(inputs, W)
                inputs = outputs
            return inputs

        def loss(params, inputs, targets):
            preds = predict(params, inputs)
            return sum((preds - targets) ** 2)

        nb_features, nb_layers = 3, 2
        X = array([0.1, 0.2, 0.3])
        Y = array([0.1, 0.2, 0.3])
        params = [
            arange(0, 1, 1.0 / (3 * 3)).reshape((nb_features, nb_features))
            for i in range(nb_layers)
        ]

        dloss = jax.grad(loss, argnums=(0,))

        from src.JaxDecompiler import decompiler

        decomp = decompiler.python_jaxpr_python(dloss, (params, X, Y))

        y_exp = dloss(params, X, Y)
        y = decomp(params[0], params[1], X, Y)
        self.assertAlmostEqual(0.0, sum(array(y) - array(y_exp)), delta=DELTA)

    def test_MLP(self):
        def predict(params, inputs):
            for W, b in params:
                outputs = dot(inputs, W) + b
                inputs = tanh(outputs)
            return inputs

        def loss(params, inputs, targets):
            preds = predict(params, inputs)
            return mean((preds - targets) ** 2)

        def model(nb_layers, nb_features):
            params = []
            key = jax.random.PRNGKey(42)

            def layer_params(n_in, n_out):
                W = jax.random.normal(shape=(n_in, n_out), key=key, dtype=float) * (
                    2.0 / (n_in + n_out)
                )
                b = jax.random.normal(shape=(n_out,), key=key, dtype=float)
                return [W, b]

            for i in range(nb_layers):
                params.append(layer_params(nb_features, nb_features))
            return params

        nb_features, nb_layers = 3, 2
        X = array([0.1, 0.2, 0.3], dtype=float)
        Y = array([0.1, 0.2, 0.3], dtype=float)

        import copy

        params = model(nb_layers, nb_features)
        params2 = copy.deepcopy(params)

        dloss = jax.grad(loss, argnums=(0,))
        from src.JaxDecompiler import decompiler

        decomp = decompiler.python_jaxpr_python(
            dloss,
            (params, X, Y),
        )

        # TRAINING WITH dloss
        history_loss_expected = []
        for i in range(4):
            mse = loss(params, X, Y)
            delta = dloss(params, X, Y)
            for i in range(len(params)):
                params[i][0] = params[i][0] - 0.5 * delta[0][i][0]  # weight
                params[i][1] = params[i][1] - 0.5 * delta[0][i][1]  # bias
            history_loss_expected.append(mse)
        # TRAINING WITH decomp
        history_loss = []

        def flatten(unflat):
            from itertools import chain

            unflat = list(chain.from_iterable(unflat))
            return unflat

        for i in range(4):
            mse = loss(params2, X, Y)
            delta = decomp(*flatten(params2), X, Y)
            j = 0
            for i in range(len(params2)):
                params2[i][0] = params2[i][0] - 0.5 * delta[j]  # weight
                j += 1
                params2[i][1] = params2[i][1] - 0.5 * delta[j]  # bias
                j += 1
            history_loss.append(mse)

        gap = sum(array(history_loss_expected) - array(history_loss))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_MZI(self):
        def MZI(teta, X):
            R = array([[cos(teta), -sin(teta)], [sin(teta), cos(teta)]])
            return dot(R, X)

        weights = 0.07
        X = array([0.1, 0.9])

        MZI_reconstructed = decompiler.python_jaxpr_python(MZI, (weights, X))

        y_exp = MZI(weights, X)
        y = MZI_reconstructed(weights, X)

        gap = sum(array(y_exp) - array(y))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)

    def test_gradient_MZI(self):
        def MZI(teta, X):
            R = array([[cos(teta), -sin(teta)], [sin(teta), cos(teta)]])
            return dot(X, R)

        weights = 0.07

        X = array([0.1, 0.9])
        Y = array([0.9, 0.1])

        def circuit_to_opt(w, x, y):
            return sum((MZI(w, x) - y) ** 2)

        deriv_circuit_to_opt = jax.grad(circuit_to_opt, argnums=(0,))

        deriv_circuit_to_opt_reconstructed = decompiler.python_jaxpr_python(
            deriv_circuit_to_opt, (weights, X, Y)
        )

        dw_expected = deriv_circuit_to_opt(weights, X, Y)
        dw = deriv_circuit_to_opt_reconstructed(weights, X, Y)

        gap = sum(array(dw_expected) - array(dw))
        self.assertAlmostEqual(0.0, gap, delta=DELTA)


if __name__ == "__main__":
    unittest.main()

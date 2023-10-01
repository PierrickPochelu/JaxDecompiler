from JaxDecompiler import decompiler
from jax import *
from jax.numpy import *
import copy
import sys
import time

EPOCHS = int(sys.argv[1])
LEARNING_RATE = 0.5
NB_FEATURES, NB_LAYERS = int(sys.argv[2]), int(sys.argv[3])

print(
    f"Bench MLP: with epochs={EPOCHS}, NB_FEATURES={NB_FEATURES}, NB_LAYERS={NB_LAYERS}"
)


def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        print(func.__name__, " time: ", round(time.time() - started_at, 6))
        return result

    return wrap


def flatten(unflat):
    from itertools import chain

    flat = list(chain.from_iterable(unflat))
    return flat


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


dloss = jax.grad(loss, argnums=(0,))

# DATA
X = arange(0, 1, 1.0 / NB_FEATURES)
Y = arange(0, 1, 1.0 / NB_FEATURES)


# init
params = model(NB_LAYERS, NB_FEATURES)
params2 = copy.deepcopy(params)

# create the second neural network with the decompiler
reconstructed_loss = decompiler.jaxpr2python(loss, params, X, Y)
reconstructed_dloss = decompiler.jaxpr2python(dloss, params, X, Y)

# TRAINING WITH dloss
@profile
def T1():  # Training performs backward and updates
    for i in range(EPOCHS):
        delta = dloss(params, X, Y)
        for i in range(len(params)):
            params[i][0] = params[i][0] - LEARNING_RATE * delta[0][i][0]  # weight
            params[i][1] = params[i][1] - LEARNING_RATE * delta[0][i][1]  # bias


@profile
def E1():  # Evaluation performs forward
    for i in range(EPOCHS):
        mse = loss(params, X, Y)


print("Jaxpr:")
T1()
E1()
print("jitted Jaxpr")
dloss = jax.jit(dloss)
loss = jax.jit(loss)
T1()
E1()


# TRAINING OF THE DECOMPILED NEURAL NETWORK
@profile
def T2():
    for i in range(EPOCHS):
        delta = reconstructed_dloss(*flatten(params2), X, Y)
        j = 0
        for i in range(len(params2)):
            params2[i][0] = params2[i][0] - LEARNING_RATE * delta[j]  # weight
            j += 1
            params2[i][1] = params2[i][1] - LEARNING_RATE * delta[j]  # bias
            j += 1


@profile
def E2():
    for i in range(EPOCHS):
        mse = reconstructed_loss(*flatten(params2), X, Y)


print("JaxDecompiler:")
T2()
E2()
print("jitted JaxDecompiler")
reconstructed_dloss = jax.jit(reconstructed_dloss)
reconstructed_loss = jax.jit(reconstructed_loss)
T2()
E2()

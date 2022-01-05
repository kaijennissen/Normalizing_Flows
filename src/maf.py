from typing import Tuple

import distrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from distrax._src.bijectors.bijector import Array
from jax import random
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class AffineScalar(distrax.Bijector):
    def __init__(self, scale, shift):

        if hasattr(scale, "shape"):
            D = scale.ndim
            assert scale.shape == shift.shape
        else:
            D = 0

        super().__init__(event_ndims_in=D, event_ndims_out=D)
        self.bijector = distrax.Chain([tfb.Shift(shift), tfb.Scale(scale)])

    def forward(self, x: Array) -> Array:
        return self.bijector.forward(x)

    def inverse(self, y: Array) -> Array:
        return self.bijector.inverse(y)

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        return self.bijector.forward_and_log_det(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        return self.bijector.inverse_and_log_det(y)


bij = AffineScalar(scale=jnp.array([0.2, 0.5]), shift=jnp.array([1.0, -1]))
bij.forward(jnp.array([[1.0, -2], [2, -4], [3, -6]]))


# MAF with
# - Transformer: Affine
# - Consitioner: Masking
def conditioner(z: Array) -> Array:
    assert z.ndim == 2
    B, D = z.shape
    norm = jnp.sqrt(jnp.cumsum(jnp.square(z), axis=-1))
    beta = jnp.concatenate([jnp.zeros((B, 1)), norm[..., :-1]], axis=-1) / 10
    alpha = jnp.exp(-beta / D)
    return alpha, beta


def transformer(z: Array) -> Array:
    alpha, beta = conditioner(z)
    return alpha * z + beta


mu = jnp.array([-1, -1])
sigma = jnp.ones(2)
dist_base = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
key = random.PRNGKey(52)

z = dist_base.sample(seed=key, sample_shape=(1000000,))

N = 32
cols = rows = int(np.ceil(np.sqrt(N / 2)))

fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))

zz = z
l = 1  # frew of switching the conditioner
for j in range(N):
    if j % l == 0:
        zz = transformer(jnp.concatenate([zz[..., 1:], zz[..., :1]], axis=1))
    else:
        zz = transformer(zz)
    if j % 2 == 0:
        i = j / 2
        r = int(i // rows)
        c = int(i % cols)
        ax = axes[r, c]
        ax.set_title(f"k={j}")
        ax.hist2d(zz[:, 0], zz[:, 1], bins=100)

fig.tight_layout()
plt.savefig("./plots/MAF.jpg")

# TODO: replace conditioner with network

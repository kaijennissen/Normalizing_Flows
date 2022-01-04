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


def broadcast_shape(shp1, shp2):
    """Broadcast the shape of those arrays

    Parameters
    ----------
    shp1 : tuple
        shape of array 1
    shp2 : tuple
        shape of array 2

    Returns
    -------
    tuple
        shape resulting from broadcasting two arrays using numpy rules

    Raises
    ------
    ValueError
        Arrays cannot be broadcasted
    """
    try:
        return np.broadcast(np.empty(shp1), np.empty(shp2)).shape
    except ValueError:
        raise ValueError(
            "Arrays cannot be broadcasted - %s and %s " % (str(shp1), str(shp2))
        )


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


# Masking - using haiku?


x1 = jnp.linspace(-5, 8, 500)
x2 = jnp.linspace(-4, 4, 500)
lpx2 = tfd.Normal(loc=0, scale=1.0).log_prob(x2)
lpx1 = tfd.Normal(loc=x2[:, jnp.newaxis] ** 2, scale=1.0).log_prob(x1)

Z = jnp.exp(lpx1 + lpx2[:, jnp.newaxis])
X1, X2 = jnp.meshgrid(x1, x2)

cm = plt.cm.get_cmap("viridis")
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X1, X2, Z, rstride=8, cstride=8, alpha=0.9, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="z", offset=-0.3, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="x", offset=-6, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="y", offset=-6, cmap=cm)
ax.set_zlim(-0.3, np.max(Z) * 1.4)
ax.set_xlabel(r"$x_{2}$")
ax.set_ylabel(r"$x_{1}$")
ax.set_zlabel(r"$f(x_{1},x_{2})$")
ax.view_init(elev=20, azim=45)
ax.grid(False)
fig.tight_layout()
plt.savefig("./plots/MVN_curved.jpg", dpi=600)
plt.close()

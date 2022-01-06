import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as I
import seaborn as sns
from jax import random
from jax.nn import sigmoid, softmax, softplus
from tensorflow_probability.substrates import jax as tfp

from BernsteinBijector import BernsteinBijector, constrain_thetas

tfd = tfp.distributions
tfb = tfp.bijectors

# Rotaton Flow
class OrthogonalProjection2D(distrax.Bijector):
    def __init__(self, theta):
        super().__init__(event_ndims_in=1, event_ndims_out=1)
        self.thetas = theta
        self.sin_theta = jnp.sin(theta)
        self.cos_theta = jnp.cos(theta)
        self.R = jnp.array(
            [[self.cos_theta, -self.sin_theta], [self.sin_theta, self.cos_theta]]
        ).T

    def forward(self, x):
        return jnp.matmul(x, self.R)

    def inverse(self, x):
        return jnp.matmul(x, self.R.T)

    def forward_and_log_det(self, x):
        y = self.forward(x)
        logdet = 1
        return y, logdet

    def inverse_and_log_det(self, x):
        y = self.inverse(x)
        logdet = 1
        return y, logdet


keys = random.split(random.PRNGKey(12), 3)

a = jnp.exp(jax.random.normal(keys[0]))
b = jax.random.normal(keys[1])
theta = jax.random.uniform(keys[2], minval=0.0, maxval=2.0 * jnp.pi)
# a = 1.5
# b = .2
# theta = np.pi / 3

T1 = distrax.Block(tfb.Scale(a), ndims=1)
T2 = distrax.Block(tfb.Shift(b), ndims=1)
T3 = OrthogonalProjection2D(theta=theta)
T4 = distrax.Block(tfb.Log(), ndims=1)
bij = distrax.Chain([T4, T3, T2, T1])


mu = jnp.zeros(2)
sigma = jnp.ones(2)
dist_base = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
dist_transformed = distrax.Transformed(dist_base, bij)

x1 = np.linspace(-3, 3, 2000)
x2 = np.linspace(-3, 3, 2000)
X1, X2 = np.meshgrid(x1, x2)
X = np.stack([X1, X2], axis=-1)
Y = np.exp(dist_base.log_prob(X))
Z = np.exp(dist_transformed.log_prob(X))

cm = plt.cm.get_cmap("viridis")  # .reversed()
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection="3d")
surf = ax.plot_surface(X1, X2, Y, rstride=8, cstride=8, alpha=0.9, cmap=cm)
cset = ax.contourf(X1, X2, Y, zdir="z", offset=-0.3, cmap=cm)
cset = ax.contourf(X1, X2, Y, zdir="x", offset=-4, cmap=cm)
cset = ax.contourf(X1, X2, Y, zdir="y", offset=-4, cmap=cm)
ax.set_zlim(-0.3, np.max(Y) * 1.4)
ax.set_xlabel(r"$x_{2}$")
ax.set_ylabel(r"$x_{1}$")
ax.set_zlabel(r"$f(x_{1},x_{2})$")
ax.view_init(elev=20, azim=45)
ax.grid(False)
ax = fig.add_subplot(122, projection="3d")
surf = ax.plot_surface(X1, X2, Z, rstride=8, cstride=8, alpha=0.9, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="z", offset=-0.3, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="x", offset=-4, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="y", offset=4.5, cmap=cm)
ax.set_zlim(-0.3, np.max(Z) * 1.35)
ax.set_xlabel(r"$x_{1}$")
ax.set_ylabel(r"$x_{2}$")
ax.set_zlabel(r"$g(x_{1},x_{2})$")

ax.view_init(elev=20, azim=-45)
ax.grid(False)
fig.tight_layout(pad=3, w_pad=0.1)
plt.savefig("./plots/MVN_3D_rotation.jpg", dpi=600)
plt.close()

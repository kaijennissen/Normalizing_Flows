import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax.nn import sigmoid, softmax, softplus
from jax.random import PRNGKey
from tensorflow_probability.substrates import jax as tfp
import scipy.interpolate as I
from bernstein_flow.bijectors import BernsteinBijector
from bernstein_flow.distributions import BernsteinFlow

tfd = tfp.distributions
tfb = tfp.bijectors


def plot_chained_bijectors(flow):
    chained_bijectors = flow.bijector.bijector.bijectors
    base_dist = flow.distribution
    cols = len(chained_bijectors) + 1
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))

    n = 200

    z_samples = np.linspace(-3, 3, n).astype(np.float32)
    log_probs = base_dist.log_prob(z_samples)

    ax[0].plot(z_samples, np.exp(log_probs))

    zz = z_samples[..., None]
    ildj = 0.0
    for i, (a, b) in enumerate(zip(ax[1:], chained_bijectors)):
        # we need to use the inverse here since we are going from z->y!
        z = b.inverse(zz)
        ildj += b.forward_log_det_jacobian(z, 1)
        # print(z.shape, zz.shape, ildj.shape)
        a.plot(z, np.exp(log_probs + ildj))
        a.set_title(b.name.replace("_", " "))
        a.set_xlabel(f"$z_{i}$")
        a.set_ylabel(f"$p(z_{i+1})$")
        zz = z
    fig.tight_layout()


def get_beta_dists(order):
    alpha = [x for x in range(1, order + 1)]
    beta = alpha[::-1]
    return tfd.Beta(alpha, beta)


def get_beta_dists_derivative(order):
    alpha = [x for x in range(1, order)]
    beta = alpha[::-1]
    return tfd.Beta(alpha, beta)


def bernstein_polynom_jacobean(theta):
    theta_shape = theta.shape
    order = theta_shape[-1]

    beta_dist_h_dash = get_beta_dists_derivative(order)

    def b_poly_dash(y):
        by = beta_dist_h_dash.prob(y)
        dtheta = theta[..., 1:] - theta[..., 0:-1]
        dz_dy = jnp.reduce_sum(by * dtheta, axis=-1)
        return dz_dy

    return b_poly_dash


def get_bernstein_polynom(theta):
    theta_shape = theta.shape
    order = theta_shape[-1]
    batch_shape = theta_shape[:-1]
    beta_dists = get_beta_dists(order)

    def bernstein_poly(x):
        sample_shape = x.shape
        bx = beta_dists.prob(x[..., jnp.newaxis])
        z = jnp.mean(bx * theta, axis=-1)

        return z

    return bernstein_poly


def constrain_thetas(theta_unconstrained, fn=softplus):

    d = jnp.concatenate(
        (
            jnp.zeros_like(theta_unconstrained[..., :1]),
            theta_unconstrained[..., :1],
            fn(theta_unconstrained[..., 1:]) + 1e-4,
        ),
        axis=-1,
    )
    return jnp.cumsum(d[..., 1:], axis=-1)


class Bernstein_Bijector(distrax.Bijector):
    def __init__(self, thetas):
        super().__init__(event_ndims_in=0, event_ndims_out=0)
        self.thetas = thetas
        self._is_injective = True

    def _forward(self, x):
        bernstein_poly = get_bernstein_polynom(self.thetas)
        x = jnp.clip(x, 0.0, 1.0)
        f2 = bernstein_poly(x)
        return f2

    def _forward_log_det(self, x):
        bernstein_poly = get_bernstein_polynom(self.thetas)
        x = jnp.clip(x, 0.0, 1.0)
        f2 = bernstein_poly(x)
        return f2

    def inverse(self, x):
        n_points = 200
        clip = 1e-7
        x_fit = jnp.linspace(clip, 1 - clip, n_points)
        y_fit = self._forward(x_fit)
        yp = jnp.interp(x, y_fit, x_fit)
        return yp

    def forward_and_log_det(self, x):
        y = self._forward(x)
        logdet = self._forward_log_det(x)
        return y, logdet

    def inverse_and_log_det(self, y):
        # Optional. Can be omitted if inverse methods are not needed.
        x = ...
        logdet = ...
        return x, logdet


a = jnp.array([6.0], dtype=np.float32)
b = jnp.array([-3.0], dtype=np.float32)
theta = jnp.array([-2.0, 2.0, 3.0, -7.0, -7.0, -7.0, -7.0, -7.0, 7.0], dtype=np.float32)
alpha = jnp.array([0.2], dtype=np.float32)
beta = jnp.array([-2.50], dtype=np.float32)

T1 = distrax.Chain([tfb.Scale(-1), tfb.Shift(2)])
T2 = distrax.Inverse(distrax.Tanh())
T3 = distrax.Chain([tfb.Scale(1), tfb.Shift(0)])

bij = distrax.Inverse(distrax.Chain([T3, T2, T1]))
z = np.random.normal(size=100000)
y = bij.forward(z)
sns.distplot(y)
plt.show()


n = 200
cols = 4
z_samples = np.linspace(-3, 3, n).astype(np.float32)
base_dist = distrax.Normal(0, 1)
fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4), sharey=True)

ildj = 0.0
log_probs = base_dist.log_prob(z_samples)
ax[0].plot(z_samples, np.exp(log_probs))
zz = z_samples

# f3
z = T3.inverse(zz)
ildj += T3.forward_log_det_jacobian(z)
ax[1].plot(z, np.exp(log_probs - ildj))
zz = z

# f2
z = T2.inverse(zz)
ildj += T2.forward_log_det_jacobian(z)
ax[2].plot(z, np.exp(log_probs + ildj))
zz = z


# f1
z = T1.inverse(zz)
ildj += T1.forward_log_det_jacobian(z)
ax[3].plot(z, np.exp(log_probs + ildj))
zz = z
plt.show()
# f_
# z = f_.inverse(zz)
# log_probs = base_dist.log_prob(z)
# ildj += f_.forward_log_det_jacobian(z)
# ax[3].plot(z, np.exp(log_probs + ildj))

plt.show()

# #
# theta = jnp.array(
#     [
#         -1.0744555,
#         -0.6429366,
#         -0.44160688,
#         0.7950939,
#         1.9249767,
#         2.1408765,
#         2.4256434,
#         3.1641612,
#         3.3939004,
#     ]
# )
# f2 = Bernstein_Bijector(thetas=theta)
# yy = np.linspace(0, 1, 200, dtype=np.float32)
# zz = f2.forward(yy)
# zi = zz + 1e-15 * np.random.random(zz.shape)
# yyy = f2.inverse(zi)
# plt.figure(figsize=(8, 8))
# plt.plot(yy, zz, alpha=0.5)
# plt.plot(yyy, zi, alpha=0.5)
# plt.title("Jax-Version")
# plt.show()

# print(
#     f"The MSE of the interpolation in this example is {np.sum((np.squeeze(yyy)-yy)**2):.3e}$."
# )

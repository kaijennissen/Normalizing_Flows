import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as I
import seaborn as sns
from distrax._src.distributions.distribution import PRNGKey
from jax import random
from jax.nn import sigmoid, softmax, softplus
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def get_beta_dists(order):
    alpha = [x for x in range(1, order + 1)]
    beta = alpha[::-1]
    return tfd.Beta(alpha, beta)


def get_bernstein_poly(theta):
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


def get_beta_dists_derivative(order):
    alpha = [x for x in range(1, order)]
    beta = alpha[::-1]
    return tfd.Beta(alpha, beta)


def get_bernstein_poly_jac(theta):
    theta_shape = theta.shape
    order = theta_shape[-1]

    beta_dist_der = get_beta_dists_derivative(order)

    def bernstein_poly_jac(y):
        by = beta_dist_der.prob(y[..., jnp.newaxis])
        dtheta = theta[..., 1:] - theta[..., 0:-1]
        dz_dy = jnp.sum(by * dtheta, axis=-1)
        return dz_dy

    return bernstein_poly_jac


def constrain_thetas(theta_unconstrained, range_min=None, range_max=None, fn=softplus):

    d = jnp.concatenate(
        (
            jnp.zeros_like(theta_unconstrained[..., :1]),
            theta_unconstrained[..., :1],
            fn(theta_unconstrained[..., 1:]) + 1e-4,
        ),
        axis=-1,
    )

    return jnp.cumsum(d[..., 1:], axis=-1)


class BernsteinBijector(distrax.Bijector):
    """Initializes a Bernstein bijector."""

    def __init__(self, thetas):
        super().__init__(event_ndims_in=0)
        self.thetas = constrain_thetas(thetas)

    def forward(self, x):
        """Computes y = f(x)."""
        bernstein_poly = get_bernstein_poly(self.thetas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)
        return bernstein_poly(x)

    def forward_log_det_jacobian(self, x):
        """Computes log|det J(f)(x)|."""
        bernstein_poly = get_bernstein_poly_jac(self.thetas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)
        return jnp.log(bernstein_poly(x))

    def inverse(self, y):
        """Computes x = f^{-1}(y)."""
        event_shape = self.thetas.shape[:-1]
        n_points = 200
        clip = 1e-7
        x_fit = jnp.linspace(clip, 1 - clip, n_points)[..., np.newaxis] * jnp.ones(
            (1,) + event_shape
        )
        y_fit = self.forward(x_fit)

        def inp(y, y_fit, x_fit):
            return jnp.interp(y, y_fit, x_fit)

        x = jax.vmap(inp, in_axes=-1)(y, y_fit, x_fit)
        # x_inp = [jnp.interp(y[i], y_fit[:, i], x_fit[:,i]) for i in range(y_fit.shape[-1])]
        # x = jnp.stack(x_inp, axis=-1)
        return x

    def forward_and_log_det(self, x):
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.forward(x)
        logdet = self.forward_log_det_jacobian(x)
        return y, logdet

    def inverse_and_log_det(self, y):
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = self.inverse(y)
        logdet = 0
        return y, logdet

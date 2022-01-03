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


class BernsteinBijector(distrax.Bijector):
    def __init__(self, thetas):
        super().__init__(event_ndims_in=0, event_ndims_out=0)
        self.thetas = thetas
        self._is_injective = True

    def _forward(self, x):
        bernstein_poly = get_bernstein_poly(self.thetas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)
        return bernstein_poly(x)

    def _forward_log_det(self, x):
        bernstein_poly = get_bernstein_poly_jac(self.thetas)
        clip = 1e-7
        x = jnp.clip(x, clip, 1.0 - clip)
        return jnp.log(bernstein_poly(x))

    def forward_log_det_jacobian(self, x):
        return self._forward_log_det(x)

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

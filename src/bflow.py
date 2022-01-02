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


class Bernstein_Bijector(distrax.Bijector):
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
        # y_max = jnp.max(y_fit)
        # y_min = jnp.min(y_fit)
        # x = jnp.clip(x, y_min, y_max)
        yp = jnp.interp(x, y_fit, x_fit)
        return yp

    def forward_and_log_det(self, x):
        y = self._forward(x)
        logdet = self._forward_log_det(x)
        return y, logdet


theta = jnp.array(
    [
        -1.0744555,
        -0.6429366,
        -0.44160688,
        0.7950939,
        1.9249767,
        2.1408765,
        2.4256434,
        3.1641612,
        3.3939004,
    ]
)
c = [
    [-1.0744555],
    [-0.6429366],
    [-0.54160688],
    [1.7950939],
    [1.9249767],
    [1.9408765],
    [2.4256434],
    [3.1641612],
    [3.3939004],
]


# bp = BPoly(c, [0, 1])
# plt.plot(np.linspace(0, 1), bp(np.linspace(0, 1)))
# plt.show()


T1 = distrax.Chain([tfb.Scale(1), tfb.Shift(0)])
T2 = distrax.Inverse(distrax.Tanh())
T3 = distrax.Chain([tfb.Scale(0.9), tfb.Shift(0)])
chained_bijectors = [T3, T2]

a = jnp.array([2.0], dtype=np.float32)
b = jnp.array([-0.1], dtype=np.float32)
theta = jnp.array([-2.5, 2, 3, -7, -7, -7, -7, -7, 7])
alpha = jnp.array([0.2], dtype=np.float32)
beta = jnp.array([-2.50], dtype=np.float32)

T1 = distrax.Chain([tfb.Scale(softplus(a)), tfb.Shift(b)])
T2 = tfb.SoftClip(low=0, high=1, hinge_softness=1.5)
T3 = Bernstein_Bijector(thetas=constrain_thetas(theta))
T4 = distrax.Chain([tfb.Scale(softplus(alpha)), tfb.Shift(beta)])
chained_bijectors = [T4, T3, T2, T1]

bij = distrax.Inverse(distrax.Chain(chained_bijectors))

z = np.random.normal(size=100000)
y = bij.forward(z)
sns.distplot(y)
# plt.show()
plt.close()

# raise ValueError()

n = 500
cols = len(chained_bijectors) + 1
z_samples = np.linspace(-3, 3, n)
dist_base = distrax.Normal(0, 1)
fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 8))

ildj = 0.0
log_probs = dist_base.log_prob(z_samples)
ax[0].plot(z_samples, np.exp(log_probs))
ax[0].set_title("Normal (0, 1)")
zz = z_samples

names = ["Shift and Scale", "Bernstein-Polynom", "SoftClip", "Shift and Scale"]
ildj = 0.0
for i, (ax, bij) in enumerate(zip(ax[1:], chained_bijectors)):
    # we need to use the inverse here since we are going from z->y!
    z = bij.inverse(zz)
    ildj += bij.forward_log_det_jacobian(z)
    ax.plot(z, np.exp(log_probs + ildj))
    ax.set_title(names[i])
    ax.set_xlabel(f"$z_{i}$")
    ax.set_ylabel(f"$p(z_{i+1})$")
    zz = z
plt.savefig("./plots/Bernstein_Flow.jpg")
# plt.show()
plt.close()


def plot_poly(theta):
    bij = Bernstein_Bijector(thetas=theta)
    yy = np.linspace(0, 1, 200, dtype=np.float32)
    zz = bij.forward(yy)
    zi = zz + 1e-15 * np.random.random(zz.shape)
    yyy = bij.inverse(zi)
    plt.figure(figsize=(8, 8))
    plt.plot(yy, zz, alpha=0.5)
    plt.plot(yyy, zi, alpha=0.5)
    plt.title("Jax-Version")
    plt.show()


# Univariate
key = jax.random.PRNGKey(1234)
mu = 0.0
sigma = 1.0
dist_base = distrax.Normal(loc=mu, scale=sigma)
bij = distrax.Inverse(distrax.Chain(chained_bijectors))
dist_transformed = distrax.Transformed(dist_base, bij)

x1 = np.linspace(-3, 3, 200)
Y = dist_base.log_prob(x1)
# z = bij.inverse(x1)
# Z = dist_base.log_prob(z) + bij.forward_log_det_jacobian(z)
Z = dist_transformed.log_prob(x1)

fig, ax = plt.subplots(1, 2, figsize=(8, 8))
cm = plt.cm.get_cmap("viridis")
ax[0].plot(x1, np.exp(Y))
ax[1].plot(x1, np.exp(Z))
# plt.show()
plt.close()

# Independent Multivariate
key = jax.random.PRNGKey(1234)
mu = 0.0
sigma = 1.0
dist_base = distrax.Normal(loc=mu, scale=sigma)
bij = distrax.Inverse(distrax.Chain(chained_bijectors))
dist_transformed = distrax.Transformed(dist_base, bij)

x1 = np.linspace(-3, 3, 400)
x2 = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Y = dist_base.log_prob(X1) + dist_base.log_prob(X2)
Z = dist_transformed.log_prob(X1) + dist_transformed.log_prob(X2)

fig, ax = plt.subplots(1, 2, figsize=(8, 8))
cm = plt.cm.get_cmap("viridis")
ax[0].scatter(X1, X2, c=np.exp(Y), cmap=cm)
ax[1].scatter(X1, X2, c=np.exp(Z), cmap=cm)
# plt.show()
plt.close()


fig, ax = plt.subplots(1, 2, figsize=(14, 8))
ax[0].contourf(X1, X2, np.exp(Y))
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title("Base Distribution")
ax[1].contourf(X1, X2, np.exp(Z))
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title("Transformed Distribution")
plt.savefig("./plots/Density_MVN.jpg")
plt.close()
# Multivariate
key = jax.random.PRNGKey(1234)
mu = jnp.zeros(2)
sigma = jnp.ones(2)
dist_base = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
bij = distrax.Block(distrax.Inverse(distrax.Chain(chained_bijectors)), ndims=1)
dist_transformed = distrax.Transformed(dist_base, bij)

x1 = np.linspace(-3, 3, 400)
x2 = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Y = dist_base.log_prob(np.stack([X1, X2], axis=-1))
Z = dist_transformed.log_prob(np.stack([X1, X2], axis=-1))


fig, ax = plt.subplots(1, 2, figsize=(14, 8))
ax[0].contourf(X1, X2, np.exp(Y))
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title("Base Distribution")
ax[1].contourf(X1, X2, np.exp(Z))
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title("Transformed Distribution")

plt.savefig("./plots/MVN.jpg")

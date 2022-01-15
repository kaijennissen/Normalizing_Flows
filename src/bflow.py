import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as I
import seaborn as sns
from jax.nn import sigmoid, softmax, softplus
from tensorflow_probability.substrates import jax as tfp

from BernsteinBijector import BernsteinBijector, constrain_thetas

tfd = tfp.distributions
tfb = tfp.bijectors

# T1 = distrax.Chain([tfb.Scale(1), tfb.Shift(0)])
# T2 = distrax.Inverse(distrax.Tanh())
# T3 = distrax.Chain([tfb.Scale(0.9), tfb.Shift(0)])
# chained_bijectors = [T3, T2]

# Parameters for the BernsteinBijector
a = jnp.array([2.0], dtype=np.float32)
b = jnp.array([-0.1], dtype=np.float32)
theta = jnp.array([-2.5, 2, 3, -7, -7, -7, -7, -7, 7])
alpha = jnp.array([0.2], dtype=np.float32)
beta = jnp.array([-2.50], dtype=np.float32)

T1 = distrax.Chain([tfb.Scale(softplus(a)), tfb.Shift(b)])
T2 = tfb.SoftClip(low=0, high=1, hinge_softness=1.5)
T3 = BernsteinBijector(thetas=constrain_thetas(theta))
T4 = distrax.Chain([tfb.Scale(softplus(alpha)), tfb.Shift(beta)])
chained_bijectors = [T4, T3, T2, T1]

bij = distrax.Inverse(distrax.Chain(chained_bijectors))

# Plot change in density
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
    bij = BernsteinBijector(thetas=theta)
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

x1 = np.linspace(-3, 3, 2000)
x2 = np.linspace(-3, 3, 2000)
X1, X2 = np.meshgrid(x1, x2)
Y = np.exp(dist_base.log_prob(X1) + dist_base.log_prob(X2))
Z = np.exp(dist_transformed.log_prob(X1) + dist_transformed.log_prob(X2))

fig, ax = plt.subplots(1, 2, figsize=(8, 8))
cm = plt.cm.get_cmap("viridis")
ax[0].scatter(X1, X2, c=Y, cmap=cm)
ax[1].scatter(X1, X2, c=Z, cmap=cm)
# plt.show()
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 8))
ax[0].contourf(X1, X2, Y)
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title("Base Distribution")
ax[1].contourf(X1, X2, Z)
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title("Transformed Distribution")
plt.savefig("./plots/Density_MVN.jpg")
plt.close()

# Multivariate
mu = jnp.zeros(2)
sigma = jnp.ones(2)
dist_base = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
bij = distrax.Block(distrax.Inverse(distrax.Chain(chained_bijectors)), ndims=1)
dist_transformed = distrax.Transformed(dist_base, bij)

x1 = np.linspace(-3, 3, 2000)
x2 = np.linspace(-3, 3, 2000)
X1, X2 = np.meshgrid(x1, x2)
X = np.stack([X1, X2], axis=-1)
Y = np.exp(dist_base.log_prob(X))
Z = np.exp(dist_transformed.log_prob(X))

cm = plt.cm.get_cmap("viridis")
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
cset = ax.contourf(X1, X2, Z, zdir="z", offset=-0.6, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="x", offset=-4, cmap=cm)
cset = ax.contourf(X1, X2, Z, zdir="y", offset=4, cmap=cm)
ax.set_zlim(-0.6, np.max(Z) * 1.35)
ax.set_xlabel(r"$x_{1}$")
ax.set_ylabel(r"$x_{2}$")
ax.set_zlabel(r"$g(x_{1},x_{2})$")

ax.view_init(elev=20, azim=-45)
ax.grid(False)
fig.tight_layout(pad=3, w_pad=0.1)
plt.savefig("./plots/MVN_3D.jpg", dpi=600)
plt.close()

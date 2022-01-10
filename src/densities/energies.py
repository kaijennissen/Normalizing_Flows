import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from haiku import PRNGSequence
from jax import random
from jax.random import PRNGKey
from scipy.special import expit
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
DENSITY_NAME = "energy2"


def w3(z):
    z1 = z[:, 0]
    return 3 * expit((z1 - 1) / 0.3)


def energy_1_pdf(z):
    z1, z2 = z[:, 0], z[:, 1]
    norm = (z1 ** 2 + z2 ** 2) ** 0.5
    exp1 = np.exp(-0.5 * ((z1 - 2) / 0.6) ** 2)
    exp2 = np.exp(-0.5 * ((z1 + 2) / 0.6) ** 2)
    u = 0.5 * ((norm - 2) / 0.4) ** 2 - np.log(exp1 + exp2)
    return -u


def pot1f(z):
    return np.exp(-energy_1_log_pdf(z))


def energy_2_log_pdf(z):
    z2 = z[:, 1]
    return -0.5 * ((z2 - w1(z)) / 0.4) ** 2


def make_dataset_energy_2(seed: int, batch_size: int = 8, num_batches: int = 1):
    prng_seq = PRNGSequence(seed)

    def w1(z):
        return jnp.sin(2 * jnp.pi * z / 4)

    for _ in range(num_batches):
        key = next(prng_seq)
        dist = tfd.JointDistributionSequential(
            [
                tfd.Uniform(low=-3 / 2 * jnp.pi, high=3 / 2 * jnp.pi),
                lambda mu: tfd.Normal(loc=w1(mu), scale=0.50),
            ]
        )
        x1, x2 = dist.sample(seed=key, sample_shape=(batch_size,))
        yield jnp.stack([x1, x2], axis=-1)


def pot2f(z):
    return np.exp(-energy_2_log_pdf(z))


def energy_3_log_pdf(z):
    z2 = z[:, 1]

    x1 = -0.5 * ((z2 - w1(z)) / 0.35) ** 2
    x2 = -0.5 * ((z2 - w1(z) + w2(z)) / 0.35) ** 2
    a = np.maximum(x1, x2)
    exp1 = np.exp(x1 - a)
    exp2 = np.exp(x2 - a)
    return a + np.log(exp1 + exp2)


def pot3f(z):
    return np.exp(-energy_3_log_pdf(z))


def make_dataset_energy_3(seed: int, batch_size: int = 8, num_batches: int = 1):
    prng_seq = PRNGSequence(seed)

    def w1(z):
        return jnp.sin(2 * jnp.pi * z / 4)

    def w2(z):
        exp_arg = -0.5 * ((z - 1) / 0.6) ** 2
        return 3 * jnp.exp(exp_arg)

    for _ in range(num_batches):
        key = next(prng_seq)
        dist = tfd.JointDistributionSequential(
            [
                tfd.Uniform(low=-3 / 2 * jnp.pi, high=3 / 2 * jnp.pi),
                lambda mu: tfd.Normal(loc=w1(mu), scale=0.50),
            ]
        )
        x1, x2 = dist.sample(seed=key, sample_shape=(batch_size,))
        yield jnp.stack([x1, x2], axis=-1)


def energy_4_log_pdf(z):
    z2 = z[:, 1]
    x1 = -0.5 * ((z2 - w1(z)) / 0.4) ** 2
    x2 = -0.5 * ((z2 - w1(z) + w3(z)) / 0.35) ** 2
    a = np.maximum(x1, x2)
    exp1 = np.exp(x1 - a)
    exp2 = np.exp(x2 - a)
    return a + np.log(exp1 + exp2)  # Try adding a small value to prevent


def pot4f(z):
    return np.exp(-energy_4_log_pdf(z))


def contour_pot(potf, ax=None, title=None, xlim=5, ylim=5):
    grid = np.mgrid[-xlim:xlim:100j, -ylim:ylim:100j]
    grid_2d = grid.reshape(2, -1).T
    cmap = plt.get_cmap("inferno")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 9))
    pdf1e = np.exp(-potf(grid_2d))
    contour = ax.contourf(grid[0], grid[1], pdf1e.reshape(100, 100), cmap=cmap)
    if title is not None:
        ax.set_title(title, fontsize=16)
    return ax


# fig, ax = plt.subplots(2, 2, figsize=(12, 12))
# ax = ax.flatten()
# contour_pot(pot1f, ax[0], "Energy 1")
# contour_pot(pot2f, ax[1], "Energy 2")
# contour_pot(pot3f, ax[2], "Energy 3")
# contour_pot(pot4f, ax[3], "Energy 4")
# fig.tight_layout()

# plt.show()


if __name__ == "__main__":
    # plot density
    # x1 = jnp.linspace(-4, 4, 2000)
    # x2 = jnp.linspace(-4, 4, 2000)
    # X1, X2 = jnp.meshgrid(x1, x2)
    # Z = (jnp.stack([X1, X2], axis=-1), num_blobs=NUM_BLOBS, r=R)

    # cm = plt.cm.get_cmap("viridis")
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # surf = ax.plot_surface(X1, X2, Z, rstride=8, cstride=8, alpha=0.9, cmap=cm)
    # cset = ax.contourf(X1, X2, Z, zdir="z", offset=-0.15, cmap=cm)
    # cset = ax.contourf(X1, X2, Z, zdir="x", offset=-7, cmap=cm)
    # cset = ax.contourf(X1, X2, Z, zdir="y", offset=-10, cmap=cm)
    # ax.set_zlim(-0.15, jnp.max(Z) * 1.4)
    # ax.set_xlabel(r"$x_{2}$")
    # ax.set_ylabel(r"$x_{1}$")
    # ax.set_zlabel(r"$f(x_{1},x_{2})$")
    # ax.view_init(elev=20, azim=45)
    # ax.grid(False)
    # fig.tight_layout()
    # plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_pdf_3D.jpg", dpi=600)
    # plt.close()

    # cm = plt.cm.get_cmap("viridis")
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(111)
    # cset = ax.contourf(X1, X2, Z, cmap=cm)
    # ax.set_xlabel(r"$x_{1}$")
    # ax.set_ylabel(r"$x_{2}$")
    # fig.tight_layout()
    # plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}.jpg", dpi=600)
    # plt.close()

    # plot samples
    samples = next(make_dataset_energy_2(seed=45, batch_size=1000000))
    plt.hist2d(
        x=samples[:, 0],
        y=samples[:, 1],
        bins=100,
        range=np.array([[-4, 4], [-4, 4]]),
        cmap="viridis",
    )
    plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_samples.jpg", dpi=600)

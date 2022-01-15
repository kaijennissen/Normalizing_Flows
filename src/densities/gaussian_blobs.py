import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from haiku import PRNGSequence
from jax import random
from jax.random import PRNGKey
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
DENSITY_NAME = "gaussian_blobs"
SCALE = 0.3


def gaussian_blobs_pdf(data, num_blobs: int = 8, r: float = 2.0):
    offset = 0  # jnp.pi / num_blobs
    thetas = jnp.linspace(offset, 2 * jnp.pi + offset, num_blobs, endpoint=False)
    x = r * jnp.cos(thetas)
    y = r * jnp.sin(thetas)
    loc = jnp.stack([x, y], axis=-1)

    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=[1 / num_blobs for _ in range(num_blobs)]
        ),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=SCALE * jnp.ones((num_blobs, 2))
        ),
    )

    return dist.log_prob(data)


def make_dataset_gaussian_blobs(
    seed: int,
    batch_size: int = 8,
    num_batches: int = 1,
    num_blobs: int = 8,
    r: float = 2.0,
):
    prng_seq = PRNGSequence(seed)
    offset = 0  # jnp.pi / num_blobs
    thetas = jnp.linspace(offset, 2 * jnp.pi + offset, num_blobs, endpoint=False)
    x = r * jnp.cos(thetas)
    y = r * jnp.sin(thetas)
    loc = jnp.stack([x, y], axis=-1)

    dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            probs=[1 / num_blobs for _ in range(num_blobs)]
        ),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=SCALE * jnp.ones((num_blobs, 2))
        ),
    )

    for _ in range(num_batches):
        key = next(prng_seq)
        yield dist.sample(seed=key, sample_shape=(batch_size,))


if __name__ == "__main__":
    NUM_BLOBS = 8
    R = 2.0
    # plot density
    x1 = jnp.linspace(-6, 6, 2000)
    x2 = jnp.linspace(-6, 6, 2000)
    X1, X2 = jnp.meshgrid(x1, x2)
    Z = jnp.exp(
        gaussian_blobs_pdf(jnp.stack([X1, X2], axis=-1), num_blobs=NUM_BLOBS, r=R)
    )

    cm = plt.cm.get_cmap("viridis")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, rstride=8, cstride=8, alpha=0.9, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="z", offset=-0.3, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="x", offset=-7, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="y", offset=-10, cmap=cm)
    ax.set_zlim(-0.3, jnp.max(Z) * 1.4)
    ax.set_xlabel(r"$x_{2}$")
    ax.set_ylabel(r"$x_{1}$")
    ax.set_zlabel(r"$f(x_{1},x_{2})$")
    ax.view_init(elev=20, azim=45)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_pdf_3D.jpg", dpi=600)
    plt.close()

    cm = plt.cm.get_cmap("viridis")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    cset = ax.contourf(X1, X2, Z, cmap=cm)
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    fig.tight_layout()
    plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_pdf_2D.jpg", dpi=600)
    plt.close()

    # plot samples
    samples = next(
        make_dataset_gaussian_blobs(
            seed=45, batch_size=1000000, num_blobs=NUM_BLOBS, r=R
        )
    )

    plt.hist2d(
        x=samples[:, 0],
        y=samples[:, 1],
        bins=100,
        range=np.array([[-4, 4], [-4, 4]]),
        cmap="viridis",
    )
    plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_samples.jpg", dpi=600)

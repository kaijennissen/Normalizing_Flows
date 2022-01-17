import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from haiku import PRNGSequence
from jax import random
from jax.random import PRNGKey
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
DENSITY_NAME = "banana"

# Density 1
def banana_pdf(x1, x2):
    """pdf(x1,x2)=N(x1|(1/4)*x2**2,1)N(x2|0,4)"""
    lpx2 = tfd.Normal(loc=0, scale=jnp.sqrt(3)).log_prob(x2)
    lpx1 = tfd.Normal(loc=x2[:, jnp.newaxis] ** 2 / 3 - 1, scale=jnp.sqrt(3)).log_prob(
        x1
    )
    return jnp.exp(lpx1 + lpx2[:, jnp.newaxis])


# def banana_sample(prng_key, sample_shape):
#     """pdf(x1,x2)=N(x1|(1/4)*x2**2,1)N(x2|0,4)"""
#     key, subkey = random.split(prng_key, num=2)
#     x2 = tfd.Normal(loc=0, scale=jnp.sqrt(3)).sample(seed=key, sample_shape=sample_shape)
#     x1 = tfd.Normal(loc=x2 ** 2 / 3-2, scale=jnp.sqrt(3)/2).sample(seed=subkey)
#     return jnp.stack([x1, x2], axis=-1)


def make_dataset_banana(seed: int, batch_size: int = 8, num_batches: int = 1):
    """pdf(x1,x2)=N(x1|(1/4)*x2**2,1)N(x2|0,4)"""

    prng_seq = PRNGSequence(seed)
    for _ in range(num_batches):
        key, subkey = random.split(next(prng_seq), num=2)
        x2 = tfd.Normal(loc=0, scale=jnp.sqrt(3)).sample(
            seed=key, sample_shape=batch_size
        )
        x1 = tfd.Normal(loc=x2 ** 2 / 3 - 2, scale=jnp.sqrt(3) / 2).sample(seed=subkey)
        yield jnp.stack([x1, x2], axis=-1)


if __name__ == "__main__":
    # plot density
    x1 = jnp.linspace(-5, 8, 2000)
    x2 = jnp.linspace(-8, 8, 2000)
    X1, X2 = jnp.meshgrid(x1, x2)
    Z = banana_pdf(x1, x2)

    cm = plt.cm.get_cmap("viridis")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, rstride=8, cstride=8, alpha=0.9, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="z", offset=-0.15, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="x", offset=-7, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="y", offset=-10, cmap=cm)
    ax.set_zlim(-0.15, jnp.max(Z) * 1.4)
    ax.set_xlabel(r"$x_{2}$")
    ax.set_ylabel(r"$x_{1}$")
    ax.set_zlabel(r"$f(x_{1},x_{2})$")
    ax.view_init(elev=20, azim=45)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_pdf_3D.jpg", dpi=600)
    plt.close()

    cm = plt.cm.get_cmap("viridis")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    cset = ax.contourf(X1, X2, Z, cmap=cm)
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    fig.tight_layout()
    plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_pdf_2D.jpg", dpi=600)
    plt.close()

    # plot samples
    key = PRNGKey(254)
    samples = next(make_dataset_banana(seed=45, batch_size=1000000))
    plot_range = np.array([[-4, 4], [-4, 4]])
    figsize = (8, 8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist2d(
        x=samples[:, 0],
        y=samples[:, 1],
        bins=100,
        range=plot_range,
        cmap="viridis",
    )
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    ax.set_title("True")
    fig.tight_layout()
    plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_samples.jpg", dpi=750)

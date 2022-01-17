import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from haiku import PRNGSequence
from jax import random
from jax.random import PRNGKey
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def w1(z):
    return jnp.sin(2 * jnp.pi * z / 4)


def w2(z):
    exp_arg = -0.5 * ((z - 1) / 0.6) ** 2
    return 3 * jnp.exp(exp_arg)


def w3(z):
    return 3 * expit((z - 1) / 0.3)


def expit(x):
    return 1 / (1 + jnp.exp(-x))


def energy_1_pdf(data):
    def loc(z):
        mu = 2 - z ** 2 / 32
        return jnp.stack([mu, -mu], axis=-1)

    dist = tfd.JointDistributionSequential(
        [
            tfd.Normal(loc=0, scale=2),
            lambda mu: tfd.MixtureSameFamily(
                tfd.Categorical(probs=[0.5, 0.5]),
                tfd.Normal(loc=loc(mu), scale=0.3),
            ),
        ]
    )
    return jnp.sum(dist.log_prob(data), axis=-1)


def make_dataset_energy_1(seed: int, batch_size: int = 8, num_batches: int = 1):
    prng_seq = PRNGSequence(seed)

    def loc(z):
        mu = 2 - z ** 2 / 32
        return jnp.stack([mu, -mu], axis=-1)

    for _ in range(num_batches):
        key = next(prng_seq)
        dist = tfd.JointDistributionSequential(
            [
                tfd.Normal(loc=0, scale=2),
                lambda mu: tfd.MixtureSameFamily(
                    tfd.Categorical(probs=[0.5, 0.5]),
                    tfd.Normal(loc=loc(mu), scale=0.3),
                ),
            ]
        )
        x2, x1 = dist.sample(seed=key, sample_shape=(batch_size,))
        yield jnp.stack([x1, x2], axis=-1)


def make_dataset_energy_2(seed: int, batch_size: int = 8, num_batches: int = 1):
    prng_seq = PRNGSequence(seed)

    dist = tfd.JointDistributionSequential(
        [
            tfd.Uniform(low=-3 / 2 * jnp.pi, high=3 / 2 * jnp.pi),
            lambda mu: tfd.Normal(loc=w1(mu), scale=0.50),
        ]
    )

    for _ in range(num_batches):
        key = next(prng_seq)
        x1, x2 = dist.sample(seed=key, sample_shape=(batch_size,))
        yield jnp.stack([x1, x2], axis=-1)


def make_dataset_energy_3(seed: int, batch_size: int = 8, num_batches: int = 1):
    prng_seq = PRNGSequence(seed)

    def loc(x1):
        m1 = w1(x1)
        return jnp.stack([m1, m1 - w2(x1)], axis=-1)

    for _ in range(num_batches):
        key = next(prng_seq)
        dist = tfd.JointDistributionSequential(
            [
                tfd.Uniform(low=-3 / 2 * jnp.pi, high=3 / 2 * jnp.pi),
                lambda mu: tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
                    components_distribution=tfd.Normal(loc=loc(mu), scale=0.50),
                ),
            ]
        )
        x1, x2 = dist.sample(seed=key, sample_shape=(batch_size,))
        yield jnp.stack([x1, x2], axis=-1)


def make_dataset_energy_4(seed: int, batch_size: int = 8, num_batches: int = 1):
    prng_seq = PRNGSequence(seed)

    def loc(x1):
        m1 = w1(x1)
        m2 = m1 + w3(x1)
        return jnp.stack([m1, m2], axis=-1)

    for _ in range(num_batches):
        key = next(prng_seq)
        dist = tfd.JointDistributionSequential(
            [
                tfd.Uniform(low=-3 / 2 * jnp.pi, high=3 / 2 * jnp.pi),
                lambda mu: tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
                    components_distribution=tfd.Normal(loc=loc(mu), scale=0.50),
                ),
            ]
        )
        x1, x2 = dist.sample(seed=key, sample_shape=(batch_size,))
        yield jnp.stack([x1, x2], axis=-1)


if __name__ == "__main__":

    plot_range = np.array([[-4, 4], [-4, 4]])
    figsize = (8, 8)
    # plot samples
    densities = [
        make_dataset_energy_1,
        make_dataset_energy_2,
        make_dataset_energy_3,
        make_dataset_energy_4,
    ]
    for i, fn in enumerate(densities, start=1):
        DENSITY_NAME = f"energy{i}"
        samples = next(fn(seed=15, batch_size=1000000))
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

    # # plot samples
    # DENSITY_NAME = "energy2"
    # samples = next(make_dataset_energy_2(seed=67, batch_size=1000000))
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    # plt.hist2d(
    #     x=samples[:, 0],
    #     y=samples[:, 1],
    #     bins=100,
    #     range=plot_range,
    #     cmap="viridis",
    # )
    # plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_samples.jpg", dpi=600)

    # # plot samples
    # DENSITY_NAME = "energy3"
    # samples = next(make_dataset_energy_3(seed=45, batch_size=1000000))
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    # plt.hist2d(
    #     x=samples[:, 0],
    #     y=samples[:, 1],
    #     bins=100,
    #     range=plot_range,
    #     cmap="viridis",
    # )
    # plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_samples.jpg", dpi=600)

    # # plot samples
    # DENSITY_NAME = "energy4"
    # samples = next(make_dataset_energy_4(seed=21, batch_size=1000000))
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    # plt.hist2d(
    #     x=samples[:, 0],
    #     y=samples[:, 1],
    #     bins=100,
    #     range=plot_range,
    #     cmap="viridis",
    # )
    # ax.set_xlabel(r"$x_{1}$")
    # ax.set_ylabel(r"$x_{2}$")
    # ax.set_table("True")
    # plt.savefig(f"./plots/{DENSITY_NAME}/{DENSITY_NAME}_samples.jpg", dpi=600)

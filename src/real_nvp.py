import os
from typing import Any, Callable, List, Tuple

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from jax.random import KeyArray
from tensorflow_probability.substrates import jax as tfp

from densities.banana import make_dataset_banana
from densities.energies import (
    make_dataset_energy_1,
    make_dataset_energy_2,
    make_dataset_energy_3,
    make_dataset_energy_4,
)
from densities.gaussian_blobs import make_dataset_gaussian_blobs

tfd = tfp.distributions
tfb = tfp.bijectors

DENSITY = "energy2"


if DENSITY == "banana":
    make_dataset = make_dataset_banana
elif DENSITY == "gaussian_blobs":
    make_dataset = make_dataset_gaussian_blobs
elif DENSITY == "energy1":
    make_dataset = make_dataset_energy_1
elif DENSITY == "energy2":
    make_dataset = make_dataset_energy_2
elif DENSITY == "energy3":
    make_dataset = make_dataset_energy_3
elif DENSITY == "energy4":
    make_dataset = make_dataset_energy_4


def sample_N01(N: int, seed: KeyArray) -> jnp.ndarray:
    """Sample from the base distribution.

    Args:
        N (int): Number of samples.
        seed (KeyArray): Random see.

    Returns:
        [type]: [description]
    """
    return tfd.MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=jnp.ones(2)).sample(
        seed=seed, sample_shape=(N,)
    )


def log_prob_N01(x: jnp.ndarray) -> Callable:
    """Log probability of base distribution.

    Args:
        x (jnp.ndarray): [description]

    Returns:
        [type]: [description]
    """
    return tfd.MultivariateNormalDiag(
        loc=jnp.zeros(2), scale_diag=jnp.ones(2)
    ).log_prob(x)


def forward_nvp(
    net_params: list,
    shift_and_log_scale_fn: Callable,
    x: jnp.ndarray,
    flip: bool = False,
) -> jnp.ndarray:
    """[summary]

    Args:
        net_params (list): [description]
        shift_and_log_scale_fn (Callable): [description]
        x (np.ndarray): [description]
        flip (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    d = x.shape[-1] // 2
    x1 = x[:, :d]
    x2 = x[:, d:]
    if flip:
        x2, x1 = x1, x2
    shift, log_scale = shift_and_log_scale_fn(net_params, x1)

    y1 = x1
    y2 = x2 * jnp.exp(log_scale) + shift
    if flip:
        y1, y2 = y2, y1

    y = jnp.concatenate([y1, y2], axis=-1)
    return y


def inverse_nvp(
    net_params: list,
    shift_and_log_scale_fn: Callable,
    y: jnp.ndarray,
    flip: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """[summary]

    Args:
        net_params (list): [description]
        shift_and_log_scale_fn (Callable): [description]
        y (np.ndarray): [description]
        flip (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    d = y.shape[-1] // 2
    y1 = y[:, :d]
    y2 = y[:, d:]
    if flip:
        y1, y2 = y2, y1
    shift, log_scale = shift_and_log_scale_fn(net_params, y1)

    x1 = y1
    x2 = (y2 - shift) * jnp.exp(-log_scale)
    if flip:
        x1, x2 = x2, x1

    x = jnp.concatenate([x1, x2], axis=-1)
    return x, log_scale


def log_prob_nvp(
    net_params: list,
    shift_and_log_scale_fn: Callable,
    base_log_prob_fn: Callable,
    y: jnp.ndarray,
    flip: bool = False,
) -> float:
    """[summary]

    Args:
        net_params (list): [description]
        shift_and_log_scale_fn (Callable): [description]
        log_prob_fn (Callable): [description]
        y (jnp.ndarray): [description]
        flip (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    x, log_scale = inverse_nvp(net_params, shift_and_log_scale_fn, y, flip)
    inv_log_det = -jnp.sum(log_scale, axis=-1)
    return base_log_prob_fn(x) + inv_log_det


def shift_and_log_scale_fn(
    net_params: list, x: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    act = x
    for W, b in net_params[:-1]:
        out = jnp.matmul(act, W) + b
        act = jax.nn.leaky_relu(out)

    W, b = net_params[-1]
    out = jnp.dot(act, W) + b
    return jnp.split(out, 2, axis=1)


def init_mlp(hidden_units: List[int] = [4]) -> List[List[jnp.ndarray]]:
    """[summary]

    Args:
        hidden_units (list, optional): [description]. Defaults to [4].

    Returns:
        [type]: [description]
    """
    # two output nodes, shift and log_scale
    # D: int = int(2)
    # hidden_units.insert(0, 1)
    # hidden_units.append(D)
    hidden_units = [1, 4, 2]
    params = []
    key = random.PRNGKey(534)
    for m, n in zip(hidden_units[:-1], hidden_units[1:]):
        key, subkey = random.split(key=key, num=2)
        W = random.normal(subkey, shape=(m, n)) * 0.1
        b = jnp.zeros((n,))
        params.append([W, b])
    return params


def init_nvp_chain(
    n: int = 2,
) -> Tuple[List[List[List[jnp.ndarray]]], List[Tuple[Callable, bool]]]:
    """[summary]

    Args:
        n (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    flip = False
    params = []
    fns_config = []

    fn = shift_and_log_scale_fn
    for _ in range(n):
        net_params = init_mlp()
        params.append(net_params)
        fns_config.append((fn, flip))
        flip = not flip

    return params, fns_config


# def sample_nvp(net_params, shift_log_scale_fn, base_sample_fn, N, flip: bool = False):
#     x = base_sample_fn(N)
#     return forward_nvp(net_params, shift_log_scale_fn, x, flip=flip)


def sample_nvp_chain(
    params: list, fns_config: list, base_sample_fn: Callable, N: int, key: KeyArray
) -> jnp.ndarray:
    """[summary]

    Args:
        params (list): [description]
        fns_config (list): [description]
        base_sample_fn (Callable): [description]
        N (int): [description]
        key (KeyArray): [description]

    Returns:
        [type]: [description]
    """

    z = base_sample_fn(N, key)
    for net_params, (fn, flip) in zip(params, fns_config):
        z = forward_nvp(net_params, fn, z, flip=flip)

    return z


def make_log_prob_fn(
    net_params: list, log_prob_fn: Callable, fn_config: Tuple
) -> Callable:
    """[summary]

    Args:
        net_params ([type]): [description]
        log_prob_fn ([type]): [description]
        fns_config ([type]): [description]

    Returns:
        [type]: [description]
    """
    fn, flip = fn_config
    return lambda x: log_prob_nvp(net_params, fn, log_prob_fn, x, flip=flip)


def log_prob_nvp_chains(params, fns_config, base_log_prob_fn, y) -> jnp.ndarray:
    """[summary]

    Args:
        params ([type]): [description]
        fns_config ([type]): [description]
        base_log_prob_fn ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    log_prob_fn = base_log_prob_fn
    for net_params, fn_config in zip(params, fns_config):
        log_prob_fn = make_log_prob_fn(net_params, log_prob_fn, fn_config)
    return log_prob_fn(y)


if __name__ == "__main__":
    training_steps = 50001
    learning_rate = 1e-2
    params, fns_config = init_nvp_chain(n=8)
    # filenames = []

    def loss(params, batch) -> jnp.float32:
        return -jnp.mean(log_prob_nvp_chains(params, fns_config, log_prob_N01, batch))

    @jax.jit
    def update(params, batch) -> Tuple[jnp.float32, List[List[List[jnp.ndarray]]]]:
        nll, grads = jax.value_and_grad(loss)(params, batch)
        params = [
            [
                [W - learning_rate * dW, b - learning_rate * db]
                for (W, b), (dW, db) in zip(net_params, net_grads)
            ]
            for net_params, net_grads in zip(params, grads)
        ]
        return nll, params

    # Training
    train_ds = make_dataset(seed=231, batch_size=128, num_batches=training_steps)

    for step in range(training_steps):
        nll, params = update(params, next(train_ds))

        if step % 1000 == 0:
            print(f"Step: {step}; NLL: {nll}")

        if step % 2500 == 0:

            key = random.PRNGKey(34)
            x = next(make_dataset(seed=key, batch_size=1000000, num_batches=1))
            y = sample_nvp_chain(
                params, fns_config, sample_N01, 1000000, random.PRNGKey(7809)
            )
            plot_range = np.array([[-4, 4], [-4, 4]])
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].hist2d(x[:, 0], x[:, 1], bins=100, cmap="viridis", range=plot_range)
            axes[1].hist2d(y[:, 0], y[:, 1], bins=100, cmap="viridis", range=plot_range)
            filename = f"plots/{DENSITY}/real_nvp_{DENSITY}_{step}.jpg"
            fig.tight_layout()
            plt.savefig(filename, dpi=75)
            plt.close()

    # print("Creating gif\n")

    # with imageio.get_writer(
    #     "plots/{DENSITY}/real_nvp_{DENSITY}.gif", mode="I"
    # ) as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    # print("Gif saved\n")
    # print("Removing Images\n")
    # # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)
    # print("DONE")

    key = random.PRNGKey(34)
    x = next(make_dataset(seed=key, batch_size=1000000, num_batches=1))
    y = sample_nvp_chain(params, fns_config, sample_N01, 1000000, random.PRNGKey(7809))
    plot_range = np.array([[-4, 4], [-4, 4]])
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axes[0].hist2d(x[:, 0], x[:, 1], bins=100, cmap="viridis", range=plot_range)
    ax.hist2d(y[:, 0], y[:, 1], bins=100, cmap="viridis", range=plot_range)
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    ax.set_title("Real NVP")
    fig.tight_layout()
    plt.savefig(f"plots/{DENSITY}/real_nvp_{DENSITY}.jpg", dpi=750)
    plt.close()

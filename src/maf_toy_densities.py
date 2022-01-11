import argparse
from re import L
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from distrax._src.bijectors.bijector import Array
from jax import random
from tensorflow_probability.substrates import jax as tfp

from BernsteinBijector import BernsteinBijector
from densities.banana import banana_pdf, make_dataset_banana
from densities.energies import (
    energy_1_pdf,
    make_dataset_energy_1,
    make_dataset_energy_2,
    make_dataset_energy_3,
    make_dataset_energy_4,
)
from densities.gaussian_blobs import gaussian_blobs_pdf, make_dataset_gaussian_blobs

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


# TODO: rewrite to allow for arbitrary density
# mask make_dataset with dataset specific generator which is importet from the densities folder
# e.g. make_dataset=make_dataset_banana
# same with dataset_pdf
# e.g. dataset_pdf=banana_pdf

DENSITY = "energy1"

FLOW_NUM_LAYERS = 2
HIDDEN_SIZE = 32
MLP_NUM_LAYERS = 2
FLOW_NUM_PARAMS = 12

if DENSITY == "banana":
    make_dataset = make_dataset_banana
    dataset_pdf = banana_pdf
elif DENSITY == "gaussian_blobs":
    make_dataset = make_dataset_gaussian_blobs
    dataset_pdf = gaussian_blobs_pdf
elif DENSITY == "energy1":
    make_dataset = make_dataset_energy_1
    dataset_pdf = energy_1_pdf
elif DENSITY == "energy2":
    make_dataset = make_dataset_energy_2
elif DENSITY == "energy3":
    make_dataset = make_dataset_energy_3
elif DENSITY == "energy4":
    make_dataset = make_dataset_energy_4


def make_conditioner(
    event_shape: Sequence[int], hidden_sizes: Sequence[int], num_bijector_params: int
) -> hk.Sequential:

    return hk.Sequential(
        [
            hk.Flatten(preserve_dims=-1),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),
            hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ]
    )


def make_flow_model(
    event_shape: Sequence[int],
    num_layers: int,
    hidden_sizes: Sequence[int],
    flow_num_params: int,
) -> distrax.Transformed:

    mask = jnp.arange(0, np.prod(event_shape)) % 2  # every second element is masked
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    # def bijector_fn(params: Array):
    #     return BernsteinBijector(params)

    flow_num_params = 3 * flow_num_params + 1

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(params, range_min=-8.0, range_max=8.0)

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            bijector=bijector_fn,
            conditioner=make_conditioner(
                event_shape=event_shape,
                hidden_sizes=hidden_sizes,
                num_bijector_params=flow_num_params,
            ),
        )

        layers.append(layer)
        # Flip mask after each layer
        mask = jnp.logical_not(mask)

    flow = distrax.Inverse(distrax.Chain(layers))
    base_distribution = distrax.Independent(
        distrax.Normal(loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )

    return distrax.Transformed(base_distribution, flow)


@hk.without_apply_rng
@hk.transform
def log_prob(data: Array) -> Array:

    model = make_flow_model(
        event_shape=data.shape[-1:],
        num_layers=FLOW_NUM_LAYERS,
        hidden_sizes=[HIDDEN_SIZE] * MLP_NUM_LAYERS,
        flow_num_params=FLOW_NUM_PARAMS,  # num_bins / bernstein_order
    )

    return model.log_prob(data)


@hk.without_apply_rng
@hk.transform
def sample(event_shape: Tuple, prng_key: PRNGKey, num_samples: int):  # type: ignore

    model = make_flow_model(
        event_shape=event_shape,
        num_layers=FLOW_NUM_LAYERS,
        hidden_sizes=[HIDDEN_SIZE] * MLP_NUM_LAYERS,
        flow_num_params=FLOW_NUM_PARAMS,  # num_bins / bernstein_order
    )

    return model.sample(seed=prng_key, sample_shape=(num_samples,))


# @jax.jit
def loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:  # type: ignore
    # data = prepare_data(batch, prng_key)
    # Loss is average negative log likelihood.
    loss = -jnp.mean(log_prob.apply(params, batch))
    return loss


# @jax.jit
def eval_fn(params: hk.Params, batch: Batch) -> Array:
    # data = prepare_data(batch)  # We don't dequantize during evaluation.
    loss = -jnp.mean(log_prob.apply(params, batch))
    return loss


def main(
    batch_size,
    learning_rate,
    training_steps,
    eval_frequency,
):

    optimizer = optax.adam(learning_rate)

    # @jax.jit
    def update(
        params: hk.Params, prng_key: PRNGKey, opt_state: OptState, batch: Batch
    ) -> Tuple[hk.Params, OptState]:

        """Single SGD update step."""
        grads = jax.grad(loss_fn)(params, prng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    prng_seq = hk.PRNGSequence(42)
    params = log_prob.init(next(prng_seq), np.zeros((1, 2)))
    opt_state = optimizer.init(params)

    train_ds = make_dataset(
        seed=2123, batch_size=batch_size, num_batches=2 * training_steps
    )
    valid_ds = make_dataset(
        seed=2235, batch_size=batch_size, num_batches=training_steps
    )

    for step in range(training_steps):
        params, opt_state = update(params, next(prng_seq), opt_state, next(train_ds))

        if step % eval_frequency == 0:
            train_loss = eval_fn(params, next(train_ds))
            val_loss = eval_fn(params, next(valid_ds))
            print(
                f"STEP: {step}; training loss: {train_loss}, validation loss: {val_loss}"
            )

    # Evaluate
    N = 1000000
    samples_maf = sample.apply(
        params=params,
        prng_key=next(prng_seq),
        event_shape=(2,),
        num_samples=N,
    )
    samples = next(make_dataset(seed=99, batch_size=N, num_batches=1))

    # make plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    plot_range = np.array([[-4, 4], [-4, 4]])
    # Samples from true distrubution
    ax = axes[0]
    ax.hist2d(samples[:, 0], samples[:, 1], bins=100, range=plot_range)
    ax.set_title("True")

    # Samples from learned distrubution
    ax = axes[1]
    ax.hist2d(samples_maf[:, 0], samples_maf[:, 1], bins=100, range=plot_range)
    ax.set_title("Reconstructed")

    for ax in axes.reshape(-1):
        ax.set_xlabel(r"$x_{1}$")
        ax.set_ylabel(r"$x_{2}$")
    fig.tight_layout()
    plt.savefig(f"./plots/{DENSITY}/maf_{DENSITY}.jpg", dpi=600)

    # num_points = 2000
    # x1 = jnp.linspace(-4, 4, num_points)
    # x2 = jnp.linspace(-4, 4, num_points)
    # X1, X2 = jnp.meshgrid(x1, x2)

    # # pdf values of true and learned distribution
    # X1X2 = jnp.stack([X1, X2], axis=-1)
    # Z1 = dataset_pdf(X1X2)
    # Z2 = jnp.exp(log_prob.apply(params, X1X2))

    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # # axes[0].contourf(X1, X2, Z1, cmap="viridis")
    # axes[1].contourf(X1, X2, Z2, cmap="viridis")
    # plt.savefig(f"plots/{DENSITY}/{DENSITY}_pdf.jpg", dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow_num_layers",
        default=8,
        type=int,
        help="Number of layers to use in the flow.",
    )
    parser.add_argument(
        "--mlp_num_layers",
        default=2,
        type=int,
        help="Number of layers to use in the MLP conditioner.",
    )
    parser.add_argument(
        "--hidden_size",
        default=50,
        type=int,
        help="Hidden size of the MLP conditioner.",
    )
    parser.add_argument(
        "--flow_num_params",
        default=4,
        type=int,
        help="Order of the Bernstein-Polynomial.",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--training_steps",
        default=5000,
        type=int,
        help="Number of training steps to run.",
    )
    parser.add_argument(
        "--eval-frequency",
        default=100,
        type=int,
        help="How often to evaluate the model.",
    )

    args = parser.parse_args()
    main(
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_frequency=args.eval_frequency,
    )

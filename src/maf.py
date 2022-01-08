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
import seaborn as sns
import tensorflow_datasets as tfds
from distrax._src.bijectors.bijector import Array
from jax import random
from tensorflow_probability.substrates import jax as tfp

from BernsteinBijector import BernsteinBijector

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any

MNIST_IMAGE_SHAPE = (28, 28, 1)


FLOW_NUM_LAYERS = 10
HIDDEN_SIZE = 500
MLP_NUM_LAYERS = 3
FLOW_NUM_PARAMS = 6


def make_conditioner(
    event_shape: Sequence[int], hidden_sizes: Sequence[int], num_bijector_params: int
) -> hk.Sequential:

    return hk.Sequential(
        [
            hk.Flatten(preserve_dims=-len(event_shape)),
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
        return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

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
    # base_distribution = distrax.Independent(
    #     distrax.Normal(loc=jnp.zeros(event_shape), scale=jnp.ones(event_shape)),
    #     reinterpreted_batch_ndims=len(event_shape),
    # )
    base_distribution = distrax.Independent(
        distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )
    return distrax.Transformed(base_distribution, flow)


def load_dataset(split: tfds.Split, batch_size: int) -> Iterator[Batch]:
    ds = tfds.load("mnist", split=split, shuffle_files=True)
    ds = ds.shuffle(buffer_size=10 * batch_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=5)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:  # type: ignore
    data = batch["image"].astype(np.float32)
    if prng_key is not None:
        # Dequantize pixel values {0, 1, ..., 255} with uniform noise [0, 1).
        data += jax.random.uniform(prng_key, data.shape)
    return data / 256.0  # Normalize pixel values from [0, 256) to [0, 1).


@hk.without_apply_rng
@hk.transform
def log_prob(data: Array) -> Array:
    model = make_flow_model(
        event_shape=data.shape[1:],
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


def loss_fn(params: hk.Params, prng_key: PRNGKey, batch: Batch) -> Array:  # type: ignore
    data = prepare_data(batch, prng_key)
    # Loss is average negative log likelihood.
    loss = -jnp.mean(log_prob.apply(params, data))
    return loss


@jax.jit
def eval_fn(params: hk.Params, batch: Batch) -> Array:
    data = prepare_data(batch)  # We don't dequantize during evaluation.
    loss = -jnp.mean(log_prob.apply(params, data))
    return loss


def main(
    flow_num_layers,
    mlp_num_layers,
    flow_num_params,
    hidden_size,
    batch_size,
    learning_rate,
    training_steps,
    eval_frequency,
):
    optimizer = optax.adam(learning_rate)

    @jax.jit
    def update(
        params: hk.Params, prng_key: PRNGKey, opt_state: OptState, batch: Batch
    ) -> Tuple[hk.Params, OptState]:

        """Single SGD update step."""
        grads = jax.grad(loss_fn)(params, prng_key, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    prng_seq = hk.PRNGSequence(42)
    params = log_prob.init(next(prng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE)))
    opt_state = optimizer.init(params)

    train_ds = load_dataset(tfds.Split.TRAIN, batch_size)
    valid_ds = load_dataset(tfds.Split.TEST, batch_size)

    for step in range(training_steps):
        params, opt_state = update(params, next(prng_seq), opt_state, next(train_ds))

        if step % eval_frequency == 0:
            val_loss = eval_fn(params, next(valid_ds))
            print(f"STEP: {step}; validation loss: {val_loss}")

    N = 16
    samples = sample.apply(
        params=params,
        prng_key=next(prng_seq),
        event_shape=MNIST_IMAGE_SHAPE,
        num_samples=N,
    )
    cols = rows = int(np.ceil(np.sqrt(N)))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for i in range(N):
        r = int(i // rows)
        c = int(i % cols)
        ax = axes[r, c]
        ax.imshow(samples[i, ...])
    # plt.show()
    plt.savefig("./plots/MNIST_samples.jpg")


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
        "--batch_size",
        default=128,
        type=int,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
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
        "--eval_frequency",
        default=100,
        type=int,
        help="How often to evaluate the model.",
    )

    args = parser.parse_args()
    main(
        flow_num_layers=args.flow_num_layers,
        mlp_num_layers=args.mlp_num_layers,
        hidden_size=args.hidden_size,
        flow_num_params=args.flow_num_params,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_frequency=args.eval_frequency,
    )

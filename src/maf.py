import argparse
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
from absl import app, flags, logging
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


class AffineScalar(distrax.Bijector):
    def __init__(self, scale, shift):

        if hasattr(scale, "shape"):
            D = scale.ndim
            assert scale.shape == shift.shape
        else:
            D = 0

        super().__init__(event_ndims_in=D, event_ndims_out=D)
        self.bijector = distrax.Chain([tfb.Shift(shift), tfb.Scale(scale)])

    def forward(self, x: Array) -> Array:
        return self.bijector.forward(x)

    def inverse(self, y: Array) -> Array:
        return self.bijector.inverse(y)

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        return self.bijector.forward_and_log_det(x)

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        return self.bijector.inverse_and_log_det(y)


bij = AffineScalar(scale=jnp.array([0.2, 0.5]), shift=jnp.array([1.0, -1]))
bij.forward(jnp.array([[1.0, -2], [2, -4], [3, -6]]))


# MAF with
# - Transformer: Affine
# - Consitioner: Masking
def conditioner(z: Array) -> Array:
    assert z.ndim == 2
    B, D = z.shape
    norm = jnp.sqrt(jnp.cumsum(jnp.square(z), axis=-1))
    beta = jnp.concatenate([jnp.zeros((B, 1)), norm[..., :-1]], axis=-1) / 10
    alpha = jnp.exp(-beta / D)
    return alpha, beta


def transformer(z: Array) -> Array:
    alpha, beta = conditioner(z)
    return alpha * z + beta


mu = jnp.array([-1, -1])
sigma = jnp.ones(2)
dist_base = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
key = random.PRNGKey(52)

z = dist_base.sample(seed=key, sample_shape=(1000000,))

N = 32
cols = rows = int(np.ceil(np.sqrt(N / 2)))

fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))

zz = z
l = 1  # frew of switching the conditioner
for j in range(N):
    if j % l == 0:
        zz = transformer(jnp.concatenate([zz[..., 1:], zz[..., :1]], axis=1))
    else:
        zz = transformer(zz)
    if j % 2 == 0:
        i = j / 2
        r = int(i // rows)
        c = int(i % cols)
        ax = axes[r, c]
        x = zz[:, 0]
        y = zz[:, 1]
        # sns.scatterplot(x=x, y=y, s=5, color=".15",ax=ax)
        sns.histplot(x=x, y=y, bins=100, pthresh=0.1, cmap="viridis", ax=ax)
        # sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1,ax=ax)
        # ax.set_title(f"k={j}")
        # ax.hist2d(zz[:, 0], zz[:, 1], bins=100)

fig.tight_layout()
plt.savefig("./plots/MAF.jpg")

# TODO: replace conditioner with network


# network


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
    bernstein_order: int,
):

    mask = jnp.arange(0, np.prod(event_shape)) % 2  # every second element is masked
    mask = jnp.reshape(mask, event_shape)
    mask = mask.astype(bool)

    # def bijector_fn(params: Array):
    #     return BernsteinBijector(params)

    def bijector_fn(params: Array):
        return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

    layers = []
    for _ in range(num_layers):
        layer = distrax.MaskedCoupling(
            mask=mask,
            conditioner=make_conditioner(
                event_shape=event_shape,
                hidden_sizes=hidden_sizes,
                num_bijector_params=bernstein_order,
            ),
            bijector=bijector_fn,
        )
        # T1 = distrax.Chain([tfb.Scale(softplus(a)), tfb.Shift(b)])
        # T2 = tfb.SoftClip(low=0, high=1, hinge_softness=1.5)
        # T3 = BernsteinBijector(thetas=constrain_thetas(theta))
        # T4 = distrax.Chain([tfb.Scale(softplus(alpha)), tfb.Shift(beta)])
        # chained_bijectors = [T4, T3, T2, T1]
        layers.append(layer)
        # Flip mask after each layer
        mask = jnp.logical_not(mask)

    flow = distrax.Inverse(distrax.Chain(layers))
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


def prepare_data(batch: Batch, prng_key: Optional[PRNGKey] = None) -> Array:
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
        num_layers=2,
        hidden_sizes=[8] * 2,
        bernstein_order=4,
    )
    return model.log_prob(data)


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
    bernstein_order,
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
        default=500,
        type=int,
        help="Hidden size of the MLP conditioner.",
    )
    parser.add_argument(
        "--bernstein_order",
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
        bernstein_order=args.bernstein_order,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_frequency=args.eval_frequency,
    )

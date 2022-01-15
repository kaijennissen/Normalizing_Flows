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


FLOW_NUM_LAYERS = 1
HIDDEN_SIZE = 50
MLP_NUM_LAYERS = 2
FLOW_NUM_PARAMS = 4

# # MAF with
# # - Transformer: Affine
# # - Consitioner: Masking
# def conditioner(z: Array) -> Array:
#     assert z.ndim == 2
#     B, D = z.shape
#     norm = jnp.sqrt(jnp.cumsum(jnp.square(z), axis=-1))
#     beta = jnp.concatenate([jnp.zeros((B, 1)), norm[..., :-1]], axis=-1) / 10
#     alpha = jnp.exp(-beta / D)
#     return alpha, beta


# def transformer(z: Array) -> Array:
#     alpha, beta = conditioner(z)
#     return alpha * z + beta


# mu = jnp.array([-1, -1])
# sigma = jnp.ones(2)
# dist_base = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
# key = random.PRNGKey(52)

# z = dist_base.sample(seed=key, sample_shape=(1000000,))

# N = 32
# cols = rows = int(np.ceil(np.sqrt(N / 2)))

# fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))

# zz = z
# l = 1  # frew of switching the conditioner
# for j in range(N):
#     if j % l == 0:
#         zz = transformer(jnp.concatenate([zz[..., 1:], zz[..., :1]], axis=1))
#     else:
#         zz = transformer(zz)
#     if j % 2 == 0:
#         i = j / 2
#         r = int(i // rows)
#         c = int(i % cols)
#         ax = axes[r, c]
#         x = zz[:, 0]
#         y = zz[:, 1]
#         # sns.scatterplot(x=x, y=y, s=5, color=".15",ax=ax)
#         sns.histplot(x=x, y=y, bins=100, pthresh=0.1, cmap="viridis", ax=ax)
#         # sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1,ax=ax)
#         # ax.set_title(f"k={j}")
#         # ax.hist2d(zz[:, 0], zz[:, 1], bins=100)

# fig.tight_layout()
# plt.savefig("./plots/MAF.jpg")

# # TODO: replace conditioner with network


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def loss_fn(images, labels):
    mlp = hk.Sequential(
        [
            hk.Linear(300),
            jax.nn.relu,
            hk.Linear(100),
            jax.nn.relu,
            hk.Linear(8),
        ]
    )
    thetas = mlp(images)
    bij = BernsteinBijector(thetas)
    base_dist = distrax.Uniform(
        low=jnp.zeros(images.shape), high=jnp.ones(images.shape)
    )
    transformed_dist = distrax.Transformed(base_dist, bij)
    return jnp.mean(transformed_dist.log_prob(images), axis=(1, 2, 3))


loss_fn_t = hk.transform(loss_fn)
loss_fn_t = hk.without_apply_rng(loss_fn_t)


batch = next(train_ds)
x = batch["image"]
y = batch["label"]

rng = jax.random.PRNGKey(42)
dummy_images, dummy_labels = np.zeros((4, *MNIST_IMAGE_SHAPE)), np.arange(4)
params = loss_fn_t.init(rng, dummy_images, dummy_labels)
logits = loss_fn_t.apply(params, dummy_images, dummy_labels)
logits.shape

logits = loss_fn_t.apply(params, x, y)

logits = jnp.array([[0.4, 0.4, 0.2, 0], [0.4, 0.2, 0.2, 0.2]])
dist = distrax.Categorical(probs=logits)
np.exp(dist.log_prob(dummy_labels))


def update_rule(param, update):
    return param - 0.01 * update


for batch in train_ds:
    x = batch["image"]
    y = batch["label"]
    grads = jax.grad(loss_fn_t.apply)(params, x, y)
    params = jax.tree_multimap(update_rule, params, grads)

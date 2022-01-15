import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# Rotaton Flow
class OrthogonalProjection2D(distrax.Bijector):
    def __init__(self, theta):
        super().__init__(event_ndims_in=1, event_ndims_out=1)
        self.thetas = theta
        self.sin_theta = jnp.sin(theta)
        self.cos_theta = jnp.cos(theta)
        self.R = jnp.array(
            [[self.cos_theta, -self.sin_theta], [self.sin_theta, self.cos_theta]]
        ).T

    def forward(self, x):
        return jnp.matmul(x, self.R)

    def inverse(self, x):
        return jnp.matmul(x, self.R.T)

    def forward_and_log_det(self, x):
        y = self.forward(x)
        logdet = 1
        return y, logdet

    def inverse_and_log_det(self, x):
        y = self.inverse(x)
        logdet = 1
        return y, logdet


# Step 1:
# Visualize the transformation

key = random.PRNGKey(2364)
x = distrax.MultivariateNormalDiag(
    loc=jnp.zeros(2), scale_diag=jnp.array([1.5, 0.5])
).sample(seed=key, sample_shape=(1000000,))
Tx = OrthogonalProjection2D(theta=jnp.pi / 4).forward(x)


binrange = jnp.array([[-7, 7], [-7, 7]])
cm = plt.get_cmap("viridis")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(x=x[:, 0], y=x[:, 1], bins=100, cmap=cm, ax=axes[0], binrange=binrange)
sns.histplot(x=Tx[:, 0], y=Tx[:, 1], bins=100, cmap=cm, ax=axes[1], binrange=binrange)
axes[0].set_title("Base Distribution")
axes[1].set_title("Transformed Distribution")
for ax in axes:
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
fig.tight_layout()
plt.savefig("plots/rotation/rotation1.jpg")
plt.close()

# Step 2:
#  Generate Samples from the transformed distribution given a set of parameters


def make_dataset(seed, batch_size: int, num_batches: int):

    bij = OrthogonalProjection2D(theta=jnp.pi / 3)
    base_dist = distrax.MultivariateNormalDiag(
        loc=jnp.array([3.0, 1.0]), scale_diag=jnp.array([0.75, 3.0])
    )
    true_dist = distrax.Transformed(base_dist, bij)

    key = random.PRNGKey(seed)
    for _ in range(num_batches):
        key, subkey = random.split(key)
        yield true_dist.sample(seed=subkey, sample_shape=(batch_size,))


# Step 3:
def make_flow(params):
    a = params[0]
    b = params[1]
    theta = params[2]
    T1 = distrax.Block(tfb.Scale(a), ndims=1)
    T2 = distrax.Block(tfb.Shift(b), ndims=1)
    T3 = OrthogonalProjection2D(theta=theta)
    bij = distrax.Chain([T3, T2, T1])

    base_dist = distrax.MultivariateNormalDiag(loc=jnp.zeros(2), scale_diag=jnp.ones(2))
    flow_dist = distrax.Transformed(base_dist, bij)

    return flow_dist


def nll(params, batch):
    flow_dist = make_flow(params)
    return -jnp.mean(flow_dist.log_prob(batch))


@jax.jit
def update(params, batch, learning_rate):
    loss, grads = jax.value_and_grad(nll)(params, batch)
    params = [p - learning_rate * g for p, g in zip(params, grads)]
    return params, loss


training_steps = 50000
a_init = jnp.array([1.0, 1.0])
b_init = jnp.array([0.0, 0.0])
theta_init = jnp.pi
params = [a_init, b_init, theta_init]
train_ds = make_dataset(seed=435, batch_size=128, num_batches=training_steps)
for step in range(training_steps):
    params, loss = update(params=params, batch=next(train_ds), learning_rate=1e-3)
    if step % 1000 == 0:
        print(f"Loss: {loss}")

flow_dist = make_flow(params)
x = next(make_dataset(seed=435, batch_size=1000000, num_batches=1))
Tx = flow_dist.sample(seed=random.PRNGKey(23), sample_shape=(1000000,))

binrange = jnp.array([[-13, 13], [-7, 13]])
cm = plt.get_cmap("viridis")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(x=x[:, 0], y=x[:, 1], bins=100, cmap=cm, ax=axes[0], binrange=binrange)
sns.histplot(x=Tx[:, 0], y=Tx[:, 1], bins=100, cmap=cm, ax=axes[1], binrange=binrange)
axes[0].set_title("True")
axes[1].set_title("Learned")
fig.tight_layout()
plt.savefig("plots/rotation/rotation2.jpg")

plt.close()

# TODO: learn parameters of rotation with simple jax NN

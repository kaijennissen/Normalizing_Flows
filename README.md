# Bayesian Deep Learning
In this repository I collect some toy examples of Bayesian Deep Learning.
I've choosen to work with [jax](https://github.com/google/jax) and [numpyro](https://github.com/pyro-ppl/numpyro) because they provide the inference tools (`HMC`, `SVI`) so that I can focus on model specificaton. F.e. it is possible to specify a (True) Bayesian Neural Network - HMC/NUTS and not Bayes by Backprob - in only ~30 lines of code.

```python
import jax.numpy as jnp
from numpyro import deterministic, plate, sample
from numpyro.distributions import Gamma, Normal


def BNN(X, y=None):

    N, k = X.shape
    D_H1 = 4
    D_H2 = 4

    # layer 1
    W1 = sample("W1", Normal(loc=jnp.zeros((k, D_H1)), scale=jnp.ones((k, D_H1))))
    b1 = sample("b1", Normal(loc=jnp.zeros(D_H1), scale=1.0))
    out1 = jnp.tanh(jnp.matmul(X, W1)) + b1

    # layer 2
    W2 = sample("W2", Normal(loc=jnp.zeros((D_H1, D_H2)), scale=jnp.ones((D_H1, D_H2))))
    b2 = sample("b2", Normal(loc=jnp.zeros(D_H2), scale=jnp.ones(D_H2)))
    out2 = jnp.tanh(jnp.matmul(out1, W2)) + b2

    # output layer
    W3 = sample("W3", Normal(loc=jnp.zeros((D_H2, 1)), scale=jnp.ones((D_H2, 1))))
    b3 = sample("b3", Normal(loc=jnp.zeros(1), scale=jnp.ones(1)))

    mean = deterministic("mean", jnp.matmul(out2, W3) + b3)
    prec_obs = sample("prec_obs", Gamma(3.0, 1.0))
    scale = 1.0 / jnp.sqrt(prec_obs)

    with plate("data", size=N, dim=-2):
        sample("y", Normal(loc=mean, scale=scale), obs=y)
```

![Bayesian Neural Net](./plots/BayesianDNN_2021_12_28_13_07.jpg)


## Reference
- [Hands-on Bayesian Neural Networks – a Tutorial
for Deep Learning Users](https://arxiv.org/pdf/2007.06823.pdf) Laurent Valentin Jospin, Hamid Laga, Farid Boussaid, Wray Buntine, Mohammed Bennamoun]

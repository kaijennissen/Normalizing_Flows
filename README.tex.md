# Normalizing Flows

![Real NVP Example](./plots/banana/real_nvp_banana.gif)
Example of a normalizing flow (Real NVP) learning the distribution on the left.

## Intro

> Normalizing flows operate by pushing a simple density through a series of transformations to produce a richer, potentially more multi-modal distribution.
> -- <cite>Papamakarios et al. 2021 </cite>

These transformations $T$ have to be bijective, differentiable with a differentible inverse $T^{-1}$ and a functional determinant $det(T^{-1})\neq 0$, in short T is a diffeomorphism (Note that in the NF literature the terms `Bijector` and `diffeomorphism` are used interchangably).

## Building a custom Bijector with distrax

We start with a linear map $T$ given by:

$T:\mathbb{R^{2}} \rightarrow \mathbb{R^{2}}, \left(\begin{array}{c}
 x_{1}\\
 x_{2}
 \end{array}\right) \mapsto \left(\begin{array}{cc}
 cos(\theta) &-sin(\theta)\\
 sin(\theta) & cos(\theta)
 \end{array}\right) \left(\begin{array}{c}
 x_{1}\\
 x_{2}
 \end{array}\right)
$

with inverse $T^{-1}$ :

$T^{-1}:\mathbb{R^{2}} \rightarrow \mathbb{R^{2}}, \left(\begin{array}{c}
 x_{1}\\
 x_{2}
 \end{array}\right) \mapsto \left(\begin{array}{cc}
 cos(\theta) &sin(\theta)\\
 -sin(\theta) & cos(\theta)
 \end{array}\right) \left(\begin{array}{c}
 x_{1}\\
 x_{2}
 \end{array}\right)
$

and functional determinant $det( T^{'} )=sin^{2}(\theta)+cos^{2}(\theta)=1$.

In [distrax](https://github.com/deepmind/distrax) we can construct the above map by subclassing the `Bijector` class.

```python
import distrax
import jax.numpy as jnp

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
```

Transforming an independent multivariate Gaussian distribution $X$ with the `OrthogonalProjection2D` for $\theta =\frac{\pi}{4}= 45^{\circ}$ yields a multivariate Gaussian distribution $Y$ which is no longer independent, as can be seen below:
![Rotation Bijector](./plots/rotation/rotation1.jpg)
Since the above bijector is linear we already knew that $cov(Y)=cov(TX)=T\Sigma T^{'}$ where $\Sigma=cov(X)$.

In the image below we chained shift, scale and the Orthogonal Projector.
On the left hand side the true distribution is depicted and on the right hand side the inferred ditribution using maximum likelihood for the shift parameter $a$, the scale parameter $b$ and the rotation parameter $\theta$:
![Rotation Bijector 2](./plots/rotation/rotation2.jpg)

## MAF & Real NVP & Glow

### samples from some toy densities (left) and sampels from the inferred distribution (right)

![MAF](./plots/banana/maf_banana.jpg)
![MAF](./plots/gaussian_blobs/maf_gaussian_blobs.jpg)
![MAF](./plots/energy1/maf_energy1.jpg)
![MAF](./plots/energy2/maf_energy2.jpg)
![MAF](./plots/energy3/maf_energy3.jpg)
![MAF](./plots/energy4/maf_energy4.jpg)

## Bernstein Flows

### Univariate

![Bernstein Flow](./plots/Bernstein_Flow.jpg)

### Independent Multivariate

![Bernstein Flow](./plots/MVN_3D.jpg)

## Multiplicative Normalizing Flows

## Reference

- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf) George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan
- [Deep transformation models: Tackling complex regression problems with neural network based transformation models](https://arxiv.org/pdf/2004.00464.pdf) Beate Sick, Torsten Hothorn, Oliver Dürr
- [Robust normalizing flows using Bernstein-type polynomials](https://arxiv.org/pdf/2102.03509.pdf) Sameera Ramasinghe, Kasun Fernando, Salman Khan, Nick Barnes
- [Building custom bijectors with Tensorflow Probability](https://romainlhardy.medium.com/building-custom-bijectors-with-tensorflow-probability-22241cb6a691)
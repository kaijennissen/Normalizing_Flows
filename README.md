# Normalizing Flows

![Real NVP Example](./plots/banana/real_nvp_banana.gif)
Example of a normalizing flow (Real NVP) learning the distribution on the left.

## Intro

> Normalizing flows operate by pushing a simple density through a series of transformations to produce a richer, potentially more multi-modal distribution.
> -- <cite>Papamakarios et al. 2021 </cite>

These transformations <img src="https://latex.codecogs.com/svg.image?T" title="T" /> have to be bijective, differentiable with a differentible inverse <img src="https://latex.codecogs.com/svg.image?T^{-1}" title="T" /> and a functional determinant<img src="https://latex.codecogs.com/svg.image?det(T^{-1})\neq&space;0" title="det(T^{-1})\neq 0" />, in short <img src="https://latex.codecogs.com/svg.image?T" title="T" /> is a diffeomorphism (Note that in the NF literature the terms `Bijector` and `diffeomorphism` are used interchangably).

## Building a custom Bijector with distrax

We start with a linear map <img src="https://latex.codecogs.com/svg.image?T" title="T" /> given by:

<img src="https://latex.codecogs.com/svg.image?T:\mathbb{R}^{2}&space;\rightarrow&space;\mathbb{R}^{2},&space;\left(\begin{array}{c}&space;x_{1}\\&space;x_{2}&space;\end{array}\right)&space;\mapsto&space;\left(\begin{array}{cc}&space;cos(\theta)&space;&-sin(\theta)\\&space;sin(\theta)&space;&&space;cos(\theta)&space;\end{array}\right)&space;\left(\begin{array}{c}&space;x_{1}\\&space;x_{2}&space;\end{array}\right)&space;&space;" title="T:\mathbb{R}^{2} \rightarrow \mathbb{R}^{2}, \left(\begin{array}{c} x_{1}\\ x_{2} \end{array}\right) \mapsto \left(\begin{array}{cc} cos(\theta) &-sin(\theta)\\ sin(\theta) & cos(\theta) \end{array}\right) \left(\begin{array}{c} x_{1}\\ x_{2} \end{array}\right) " />

with inverse <img src="https://latex.codecogs.com/svg.image?T^{-1}" title="T^{-1}" /> :

<img src="https://latex.codecogs.com/svg.image?T^{-1}:\mathbb{R}^{2}&space;\rightarrow&space;\mathbb{R}^{2},&space;\left(\begin{array}{c}&space;x_{1}\\&space;x_{2}&space;\end{array}\right)&space;\mapsto&space;\left(\begin{array}{cc}&space;cos(\theta)&space;&sin(\theta)\\&space;-sin(\theta)&space;&&space;cos(\theta)&space;\end{array}\right)&space;\left(\begin{array}{c}&space;x_{1}\\&space;x_{2}&space;\end{array}\right)" title="T^{-1}:\mathbb{R}^{2} \rightarrow \mathbb{R}^{2}, \left(\begin{array}{c} x_{1}\\ x_{2} \end{array}\right) \mapsto \left(\begin{array}{cc} cos(\theta) &sin(\theta)\\ -sin(\theta) & cos(\theta) \end{array}\right) \left(\begin{array}{c} x_{1}\\ x_{2} \end{array}\right)" />


and functional determinant <img src="https://latex.codecogs.com/svg.image?det(T^{'})=sin^{2}(\theta)&plus;cos^{2}(\theta)=1" title="det(T^{'})=sin^{2}(\theta)+cos^{2}(\theta)=1" />  .

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

Transforming an independent multivariate Gaussian distribution <img src="https://latex.codecogs.com/svg.image?X" title="\Sigma=cov(X)" /> with the `OrthogonalProjection2D` for <img src="https://latex.codecogs.com/svg.image?\theta=45^{\circ}" title="\theta=45^{\circ}" /> yields a multivariate Gaussian distribution <img src="https://latex.codecogs.com/svg.image?Y" title="T" /> which is no longer independent, as can be seen below:
![Rotation Bijector](./plots/rotation/rotation1.jpg)
Since the above bijector is linear we already knew that <img src="https://latex.codecogs.com/svg.image?cov(Y)=cov(TX)=T\Sigma&space;T^{'}" title="cov(Y)=cov(TX)=T\Sigma T^{'}" /> where <img src="https://latex.codecogs.com/svg.image?\Sigma=cov(X)" title="\Sigma=cov(X)" />.

In the image below we chained shift, scale and the Orthogonal Projector.
On the left hand side the true distribution is depicted and on the right hand side the inferred ditribution using maximum likelihood for the shift parameter $a$, the scale parameter $b$ and the rotation parameter $\theta$:
![Rotation Bijector 2](./plots/rotation/rotation2.jpg)

## MAF & Real NVP & Glow

### samples from toy densities (left) and the inferred distribution with MAF (middle) and Real NVP (right)

|                           True                            |                          MAF                          |                          Real NVP                          |
| :-------------------------------------------------------: | :---------------------------------------------------: | :--------------------------------------------------------: |
|         ![MAF](./plots/banana/banana_samples.jpg)         |         ![MAF](./plots/banana/maf_banana.jpg)         |         ![MAF](./plots/banana/real_nvp_banana.jpg)         |
| ![MAF](./plots/gaussian_blobs/gaussian_blobs_samples.jpg) | ![MAF](./plots/gaussian_blobs/maf_gaussian_blobs.jpg) | ![MAF](./plots/gaussian_blobs/real_nvp_gaussian_blobs.jpg) |
|        ![MAF](./plots/energy1/energy1_samples.jpg)        |        ![MAF](./plots/energy1/maf_energy1.jpg)        |        ![MAF](./plots/energy1/real_nvp_energy1.jpg)        |
|        ![MAF](./plots/energy2/energy2_samples.jpg)        |        ![MAF](./plots/energy2/maf_energy2.jpg)        |        ![MAF](./plots/energy2/real_nvp_energy2.jpg)        |
|        ![MAF](./plots/energy3/energy3_samples.jpg)        |        ![MAF](./plots/energy3/maf_energy3.jpg)        |        ![MAF](./plots/energy3/real_nvp_energy3.jpg)        |
|        ![MAF](./plots/energy4/energy4_samples.jpg)        |        ![MAF](./plots/energy4/maf_energy4.jpg)        |        ![MAF](./plots/energy4/real_nvp_energy4.jpg)        |

## Bernstein Flows

### Univariate

![Bernstein Flow](./plots/Bernstein_Flow.jpg)

### Independent Multivariate

![Bernstein Flow](./plots/MVN_3D.jpg)

## Multiplicative Normalizing Flows

## HINT: Hierarchical Invertible Neural Transport

## Reference

- [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf) George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan
- [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
- [Deep transformation models: Tackling complex regression problems with neural network based transformation models](https://arxiv.org/pdf/2004.00464.pdf) Beate Sick, Torsten Hothorn, Oliver Dürr
- [Robust normalizing flows using Bernstein-type polynomials](https://arxiv.org/pdf/2102.03509.pdf) Sameera Ramasinghe, Kasun Fernando, Salman Khan, Nick Barnes
- [Building custom bijectors with Tensorflow Probability](https://romainlhardy.medium.com/building-custom-bijectors-with-tensorflow-probability-22241cb6a691)

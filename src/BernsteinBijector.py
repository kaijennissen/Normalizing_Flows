import matplotlib.pyplot as plt
import numpy as np
from bernstein_flow.bijectors import BernsteinBijector
from bernstein_flow.distributions import BernsteinFlow


def plot_chained_bijectors(flow):
    chained_bijectors = flow.bijector.bijector.bijectors
    base_dist = flow.distribution
    cols = len(chained_bijectors) + 1
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))

    n = 200

    z_samples = np.linspace(-3, 3, n).astype(np.float32)
    log_probs = base_dist.log_prob(z_samples)

    ax[0].plot(z_samples, np.exp(log_probs))

    zz = z_samples[..., None]
    ildj = 0.0
    for i, (a, b) in enumerate(zip(ax[1:], chained_bijectors)):
        # we need to use the inverse here since we are going from z->y!
        z = b.inverse(zz)
        ildj += b.forward_log_det_jacobian(z, 1)
        a.plot(z, np.exp(log_probs + ildj))
        a.set_title(b.name.replace("_", " "))
        a.set_xlabel(f"$z_{i}$")
        a.set_ylabel(f"$p(z_{i+1})$")
        zz = z
    fig.tight_layout()


a = np.array([6.0], dtype=np.float32)
b = np.array([-3.0], dtype=np.float32)
theta = np.array([-2.0, 2.0, 3.0, -7.0, -7.0, -7.0, -7.0, -7.0, 7.0], dtype=np.float32)
alpha = np.array([0.2], dtype=np.float32)
beta = np.array([-2.50], dtype=np.float32)

flow = BernsteinFlow(np.concatenate([a, -b, theta, alpha, -beta]))
plot_chained_bijectors(flow)
plt.show()
yy = np.linspace(0, 1, 200, dtype=np.float32)
bs = flow.bijector  # .bijector.bijectors[2]
zz = bs.forward(yy)
plt.figure(figsize=(8, 8))
plt.plot(yy, zz)
plt.show()


theta = np.array(
    [
        -1.0744555,
        -0.6429366,
        -0.44160688,
        0.7950939,
        1.9249767,
        2.1408765,
        2.4256434,
        3.1641612,
        3.3939004,
    ],
    dtype=np.float32,
)
bs = BernsteinBijector(theta=theta)
yy = np.linspace(0, 1, 200, dtype=np.float32)
zz = bs.forward(yy)
# Prevent caching -> use spline interpolation
zi = zz + 1e-15 * np.random.random(zz.shape)
yyy = bs.inverse(zi)

plt.figure(figsize=(8, 8))
plt.plot(yy, zz, alpha=0.5)
plt.plot(yyy, zi, alpha=0.5)
plt.title("Tensorflow-Version")
plt.show()


print(
    f"The MSE of the interpolation in this example is {np.sum((np.squeeze(yyy)-yy)**2):.3e}$."
)

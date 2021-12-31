from bernstein_flow.bijectors import BernsteinBijector
from bernstein_flow.distributions import BernsteinFlow
import numpy as np
from bernstein_flow.util.visualization import plot_chained_bijectors
import matplotlib.pyplot as plt


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

import jax.numpy as jnp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


# Density 2
def half_circle_pdf(pos):
    x1 = pos[..., 0]
    x2 = pos[..., 1]
    theta = jnp.arctan2(x2, x1)
    r = jnp.sqrt(jnp.sum(pos ** 2, axis=-1))
    return jnp.exp(-0.5 * theta ** 2) * jnp.exp(-0.5 * ((r - 1) / 0.2) ** 2)


if __name__ == "__main__":

    x1 = jnp.linspace(-2, 2, 2000)
    x2 = jnp.linspace(-2, 2, 2000)

    X1, X2 = jnp.meshgrid(x1, x2)
    Z = half_circle_pdf(jnp.stack([X1, X2], axis=-1))

    cm = plt.cm.get_cmap("viridis")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, rstride=8, cstride=8, alpha=0.9, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="z", offset=-1.2, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="x", offset=3.3, cmap=cm)
    cset = ax.contourf(X1, X2, Z, zdir="y", offset=3, cmap=cm)
    ax.set_zlim(-1.2, jnp.max(Z) * 1.4)
    ax.set_xlabel(r"$x_{2}$")
    ax.set_ylabel(r"$x_{1}$")
    ax.set_zlabel(r"$f(x_{1},x_{2})$")
    ax.view_init(elev=17, azim=-135)
    ax.grid(False)
    fig.tight_layout()
    # plt.show()
    plt.savefig("./plots/half_circle_pdf_3D.jpg", dpi=600)
    plt.close()

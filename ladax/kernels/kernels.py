import jax.numpy as jnp
from flax import struct
from typing import Callable


def rbf_kernel_fun(x, x2, amplitude, lengthscale):
    """ Functional definition of an RBF kernel. """
    pwd_dists = (x[..., jnp.newaxis, :] - x2[..., jnp.newaxis, :, :]) / lengthscale
    kernel_matrix = jnp.exp(-.5 * jnp.sum(pwd_dists ** 2, axis=-1))
    return amplitude**2 * kernel_matrix


@struct.dataclass
class Kernel:
    kernel_fn: Callable = struct.field(pytree_node=False)

    def apply(self, x, x2):
        return self.kernel_fn(x, x2)

    def __call__(self, x, x2=None):
        x2 = x if x2 is None else x2
        return self.appyl(x, x2)

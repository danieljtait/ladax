from typing import Callable
import jax.numpy as jnp
import jax
from flax import linen as nn

from ladax.kernels import Kernel, SchurComplementKernel
from ladax import utils

from jax.random import PRNGKey
from jax.numpy import ndarray as Array, dtype as Dtype, shape as Shape


def rbf_kernel_fun(x, x2, amplitude, lengthscale):
    """ Functional definition of an RBF kernel. """
    pwd_dists = (x[..., jnp.newaxis, :] - x2[..., jnp.newaxis, :, :]) / lengthscale
    kernel_matrix = jnp.exp(-.5 * jnp.sum(pwd_dists ** 2, axis=-1))
    return amplitude**2 * kernel_matrix


class RBFKernelProvider(nn.Module):
    """ Provides an RBF kernel function.
    The role of a kernel provider is to handle initialisation, and
    parameter storage of a particular kernel function. Allowing
    functionally defined kernels to be slotted into more complex models
    built using the Flax functional api.
    """
    amplitude_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones
    length_scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones

    @nn.compact
    def __call__(self, index_points: jnp.ndarray) -> Callable:
        """
        Args:
            index_points: The nd-array of index points to the kernel. Only used for
              feature shape finding.
        Returns:
            rbf_kernel_fun: Callable kernel function.
        """
        amplitude = jax.nn.softplus(
            self.param('amplitude',
                       self.amplitude_init, (1,), )) + jnp.finfo(float).tiny

        length_scale = jax.nn.softplus(
            self.param('length_scale',
                       self.length_scale_init, (index_points.shape[-1],),
                       )) + jnp.finfo(float).tiny

        return Kernel(lambda x_, y_: rbf_kernel_fun(x_, y_, amplitude, length_scale))


class SchurComplementKernelProvider(nn.Module):
    """ Provides a schur complement kernel. """
    fixed_index_points: Array
    diag_shift: Array = jnp.zeros((1))

    @nn.compact
    def __call__(self, base_kernel_fun: Callable) -> SchurComplementKernel:
        """
        Args:
            kernel_fun:
        Returns:
        """
        # compute the "divisor-matrix"
        divisor_matrix = base_kernel_fun(
            self.fixed_index_points, self.fixed_index_points)

        divisor_matrix_cholesky = jnp.linalg.cholesky(
            utils.diag_shift(divisor_matrix, self.diag_shift))

        return SchurComplementKernel(base_kernel_fun, self.fixed_index_points, divisor_matrix_cholesky)

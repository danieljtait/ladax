from typing import Callable
import jax.numpy as jnp
import jax
from flax import nn

from .kernels import Kernel, SchurComplementKernel
from ladax import utils


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
    def apply(self,
              index_points: jnp.ndarray,
              amplitude_init: Callable = jax.nn.initializers.ones,
              length_scale_init: Callable = jax.nn.initializers.ones) -> Callable:
        """
        Args:
            index_points: The nd-array of index points to the kernel. Only used for
              feature shape finding.
            amplitude_init: initializer function for the amplitude parameter.
            length_scale_init: initializer function for the length-scale parameter.
        Returns:
            rbf_kernel_fun: Callable kernel function.
        """
        amplitude = jax.nn.softplus(
            self.param('amplitude',
                       (1,),
                       amplitude_init)) + jnp.finfo(float).tiny

        length_scale = jax.nn.softplus(
            self.param('length_scale',
                       (index_points.shape[-1],),
                       length_scale_init)) + jnp.finfo(float).tiny

        return Kernel(
            lambda x_, y_: rbf_kernel_fun(x_, y_, amplitude, length_scale))


class SchurComplementKernelProvider(nn.Module):
    """ Provides a schur complement kernel. """
    def apply(self,
              base_kernel_fun: Callable,
              fixed_index_points: jnp.ndarray,
              diag_shift: jnp.ndarray = jnp.zeros([1])) -> SchurComplementKernel:
        """
        Args:
            kernel_fun:
            fixed_index_points:
            diag_shift: Python `float`
        Returns:
        """
        # compute the "divisor-matrix"
        divisor_matrix = base_kernel_fun(
            fixed_index_points, fixed_index_points)

        divisor_matrix_cholesky = jnp.linalg.cholesky(
            utils.diag_shift(divisor_matrix, diag_shift))

        return SchurComplementKernel(base_kernel_fun,
                                     fixed_index_points,
                                     divisor_matrix_cholesky)

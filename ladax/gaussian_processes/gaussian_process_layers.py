import jax.numpy as jnp
import jax.scipy as jscipy

from flax import linen as nn
from ladax import kernels, utils
from typing import Callable
from .gaussian_processes import VariationalGaussianProcess
from .inducing_variables import InducingPointsVariable


class SVGPLayer(nn.Module):
    mean_fn: Callable
    kernel_fn: Callable
    inducing_variables: InducingPointsVariable
    jitter: float = 1e-4

    @nn.compact
    def __call__(self,index_points):
        """
        Args:
            index_points: the nd-array of index points of the GP model.
            mean_fn: callable mean function of the GP model.
            kernel_fn: callable kernel function.
            inducing_var: inducing variables `inducing_variables.InducingPointsVariable`.
            jitter: float `jitter` term to add to the diagonal of the covariance
              function before computing Cholesky decompositions.
        Returns:
            svgp: A sparse Variational GP model.
        """
        z = self.inducing_variables.locations
        qu = self.inducing_variables.variational_distribution
        qu_mean = qu.mean
        qu_scale = qu.scale

        # cholesky of the base kernel function applied at the inducing point
        # locations.
        kzz_chol = jnp.linalg.cholesky(
            utils.diag_shift(self.kernel_fn(z, z), self.jitter))

        if self.inducing_variables.whiten:
            qu_mean = kzz_chol @ qu_mean
            qu_scale = kzz_chol @ qu_scale

        z = self.inducing_variables.locations

        var_kern = kernels.VariationalKernel(self.kernel_fn, z, qu_scale)

        def var_mean(x_):
            kxz = self.kernel_fn(x_, z)
            dev = (qu_mean - self.mean_fn(z))[..., None]
            return (self.mean_fn(x_)[..., None]
                    + kxz @ jscipy.linalg.cho_solve(
                        (kzz_chol, True), dev))[..., 0]

        return VariationalGaussianProcess(index_points,
                                          var_mean,
                                          var_kern,
                                          self.jitter,
                                          self.inducing_variables)

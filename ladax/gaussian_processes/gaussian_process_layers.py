import jax.numpy as jnp
import jax.scipy as jscipy

from flax import nn
from ladax import kernels, utils

from .gaussian_processes import VariationalGaussianProcess


class SVGPLayer(nn.Module):
    def apply(self,
              index_points,
              mean_fn,
              kernel_fn,
              inducing_var,
              jitter=1e-4):
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
        z = inducing_var.locations
        qu = inducing_var.variational_distribution
        qu_mean = qu.mean
        qu_scale = qu.scale

        # cholesky of the base kernel function applied at the inducing point
        # locations.
        kzz_chol = jnp.linalg.cholesky(
            utils.diag_shift(kernel_fn(z, z), jitter))

        if inducing_var.whiten:
            qu_mean = kzz_chol @ qu_mean
            qu_scale = kzz_chol @ qu_scale

        z = inducing_var.locations

        var_kern = kernels.VariationalKernel(
            kernel_fn, z, qu_scale)

        def var_mean(x_):
            kxz = kernel_fn(x_, z)
            dev = (qu_mean - mean_fn(z))[..., None]
            return (mean_fn(x_)[..., None]
                    + kxz @ jscipy.linalg.cho_solve(
                        (kzz_chol, True), dev))[..., 0]

        return VariationalGaussianProcess(
            index_points, var_mean, var_kern, jitter, inducing_var)

import jax.numpy as jnp
import jax.scipy as jscipy
from flax import struct, nn

from typing import Any, Callable
from ladax import distributions, kernels, utils


def multivariate_gaussian_kl(q, p):
    """ KL-divergence between multivariate Gaussian distributions defined as
        ∫ N(q.mean, q.scale) log{ N(q.mean, q.scale) / N (p.mean, p.scale) }.
    Args:
        q: `MultivariateNormal` object
        p: `MultivariateNormal` object
    Returns:
        kl: Python `float` the KL-divergence between `q` and `p`.
    """
    m_diff = q.mean - p.mean
    return .5*(2*jnp.log(jnp.diag(p.scale)).sum() - 2*jnp.log(jnp.diag(q.scale)).sum()
               - q.mean.shape[-1]
               + jnp.trace(jscipy.linalg.cho_solve((p.scale, True), q.scale) @ q.scale.T)
               + jnp.sum(m_diff * jscipy.linalg.cho_solve((p.scale, True), m_diff)))


@struct.dataclass
class GaussianProcess:
    index_points: jnp.ndarray
    mean_function: Callable = struct.field(pytree_node=False)
    kernel_function: Callable = struct.field(pytree_node=False)
    jitter: float

    def marginal(self):
        kxx = self.kernel_function(self.index_points, self.index_points)
        chol_kxx = jnp.linalg.cholesky(utils.diag_shift(kxx, self.jitter))
        mean = self.mean_function(self.index_points)
        return distributions.MultivariateNormalTriL(mean, chol_kxx)

    def posterior_gp(self, y, x_new, observation_noise_variance, jitter=None):
        """ Returns a new GP conditional on y. """
        cond_kernel_fn, _ = kernels.SchurComplementKernelProvider.init(
            None,
            self.kernel_function,
            self.index_points,
            observation_noise_variance)

        marginal = self.marginal()

        def cond_mean_fn(x):
            k_xnew_x = self.kernel_function(x, self.index_points)
            return (self.mean_function(x)
                    + k_xnew_x @ jscipy.linalg.cho_solve(
                        (cond_kernel_fn.divisor_matrix_cholesky, True),
                        y - marginal.mean))

        jitter = jitter if jitter else self.jitter
        return GaussianProcess(x_new,
                               cond_mean_fn,
                               cond_kernel_fn,
                               jitter)


@struct.dataclass
class VariationalGaussianProcess(GaussianProcess):
    """ ToDo(dan): ugly `Any` typing to avoid circular dependency with GP
          inside of inducing_variables. Ideally break this by lifting
          variational GPs into their own module.
    """
    inducing_variable: Any

    def prior_kl(self):
        if self.inducing_variable.whiten:
            return self.prior_kl_whiten()
        else:
            qu = self.inducing_variable.variational_distribution
            pu = self.inducing_variable.prior_distribution
            return multivariate_gaussian_kl(qu, pu)

    def prior_kl_whiten(self):
        qu = self.inducing_variable.variational_distribution
        log_det = 2*jnp.sum(jnp.log(jnp.diag(qu.scale)))
        dim = qu.mean.shape[-1]
        return -.5*(log_det + 0.5*dim - jnp.sum(qu.mean**2) - jnp.sum(qu.scale**2))

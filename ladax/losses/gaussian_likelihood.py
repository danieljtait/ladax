import jax
import jax.numpy as jnp
from flax import nn
from ladax.gaussian_processes import VariationalGaussianProcess


class VariationalGaussianLikelihoodLoss(nn.Module):
    """ """
    def apply(self, y, vgp: VariationalGaussianProcess):
        obs_noise_scale = jax.nn.softplus(
            self.param('observation_noise_scale', (), jax.nn.initializers.ones))

        variational_distribution = vgp.marginal()
        qu_mean = variational_distribution.mean
        qu_scale = variational_distribution.scale

        # Expected value of iid gaussians under q(u)
        expected_gll_under_qu = -.5 * jnp.squeeze(
                    (jnp.sum(jnp.square(qu_mean - y))
                     + jnp.trace(qu_scale @ qu_scale.T))
                    / obs_noise_scale ** 2
                    + y.shape[-1] * jnp.log(obs_noise_scale ** 2)
                    + jnp.log(2 * jnp.pi))

        # flip sign to minimize the elbo
        return -expected_gll_under_qu

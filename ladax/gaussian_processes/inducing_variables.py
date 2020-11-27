import jax
import jax.numpy as jnp
from flax import struct, linen as nn
from jax import random
from ladax.distributions import MultivariateNormalDiag, MultivariateNormalTriL
from ladax.gaussian_processes import GaussianProcess
from typing import Union, Callable


@struct.dataclass
class InducingVariable:
    variational_distribution: MultivariateNormalTriL
    prior_distribution: MultivariateNormalTriL


@struct.dataclass
class InducingPointsVariable(InducingVariable):
    locations: jnp.ndarray
    whiten: bool = False


class InducingPointsProvider(nn.Module):
    """ Handles parameterisation of an inducing points variable. """
    kernel_fn: Callable
    num_inducing_points: int
    inducing_locations_init: Union[Callable, None] = None
    fixed_locations: bool = False
    whiten: bool = False
    jitter: float = 1e-4
    dtype: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, index_points: jnp.ndarray) -> InducingPointsVariable:
        """
        Args:
            index_points: the nd-array of index points of the GP model.
        Returns:
            inducing_var: inducing variables `inducing_variables.InducingPointsVariable`
        """
        n_features = index_points.shape[-1]
        z_shape = (self.num_inducing_points, n_features)
        if self.inducing_locations_init is None:
            inducing_locations_init = lambda key, shape: random.normal(key, z_shape)
        else:
            inducing_locations_init = self.inducing_locations_init

        if self.fixed_locations:
            _default_key = random.PRNGKey(0)
            z = inducing_locations_init(_default_key, z_shape)
        else:
            z = self.param('locations',
                           inducing_locations_init,
                           (self.num_inducing_points, n_features), )

        qu_mean = self.param('mean',
                             lambda key, shape: jax.nn.initializers.zeros(
                                 key, z_shape[0], dtype=self.dtype),
                             (self.num_inducing_points, ))

        qu_scale = self.param(
            'scale',
            lambda key, shape: jnp.eye(self.num_inducing_points, dtype=self.dtype),
            (self.num_inducing_points, self.num_inducing_points))

        if self.whiten:
            prior = MultivariateNormalDiag(mean=jnp.zeros(index_points.shape[-1]),
                                           scale_diag=jnp.ones(index_points.shape[-2]))

        else:
            prior = GaussianProcess(z,
                                    lambda x_: jnp.zeros(x_.shape[:-1]),
                                    self.kernel_fn,
                                    self.jitter).marginal()

        return InducingPointsVariable(variational_distribution=MultivariateNormalTriL(qu_mean, jnp.tril(qu_scale)),
                                      prior_distribution=prior,
                                      locations=z,
                                      whiten=self.whiten)

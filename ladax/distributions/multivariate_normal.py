import jax.numpy as jnp
import jax.scipy as jscipy
import abc
from jax import random
from flax import struct


@struct.dataclass
class MultivariateNormal:

    @abc.abstractmethod
    def log_prob(self, x):
        pass


@struct.dataclass
class MultivariateNormalDiag(MultivariateNormal):
    mean: jnp.ndarray
    scale_diag: jnp.ndarray

    def log_prob(self, x):
        return jnp.sum(
        jscipy.stats.norm.logpdf(
            x, loc=self.mean, scale=self.scale_diag))

    def sample(self, key, shape=()):
        return random.normal(key, shape=shape) * self.scale_diag + self.mean


@struct.dataclass
class MultivariateNormalTriL(MultivariateNormal):
    mean: jnp.ndarray
    scale: jnp.ndarray

    def log_prob(self, x):
        dim = x.shape[-1]
        dev = x - self.mean
        maha = jnp.sum(dev *
                       jscipy.linalg.cho_solve((self.scale, True), dev))
        log_2_pi = jnp.log(2 * jnp.pi)
        log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(self.scale)))
        return -0.5 * (dim * log_2_pi + log_det_cov + maha)

    def sample(self, key, shape=()):
        full_shape = shape + self.mean.shape
        std_normals = random.normal(key, full_shape)
        return jnp.tensordot(std_normals, self.scale, [-1, 1]) + self.mean

    @property
    def covariance(self):
        return self.scale @ self.scale.T


@struct.dataclass
class MultivariateNormalFull(MultivariateNormal):
    mean: jnp.ndarray
    covariance: jnp.ndarray

    def log_prob(self, x):
        scale = jnp.linalg.cholesky(self.covariance)
        dim = x.shape[-1]
        dev = x - self.mean
        maha = jnp.sum(dev *
                       jscipy.linalg.cho_solve((scale, True), dev))
        log_2_pi = jnp.log(2 * jnp.pi)
        log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(scale)))
        return -0.5 * (dim * log_2_pi + log_det_cov + maha)

    def sample(self, key, shape=()):
        return random.multivariate_normal(
            key, self.mean, self.covariance, shape)

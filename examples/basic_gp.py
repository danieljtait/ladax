from jax.config import config; config.update("jax_enable_x64", True)

from absl import app
from absl import flags
from absl import logging

import jax
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

from typing import Callable, Tuple, Dict, Union
from flax import linen as nn
from flax import optim
import scipy as oscipy

from ladax import kernels, distributions, gaussian_processes, utils

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'plot', default=False,
    help=('Plot the results.', ))

flags.DEFINE_integer(
    'n_index_points', default=25,
    help=('Number of index points to generate. ', ))

flags.DEFINE_float(
    'learning_rate', default=0.001,
    help=('The learning rate for the momentum optimizer.', ))

flags.DEFINE_integer(
    'num_epochs', default=100,
    help=('Number of training epochs.', ))


class MarginalObservationModel(nn.Module):
    """ The observation model p(y|x, {hyper par}) = ∫p(y,f|x)df where f(x) ~ GP(m(x), k(x, x')). """
    @nn.compact
    def __call__(self, pf: distributions.MultivariateNormalTriL) -> distributions.MultivariateNormalFull:
        """ Applys the marginal observation model of the conditional
        Args:
            pf: distribution of the latent GP to be marginalised over,
              a `distribution.MultivariateNormal` object.
        Returns:
            py: the marginalised distribution of the observations, a
              `distributions.MultivariateNormal` object.
        """
        obs_noise_scale = jax.nn.softplus(
            self.param('observation_noise_scale',
                       jax.nn.initializers.ones, ()))

        covariance = pf.scale @ pf.scale.T
        covariance = utils.diag_shift(covariance, obs_noise_scale**2)

        return distributions.MultivariateNormalFull(pf.mean, covariance)


class GaussianProcessLayer(nn.Module):
    """ Provides a Gaussian process.
    """
    kernel_fn: Callable
    mean_fn: Union[Callable, None] = None
    jitter: float = 1e-4

    @nn.compact
    def __call__(self, index_points: jnp.ndarray):
        """
        Args:
            index_points: the nd-array of index points of the GP model
            kernel_fn: callable kernel function.
            mean_fn: callable mean function of the GP model.
              (default: `None` is equivalent to lambda x: jnp.zeros(x.shape[:-1]))
            jitter: float `jitter` term to add to the diagonal of the covariance
              function before computing downstream Cholesky decompositions.
        Returns:
            p: `distributions.MultivariateNormalTriL` object.
        """
        if self.mean_fn is None:
            mean_fn = lambda x: jnp.zeros(x.shape[:-1], dtype=index_points.dtype)
        else:
            mean_fn = self.mean_fn

        return gaussian_processes.GaussianProcess(index_points, mean_fn, self.kernel_fn, self.jitter)


class GPModel(nn.Module):
    """ Model for i.i.d noise observations from a GP with
    RBF kernel. """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> distributions.MultivariateNormalFull:
        """
        Args:
            x: the nd-array of index points of the GP model.
            dtype: the data-type of the computation (default: float64)
        Returns:
            py_x: Distribution of the observations at the index points.
        """
        kern_fn = kernels.RBFKernelProvider(name='kernel_fn')(x)
        mean_fn = lambda x: nn.Dense(features=1, name='linear_mean_fn')(x)[..., 0]
        gp_x = GaussianProcessLayer(kern_fn, mean_fn, name='gp_layer')(x)
        py_x = MarginalObservationModel(name='observation_model')(gp_x.marginal())
        return py_x


def get_datasets(sim_key: random.PRNGKey, true_obs_noise_scale: float = 0.5) -> Tuple[dict, dict]:
    """ Generate the datasets. """
    index_points = jnp.linspace(-3., 3., 25)[..., jnp.newaxis]
    y = (-0.5 + .33 * index_points[:, 0] +
         + jnp.sin(index_points[:, 0])
         + true_obs_noise_scale * random.normal(sim_key, index_points.shape[:-1]))

    test_index_points = jnp.linspace(-3., 3., 100)[:, jnp.newaxis]

    train_ds = {'index_points': index_points, 'y': y}
    test_ds = {'index_points': test_index_points,
               'y': -0.5 + .33 * test_index_points[:, 0] + jnp.sin(test_index_points[:, 0])}
    return train_ds, test_ds


def get_initial_params(key):
    init_batch = jnp.ones((FLAGS.n_index_points, 1))
    initial_variables = GPModel().init(key, init_batch)
    return initial_variables["params"]


def create_optimizer(params):
    optimizer_def = optim.Momentum(learning_rate=FLAGS.learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


@jax.jit
def train_step(optimizer, batch):
    """Train for a single step. """

    def loss_fn(params: dict) -> float:
        """ This is clumsier than the usual FLAX loss_fn. """
        py = GPModel().apply({'params': params}, batch['index_points'])
        return -py.log_prob(batch['y'])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = {'loss': loss}
    return optimizer, metrics


def train_epoch(optimizer, train_ds, epoch):
    """ Train for a single epoch. """
    optimizer, epoch_metrics = train_step(optimizer, train_ds)
    epoch_metrics_np = jax.device_get(epoch_metrics)

    logging.info('train epoch: %d, loss: %.4f',
                 epoch,
                 epoch_metrics_np['loss'])

    return optimizer, epoch_metrics_np


def train(train_ds) -> Dict:
    """ Complete training of the GP-Model.
    Args:
        train_ds: Python `dict` with entries `index_points` and `y`.
    Returns:
        trained_params: The parameters of the trained model.
    """
    rng = random.PRNGKey(0)

    # initialise the model parameters
    params = get_initial_params(rng)

    # get the optimizer
    optimizer = create_optimizer(params)

    for epoch in range(1, FLAGS.num_epochs + 1):
        optimizer, metrics = train_epoch(optimizer, train_ds, epoch)

    return optimizer.target


def main(_):
    train_ds, test_ds = get_datasets(random.PRNGKey(123))
    trained_params = train(train_ds)

    if FLAGS.plot:
        import matplotlib.pyplot as plt

        obs_noise_scale = jax.nn.softplus(trained_params['observation_model']['observation_noise_scale'])

        def learned_kernel_fn(x1, x2):
            return kernels.RBFKernelProvider().apply({"params": trained_params['kernel_fn']}, x1)(x1, x2)

        def learned_mean_fn(x):
            return nn.Dense(features=1).apply({"params": trained_params['linear_mean_fn']}, x)[:, 0]

        # prior GP model at learned model parameters
        fitted_gp = gaussian_processes.GaussianProcess(
            train_ds['index_points'],
            learned_mean_fn,
            learned_kernel_fn, 1e-4)

        posterior_gp = fitted_gp.posterior_gp(
                train_ds['y'],
                test_ds['index_points'],
                obs_noise_scale**2)

        pred_f_mean = posterior_gp.mean_function(test_ds['index_points'])
        pred_f_var = jnp.diag(
            posterior_gp.kernel_function(test_ds['index_points'], test_ds['index_points']))

        fig, ax = plt.subplots()
        ax.fill_between(test_ds['index_points'][:, 0],
                        pred_f_mean - 2*jnp.sqrt(pred_f_var),
                        pred_f_mean + 2*jnp.sqrt(pred_f_var), alpha=0.5)

        ax.plot(test_ds['index_points'][:, 0], posterior_gp.mean_function(test_ds['index_points']), '-')
        ax.plot(train_ds['index_points'], train_ds['y'], 'ks')

        plt.show()


if __name__ == '__main__':
    app.run(main)

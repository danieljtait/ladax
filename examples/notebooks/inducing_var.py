from jax.config import config;

config.update("jax_enable_x64", True)

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
from flax import nn, optim
from jax import random

import ladax
from ladax import kernels
from ladax.gaussian_processes import inducing_variables
from ladax import gaussian_processes
from ladax import likelihoods

import matplotlib.pyplot as plt
import numpy as onp

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_inducing_points', default=10,
    help=('Number of inducing points.', ))


def get_datasets():
    index_points = onp.random.uniform(size=20).reshape((10, 2))
    y = onp.sin(2*onp.pi*index_points[:, 0]) * onp.cos(onp.pi*index_points[:, 1])
    return {'index_points': index_points, 'y': y}


def main(_):
    train_ds = get_datasets()
    rng = random.PRNGKey(123)
    kernel_fn, _ = kernels.RBFKernelProvider.init(rng, train_ds['index_points'])

    z, _ = inducing_variables.InducingPointsProvider.init(
        rng, train_ds['index_points'], kernel_fn,
        num_inducing_points=FLAGS.num_inducing_points)

    def create_model(key):
        inducing_var_kwargs = {'num_inducing_points': FLAGS.num_inducing_points}

        clz = ladax.models.svgp_factory(kernels.RBFKernelProvider,
                                        inducing_variables.InducingPointsProvider,
                                        inducing_variable_kwargs=inducing_var_kwargs)
        _, params = clz.init_by_shape(key, [([10, 2], jnp.float32)], )
        return nn.Model(clz, params)

    model = create_model(rng)
    vgp = model(train_ds['index_points'])

    post_gp = vgp.posterior_gp(train_ds['y'], train_ds['index_points'], observation_noise_variance=0., jitter=1e-4)
    xx = onp.linspace(0., 1., 10)
    yy = onp.linspace(0., 1., 10)
    X = onp.column_stack(list(map(onp.ravel, onp.meshgrid(xx, yy))))
    m = post_gp.mean_function(X)

    m = m.reshape(10, 10)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, m)
    plt.show()

if __name__ == '__main__':
    app.run(main)
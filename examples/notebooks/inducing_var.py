from jax.config import config;

config.update("jax_enable_x64", True)

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp
from flax import nn, optim
from jax import random

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

    return {'index_points': index_points, }


def main(_):
    train_ds = get_datasets()
    rng = random.PRNGKey(123)
    kernel_fn, _ = kernels.RBFKernelProvider.init(rng, train_ds['index_points'])

    z, _ = inducing_variables.InducingPointsProvider.init(
        rng, train_ds['index_points'], kernel_fn,
        num_inducing_points=FLAGS.num_inducing_points)

    print(z.locations)
    fig, ax = plt.subplots()
    ax.plot(*z.locations.T, '+')
    plt.show()


if __name__ == '__main__':
    app.run(main)
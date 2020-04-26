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
from ladax import kernels, losses
from ladax.gaussian_processes import inducing_variables
from ladax import gaussian_processes
from ladax import likelihoods

import matplotlib.pyplot as plt
import numpy as onp
from collections import namedtuple

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_inducing_points', default=10,
    help=('Number of inducing points.', ))

flags.DEFINE_float(
    'learning_rate', default=0.001,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'num_epochs', default=1000,
    help=('Number of training epochs.'))

flags.DEFINE_bool(
    'plot', default=False,
    help=('Plot the results.',))


LossAndModel = namedtuple('LossAndModel', 'loss model')


def create_model(key):
    inducing_var_kwargs = {'num_inducing_points': FLAGS.num_inducing_points}

    clz = ladax.models.svgp_factory(kernels.RBFKernelProvider,
                                    inducing_variables.InducingPointsProvider,
                                    inducing_variable_kwargs=inducing_var_kwargs)
    _, params = clz.init_by_shape(key, [([10, 2], jnp.float32)], )
    return nn.Model(clz, params)


def create_loss(rng, model, train_ds):
    dist = model(train_ds['index_points'])
    _, params = losses.VariationalGaussianLikelihoodLoss.init(
        rng, train_ds['y'], dist)
    return nn.Model(losses.VariationalGaussianLikelihoodLoss, params)


def create_optimizer(loss_and_model, learning_rate, beta):
    optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
    optimizer = optimizer_def.create(loss_and_model)
    return optimizer


def get_datasets():
    index_points = onp.random.uniform(size=20).reshape((10, 2))
    y = (onp.sin(2*onp.pi*index_points[:, 0])
         * onp.cos(onp.pi*index_points[:, 1]))
    return {'index_points': index_points, 'y': y}


@jax.jit
def train_step(optimizer, batch):
    """Train for a single step."""

    def loss_fn(loss_and_model):
        vgp = loss_and_model.model(batch['index_points'])
        return loss_and_model.loss(batch['y'], vgp) + vgp.prior_kl()

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = {'loss': loss}
    return optimizer, metrics


def train_epoch(optimizer, train_ds, epoch):
    """Train for a single epoch."""
    optimizer, epoch_metrics = train_step(optimizer, train_ds)
    epoch_metrics_np = jax.device_get(epoch_metrics)

    logging.info('train epoch: %d, loss: %.4f',
                 epoch,
                 epoch_metrics_np['loss'])

    return optimizer, epoch_metrics_np


def train(train_ds):
    rng = random.PRNGKey(0)
    num_epochs = FLAGS.num_epochs

    model = create_model(rng)
    loss = create_loss(rng, model, train_ds)
    loss_and_model = LossAndModel(loss, model)
    optimizer = create_optimizer(
        loss_and_model, FLAGS.learning_rate, FLAGS.momentum)

    for epoch in range(1, num_epochs + 1):
        optimizer, metrics = train_epoch(
            optimizer, train_ds, epoch)

    return optimizer.target.model


def main(_):
    train_ds = get_datasets()

    trained_model = train(train_ds)

    #model = create_model(rng)
    #loss = create_loss(rng, model, train_ds)
    #loss_and_model = LossAndModel(loss, model)
    #vgp = model(train_ds['index_points'])
    #print(loss(train_ds['y'], vgp))

    if FLAGS.plot:
        vgp = model(train_ds['index_points'])
        post_gp = vgp.posterior_gp(train_ds['y'], train_ds['index_points'], observation_noise_variance=0., jitter=1e-4)
        xx = onp.linspace(0., 1., 10)
        yy = onp.linspace(0., 1., 10)
        x, y = onp.meshgrid(xx, yy)
        X = onp.column_stack((x.ravel(), y.ravel()))
        m = post_gp.mean_function(X)

        fig, ax = plt.subplots()
        ax.plot(*train_ds['index_points'].T, 'ks')
        ax.contourf(x, y, m.reshape(x.shape), alpha=0.5)
        plt.show()


if __name__ == '__main__':
    app.run(main)
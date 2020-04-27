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


import matplotlib.pyplot as plt
import numpy as onp
from collections import namedtuple

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_inducing_points', default=7,
    help=('Number of inducing points.', ))

flags.DEFINE_float(
    'learning_rate', default=0.0001,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_integer(
    'num_epochs', default=100,
    help=('Number of training epochs.'))

flags.DEFINE_bool(
    'plot', default=False,
    help=('Plot the results.',))

flags.DEFINE_integer(
    'num_training_points', default=50,
    help=('The number of training points.', ))


LossAndModel = namedtuple('LossAndModel', 'loss model')


def create_model(key):

    def inducing_loc_init(key, shape):
        return random.uniform(key, shape, minval=-1., maxval=1.)

    inducing_var_kwargs = {
        'num_inducing_points': FLAGS.num_inducing_points,
        'inducing_locations_init': jax.nn.initializers.normal(stddev=1.),
        'fixed_locations': False,
        'whiten': False}

    svgp_layer_kwargs = {'jitter': 1.0e-4}

    clz = ladax.models.svgp_factory(kernels.RBFKernelProvider,
                                    inducing_variables.InducingPointsProvider,
                                    inducing_variable_kwargs=inducing_var_kwargs,
                                    svgp_layer_kwargs=svgp_layer_kwargs)
    vgp, params = clz.init_by_shape(
        key, [([FLAGS.num_training_points, 2], jnp.float32)], )

    return nn.Model(clz, params)


def create_loss(rng, model, train_ds):

    loss_clz = losses.VariationalGaussianLikelihoodLoss

    dist = model(train_ds['index_points'])
    _, params = loss_clz.init(rng, train_ds['y'], dist)
    return nn.Model(loss_clz, params)


def create_optimizer(loss_and_model, learning_rate, beta):
    optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
    optimizer = optimizer_def.create(loss_and_model)
    return optimizer


def true_function(x, y):
    return jnp.cos(x) * y


def get_datasets():
    onp.random.seed(123)
    index_points = onp.random.normal(size=100).reshape((50, 2))
    y = true_function(*index_points.T)
    y += .1*onp.random.randn(*y.shape)
    return {'index_points': index_points, 'y': y}


@jax.jit
def train_step(optimizer, batch):
    """Train for a single step."""

    def loss_fn(loss_and_model):
        vgp = loss_and_model.model(batch['index_points'])
        negell = loss_and_model.loss(batch['y'], vgp)
        loss = vgp.prior_kl() + negell
        return loss, vgp.inducing_variable.locations

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, z), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = {'loss': loss,
               'z': z}
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

    z = [model.params['1']['locations'], ]

    loss_and_model = LossAndModel(loss, model)
    optimizer = create_optimizer(
        loss_and_model, FLAGS.learning_rate, FLAGS.momentum)

    for epoch in range(1, num_epochs + 1):
        optimizer, metrics = train_epoch(
            optimizer, train_ds, epoch)
        z.append(metrics['z'])

    return optimizer.target, z


def main(_):
    train_ds = get_datasets()
    trained_model_and_loss, z = train(train_ds)

    trained_model = trained_model_and_loss.model
    trained_loss = trained_model_and_loss.loss
    obs_noise_scale = jax.nn.softplus(trained_loss.params['observation_noise_scale'])
    print(obs_noise_scale)

    if FLAGS.plot:
        from matplotlib import animation

        vgp = trained_model(train_ds['index_points'])
        post_gp = vgp.posterior_gp(
            train_ds['y'],
            train_ds['index_points'],
            obs_noise_scale**2,
            jitter=1e-4)

        xmin = jnp.min(vgp.inducing_variable.locations[:, 0])
        xmax = jnp.max(vgp.inducing_variable.locations[:, 0])
        ymin = jnp.min(vgp.inducing_variable.locations[:, 1])
        ymax = jnp.max(vgp.inducing_variable.locations[:, 1])

        xmin, xmax = (-3., 3.)
        ymin, ymax = (-3., 3.)

        xx = onp.linspace(xmin, xmax, 50)
        yy = onp.linspace(ymin, ymax, 50)
        x, y = onp.meshgrid(xx, yy)
        X = onp.column_stack((x.ravel(), y.ravel()))
        m = post_gp.mean_function(X)

        true_y = true_function(*X.T)

        vmin = min(m.min(), true_y.min())
        vmax = max(m.max(), true_y.max())

        fig, axes = plt.subplots(ncols=2)
        axes[0].contourf(x, y, m.reshape(x.shape),
                         alpha=0.5, vmin=vmin, vmax=vmax)
        axes[0].plot(*z[0].T, 'C1o')
        axes[0].plot(*z[-1].T, 'C1^')

        z = jnp.array(z)

        lines = axes[0].plot(z[..., 0], z[..., 1], '-')

        axes[0].plot(*train_ds['index_points'].T, 'k+')
        axes[1].contourf(x, y, true_y.reshape(x.shape),
                         alpha=0.5, vmin=vmin, vmax=vmax)

        for ax in axes:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

        # animated plot
        fig, ax = plt.subplots(figsize=(4, 4))

        zcol = 'k'

        ax.plot(z[..., 0], z[..., 1], '-', color=zcol)

        lines = []
        for _ in range(FLAGS.num_inducing_points):
            line, = ax.plot([], [], 'D', color=zcol)
            lines.append(line)

        ax.contour(x, y, m.reshape(x.shape), 'k-', levels=10)
        ax.plot(*train_ds['index_points'].T, 'k+', label='obs. index points')

        # initialization function: plot the background of each frame
        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        nskip = 10
        def animate(i):
            #ax.set_title('Epoch {}'.format(i))
            for k in range(FLAGS.num_inducing_points):
                x = [z[i*nskip, k, 0], ]
                y = [z[i*nskip, k, 1], ]
                lines[k].set_data(x, y)
            return lines

        nframes = (FLAGS.num_epochs + 1) // nskip

        ax.plot([], [], 'D', color=zcol, label='inducing point locs.')
        ax.legend()

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=nframes,
                                       interval=20, blit=True,
                                       repeat=True)
        anim.save('inducing_point_locs.gif', writer='imagemagick', fps=30)

        plt.show()


if __name__ == '__main__':
    app.run(main)
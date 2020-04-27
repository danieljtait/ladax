import jax.numpy as jnp
from flax.nn import Module
from ladax.gaussian_processes import SVGPLayer


def svgp_factory(kernel_provider,
                 inducing_variable_provider,
                 mean_fn=None,
                 kernel_fn_kwargs=None,
                 inducing_variable_kwargs=None,
                 svgp_layer_kwargs=None):

    mean_fn = mean_fn if mean_fn else lambda x: jnp.zeros(x.shape[-2], x.dtype)
    kernel_fn_kwargs = {} if kernel_fn_kwargs is None else kernel_fn_kwargs
    inducing_variable_kwargs = {} if inducing_variable_kwargs is None else inducing_variable_kwargs
    svgp_layer_kwargs = {} if svgp_layer_kwargs is None else svgp_layer_kwargs

    class SVGP(Module):
        def apply(self, x):
            kernel_fn = kernel_provider(x, **kernel_fn_kwargs)
            inducing_var = inducing_variable_provider(x, kernel_fn, **inducing_variable_kwargs)
            vgp = SVGPLayer(x, mean_fn, kernel_fn, inducing_var, **svgp_layer_kwargs)
            return vgp

    return SVGP

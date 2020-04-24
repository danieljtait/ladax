import jax.numpy as jnp
from typing import Union
import jax.ops as ops


def diag_shift(mat: jnp.ndarray,
                val: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """ Shifts the diagonal of mat by val. """
    return ops.index_update(
        mat,
        jnp.diag_indices(mat.shape[-1], len(mat.shape)),
        jnp.diag(mat) + val)

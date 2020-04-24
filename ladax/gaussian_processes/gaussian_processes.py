import jax.numpy as jnp
import jax.scipy as jscipy
from flax import struct, nn

from typing import Any, Callable


@struct.dataclass
class GaussianProcess:
    index_points: jnp.ndarray
    mean_function: Callable = struct.field(pytree_node=False)
    kernel_function: Callable = struct.field(pytree_node=False)
    jitter: float


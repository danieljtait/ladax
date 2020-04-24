import jax.numpy as jnp
import jax.scipy as jscipy
from flax import struct
from typing import Callable


@struct.dataclass
class Kernel:
    kernel_fn: Callable = struct.field(pytree_node=False)

    def apply(self, x, x2):
        return self.kernel_fn(x, x2)

    def __call__(self, x, x2=None):
        x2 = x if x2 is None else x2
        return self.apply(x, x2)


@struct.dataclass
class SchurComplementKernel(Kernel):
    fixed_inputs: jnp.ndarray
    divisor_matrix_cholesky: jnp.ndarray

    def apply(self, x1, x2):
        k12 = self.kernel_fn(x1, x2)
        k1z = self.kernel_fn(x1, self.fixed_inputs)
        kz2 = self.kernel_fn(self.fixed_inputs, x2)
        return (k12
                - k1z @ jscipy.linalg.cho_solve(
                    (self.divisor_matrix_cholesky, True), kz2))

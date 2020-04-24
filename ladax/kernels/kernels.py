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


@struct.dataclass
class VariationalKernel(Kernel):
    fixed_inputs: jnp.ndarray
    variational_scale: jnp.ndarray
    jitter: float = 1.0e-4

    def apply(self, x1, x2):
        z = self.fixed_inputs
        kxy = self.kernel_fn(x1, x2)
        kxz = self.kernel_fn(x1, z)
        kzy = self.kernel_fn(z, x2)
        kzz = self.kernel_fn(z, z)
        kzz_cholesky = jnp.linalg.cholesky(
            kzz + self.jitter * jnp.eye(z.shape[-2]))

        kzz_chol_qu_scale = jscipy.linalg.cho_solve(
            (kzz_cholesky, True), self.variational_scale)

        return (kxy
                - kxz @ jscipy.linalg.cho_solve((kzz_cholesky, True), kzy)
                + kxz @ (kzz_chol_qu_scale @ kzz_chol_qu_scale.T) @ kzy)

import jax.numpy as jnp
from jax import Array, vmap
from rotation import diff_rotate_volume_y

def diff_volume_sinogram(volume: Array, n_angles: int = 64) -> Array:
    angles = jnp.linspace(0, jnp.pi, n_angles)
    return vmap(lambda angle: jnp.sum(diff_rotate_volume_y(volume, angle), axis=0), in_axes=0, out_axes=-1)(angles)

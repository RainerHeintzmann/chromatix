import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax import Array

def diff_rotate(volume, angle):
    Nz, Ny, Nx  = volume.shape
    # Making grid
    grid = jnp.mgrid[-Nz // 2 : Nz // 2, -Ny // 2 : Ny // 2, -Nx // 2 : Nx // 2] + 1/2
    grid = jnp.transpose(grid, (1, 2, 3, 0))
    # adding homogeneous coordinates
    grid = jnp.concatenate([grid, jnp.ones((*grid.shape[:3], 1))], axis=-1)
    
    # making affine matrix
    M = jnp.eye(4)
    M = M @ Ryz(angle)
    
    # Deforming grid and resampling
    deformed_grid = (grid @ M.transpose())[..., :3]
    return resample(volume, deformed_grid)

def resample(vol: Array, grid: Array) -> jnp.ndarray:
    # assume original coordinates were centered, i.e. -N/2 -> N/2
    offset = jnp.array(vol.shape)[:, None] // 2 - 1/2
    resampled = map_coordinates(vol, grid.reshape(-1, 3).T + offset, order=1, mode='constant', cval=0.0)
    return resampled.reshape(grid.shape[:3])
    
def R_yaw(theta: float) -> jnp.ndarray:
    """Roll rotations."""
    R = jnp.zeros((4, 4))

    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)

    R = R.at[1, 1].set(1.0)
    R = R.at[3, 3].set(1.0)  # homogeneous
    R = R.at[0, 0].set(cos_t)
    R = R.at[2, 2].set(cos_t)
    R = R.at[0, 2].set(sin_t)
    R = R.at[2, 0].set(-sin_t)

    return R
    
def Ryz(theta: float) -> jnp.ndarray:
    """Roll rotations."""
    R = jnp.zeros((4, 4))

    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)

    R = R.at[1, 1].set(1.0)
    R = R.at[3, 3].set(1.0)  # homogeneous
    R = R.at[0, 0].set(cos_t)
    R = R.at[2, 2].set(cos_t)
    R = R.at[0, 2].set(sin_t)
    R = R.at[2, 0].set(-sin_t)

    return R
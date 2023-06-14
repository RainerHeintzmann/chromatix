
import numpy as np
from typing import NewType
from jax import Array
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

degrees = NewType("degrees", float)
radians = NewType("radians", float)

# %%
def rotation_y(theta: radians) -> jnp.ndarray:
    """ Generates rotation matrix around y."""
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

def volume_homogeneous_grid(volume: Array) -> Array:
    """Given a volume, generates a centred grid of homogeneous coordinates.
    Coordinates are placed along last dimension [z, y, x, 4]"""

    Nz, Ny, Nx = volume.shape
    z = np.linspace(-(Nz-1)/2, (Nz-1)/2, Nz)
    y = np.linspace(-(Ny-1)/2, (Ny-1)/2, Ny)
    x = np.linspace(-(Nx-1)/2, (Nx-1)/2, Nx)
    grid = np.stack(np.meshgrid(z, y, x, indexing='ij'), axis=-1)
    return jnp.concatenate([grid, jnp.ones((Nz, Ny, Nx, 1))], axis=-1)

def resample(volume: Array, sample_grid: Array) -> Array:
    # assume original coordinates were centered, i.e. -N/2 -> N/2
    offset = (jnp.array(volume.shape) - 1) / 2
    sample_locations = sample_grid.reshape(-1, 3).T + offset[:, None]
    resampled = map_coordinates(volume, sample_locations, order=1, mode='constant', cval=0.0)
    return resampled.reshape(sample_grid.shape[:3])

def diff_rotate_volume_y(volume: Array, angle: radians) -> Array:
    sample_grid = (volume_homogeneous_grid(volume) @ rotation_y(angle).T)[..., :3]
    return resample(volume, sample_grid)
# %%

if __name__ == "__main__":# %%
    from skimage.transform import rotate, resize
    from skimage.data import shepp_logan_phantom
    data = resize(shepp_logan_phantom(), (128, 128))
    # %% We stack this image along y to get our 3D phantom.
    data = np.repeat(np.expand_dims(data, 1), axis=1, repeats=data.shape[-1]) # [z, y, x]

    # %% Baseline, by scikit 
    def rotate_volume_y(volume: np.ndarray, angle: degrees) -> np.ndarray:
        return np.stack([rotate(volume[:, idx, :], angle, order=1, resize=False) for idx in range(volume.shape[1])], axis=1)

    def test_jax_rotation(volume: np.ndarray, n_random_angles: int):
        angles = np.random.uniform(0, 360, n_random_angles)
        for angle in angles:
            jax_result = diff_rotate_volume_y(volume, angle / 180 * np.pi)
            baseline = rotate_volume_y(volume, angle)
            assert jnp.allclose(baseline, jax_result, atol=1e-5)

    def test_jax_rotation_single(image: np.ndarray, n_random_angles: int):
        angles = np.random.uniform(0, 360, n_random_angles)
        for angle in angles:
            jax_result = diff_rotate_volume_y(image, angle / 180 * np.pi)
            baseline = rotate_volume_y(image, angle)
            assert jnp.allclose(baseline, jax_result, atol=1e-5)
            
    test_jax_rotation(data, 5)
    test_jax_rotation_single(data[:, [10]], 5)

# %%

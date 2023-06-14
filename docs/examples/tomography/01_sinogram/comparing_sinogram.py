#%%
from skimage.transform import resize, radon
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, vmap, jit
import jax.numpy as jnp
import sys
sys.path.append("../")
from rotation import diff_rotate_volume_y

# %% Getting data ready
data = resize(shepp_logan_phantom(), (128, 128))
plt.imshow(data)

# %% We stack this image along y to get our 3D phantom.
data = np.repeat(np.expand_dims(data, 1), axis=1, repeats=data.shape[-1]) # [z, y, x]

# %% Getting baseline sinogram 
def volume_sinogram(volume: np.ndarray, n_angles: int = 64) -> np.ndarray:
    # Input shape of volume [z, y, x], rotate around y
    # Output shape [y, projection_x, n_angles]
    angles = np.linspace(0, 180, n_angles)
    return np.stack([radon(volume[:, idx], angles) for idx in range(volume.shape[1])], axis=0)

# %%
baseline = volume_sinogram(data)
# %%
def diff_volume_sinogram(volume: Array, n_angles: int = 64) -> Array:
    angles = np.linspace(0, jnp.pi, n_angles)
    return vmap(lambda angle: jnp.sum(diff_rotate_volume_y(volume, angle), axis=0), in_axes=0, out_axes=-1)(angles)


# %%
jax_result = jit(diff_volume_sinogram)(data)[:, ::-1]
# %%
jnp.allclose(jax_result, baseline) # won't be the same due to weird rotation of skimage.
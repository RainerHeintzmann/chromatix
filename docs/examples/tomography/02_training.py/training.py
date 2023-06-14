#%%
from skimage.transform import resize
from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, value_and_grad
import jax.numpy as jnp
import sys
sys.path.append("../")
from sinogram import diff_volume_sinogram
import optax

# %% Getting data ready
data = resize(shepp_logan_phantom(), (128, 128))
plt.imshow(data)

# %% We stack this image along y to get our 3D phantom.
volume = np.repeat(np.expand_dims(data, 1), axis=1, repeats=data.shape[-1]) # [z, y, x]
print(volume.shape)

# %% Making data
sinogram = diff_volume_sinogram(volume)

# %%
def loss_fn(volume, sinogram):
    return jnp.mean((sinogram - diff_volume_sinogram(volume))**2)

assert loss_fn(volume, sinogram) == 0
# %% Having a look at the grads
grad_fn =  jit(value_and_grad(loss_fn))
loss, grads = grad_fn(jnp.zeros_like(volume), sinogram)
plt.imshow(grads[:, 0, :]) # that's pretty good! 
plt.colorbar() 

# %% Training
params = jnp.zeros_like(volume)
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

n_iterations = 1000
for iteration in range(n_iterations):
    loss, grads = grad_fn(params, sinogram)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    if iteration % 10 == 0:
        print(f"Iteration {iteration} - {loss:.2f}")

# %%

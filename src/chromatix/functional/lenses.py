import jax.numpy as jnp
from chex import Array
from typing import Optional
from ..field import Field
from .pupils import circular_pupil
from chromatix.functional.convenience import optical_fft
from ..utils import l2_sq_norm, _broadcast_1d_to_grid

__all__ = ["thin_lens", "ff_lens", "df_lens"]


def thin_lens(field: Field, f: float, n: float, NA: Optional[float] = None) -> Field:
    """
    Applies a thin lens placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` directly after the lens.
    """
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * l2_sq_norm(field.grid) / L**2

    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    return field * jnp.exp(1j * phase)


def ff_lens(
    field: Field,
    f: float,
    n: float,
    NA: Optional[float] = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    # Pupil
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    if inverse:
        # if inverse, propagate over negative distance
        f = -f
    return optical_fft(field, f, n)


def df_lens(
    field: Field,
    d: float,
    f: float,
    n: float,
    NA: Optional[float] = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``d`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        d: Distance from the incoming ``Field`` to the lens.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    if inverse:
        # if inverse, propagate over negative distance
        f = -d
        d = -f
    field = optical_fft(field, f, n)

    # Phase factor due to distance d from lens
    L = jnp.sqrt(jnp.complex64(field.spectrum * f / n))  # Lengthscale L
    phase = jnp.pi * (1 - d / f) * l2_sq_norm(field.grid) / jnp.abs(L) ** 2
    return field * jnp.exp(1j * phase)


def mla(
    field: Field,
    f: float,
    ns: float,
    nu: float,
    D: float
) -> Field:
    """
    Applies a tiled phase mask created by a microlens array (MLA).

    Args:
        field: The ``Field`` to which the MLA will be applied.
        f: Focal length of the lens.
        ns: Number of lenslets in each linear dimension
        nu: Number of pixels per lenslet in each linear dimension.
        D: Diameter of each lenslet.

    Returns:
        The ``Field`` propagated a distance ``f`` after the MLA.
    """
    # TODO: adjust function so that they input field is not modified, and a new field is returned.
    # TODO: allow vector fields to be input
    # TODO: add an optional aperture
    # TODO: account for incompatible input values
    wavelength = field._spectrum
    (_, M, N, _, _) = field.shape
    lenslet_len = int(M / ns)
    nu = int(nu)
    assert lenslet_len == nu, "Lenslet length must be equal to nu"
    # Create a local optical system that retards the phase for each lenslet
    dx = field._dx[0].item()
    k = 2 * jnp.pi / wavelength
    x = jnp.linspace(-D/2 + dx/2, D/2 - dx/2, lenslet_len)
    X, Y = jnp.meshgrid(x, x)
    phase = jnp.exp(-1j * k / (2 * f) * (jnp.square(X) + jnp.square(Y)))

    # Apply the phase mask to each lenslet
    phase_dim_ext = jnp.expand_dims(phase, axis=(0, -2, -1))
    for s in jnp.arange(ns):
        for t in jnp.arange(ns):
            cell = field.u[:, s*lenslet_len:(s+1)*lenslet_len, t*lenslet_len:(t+1)*lenslet_len]
            field.u.at[:, s*lenslet_len:(s+1)*lenslet_len, t*lenslet_len:(t+1)*lenslet_len, :, :].set(cell * phase_dim_ext)
    return field


def microlens_array(
    field: Field,
    centers: Array,
    fs: Array,
    ns: Array,
    NAs: Array,
) -> Field:
    phase = jnp.zeros(field.spatial_shape)
    # Not sure the scenario where indexing is not necessary -Geneva
    focal_length_indexing_needed = True
    if focal_length_indexing_needed:
        len_focal_lengths = fs.shape[0]
    else:
        len_focal_lengths = fs.shape
    for i in range(len_focal_lengths):
        L_sq = field.spectrum * fs[i] / ns[i]
        radius = NAs[i] * fs[i] / ns[i]
        squared_distance = l2_sq_norm(
            field.grid - _broadcast_1d_to_grid(jnp.squeeze(centers[:, i]), field.ndim)
        )
        mask = jnp.squeeze(squared_distance) < radius**2
        phase += mask * jnp.squeeze(squared_distance) / L_sq
    phase *= -jnp.pi
    return field * jnp.exp(1j * phase)

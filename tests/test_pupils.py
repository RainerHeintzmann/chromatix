from chromatix.functional.pupils import circular_pupil, square_pupil
from chromatix.functional.sources import empty_field, plane_wave
import jax.numpy as jnp


def test_circular_pupil():
    field = empty_field((512, 512), 1.0, 0.532, 1.0)
    field = plane_wave(field, 1.0)

    w = 100
    field = circular_pupil(field, w)
    A_pupil = jnp.pi * (w / 2) ** 2
    A_field = 512**2

    assert jnp.allclose(A_pupil / A_field, field.power, atol=1e-3)


def test_square_pupil():
    field = empty_field((512, 512), 1.0, 0.532, 1.0)
    field = plane_wave(field, 1.0)

    w = 100
    field = square_pupil(field, w)
    A_pupil = w**2
    A_field = 512**2

    assert jnp.allclose(A_pupil / A_field, field.power, atol=1e-3)

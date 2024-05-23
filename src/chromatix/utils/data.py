import numpy as np
from typing import Tuple

try:
    import cv2

    USE_CV2 = True
except ModuleNotFoundError:
    USE_CV2 = False


def siemens_star(num_pixels: int = 512, num_spokes: int = 32) -> np.ndarray:
    """
    Generates a 2D Siemens star image of shape ``(num_pixels num_pixels)``.

    Number of spokes in the star can be controlled with ``num_spokes``. Spokes
    will alternate between black and white (0.0 and 1.0).
    """
    X, Y = np.mgrid[0:num_pixels, num_pixels:0:-1] - (num_pixels / 2.0)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(X, Y) + np.pi
    S = np.zeros_like(R)
    for spoke in range(num_spokes):
        in_spoke = (theta >= ((spoke) * 2 * np.pi / num_spokes)) & (
            theta <= ((spoke + 1) * 2 * np.pi / num_spokes)
        )
        if not spoke % 2:
            S[in_spoke] = 1.0
    S *= R < (num_pixels / 2.0)
    return S


if USE_CV2:

    def draw_disks(
        shape: Tuple[int, int], coordinates: np.ndarray, radius: int, color: int = 255
    ) -> np.ndarray:
        """
        Create a grayscale image with disks drawn at each provided coordinate.

        Args:
            image_size: The desired image size as (height, width).
            coordinates: A list of (x, y) coordinates where disks should be
                drawn.
            radius: The radius of the disks.
            color: An optional intensity for the disks (0-255).

        Returns:
            numpy.ndarray: The resulting grayscale image with disks drawn at
                the specified coordinates.
        """
        image = np.zeros(shape, dtype=np.uint8)
        for coord in coordinates:
            cv2.circle(image, (coord[0], coord[1]), radius, color, -1)
        return image

else:

    def draw_disks(
        shape: Tuple[int, int], coordinates: np.ndarray, radius: int, color: int = 255
    ) -> np.ndarray:
        """
        Create a grayscale image with disks drawn at each provided coordinate.

        Args:
            image_size: The desired image size as (height, width).
            coordinates: A list of (x, y) coordinates where disks should be
                drawn.
            radius: The radius of the disks.
            color: An optional intensity for the disks (0-255).

        Returns:
            numpy.ndarray: The resulting grayscale image with disks drawn at
                the specified coordinates.
        """
        image = np.zeros([s + radius * 2 for s in shape], dtype=np.uint8)
        _samples = np.linspace(-radius, radius, num=radius * 2, dtype=np.float32)
        circle = color * np.uint8(
            np.sum(np.array(np.meshgrid(_samples, _samples)) ** 2, axis=0)
            <= radius**2
        )
        for c in coordinates:
            slices = (slice(c[0], c[0] + radius * 2), slice(c[1], c[1] + radius * 2))
            image[slices] = image[slices] | circle
        image = image[radius : radius + shape[0], radius : radius + shape[1]]
        return image


class RandDiskGenerator:  # TODO avoid overlapping disks
    def __init__(
        self,
        N: int,
        num_points: int,
        radius: int,
        shape: Tuple[int, int],
        z_range: Tuple[int, int],
    ):
        """
        Create a dataset of random 3D coordinates and their associated image.
        Each generated sample consists of an array of x y z coordinates with
        shape: (n_points 3), accompanied by a 3D image with shape `shape`.
        The last dimension of `shape` represents the z axis, if it exists. The
        number of planes will be inferred from the `shape` argument. This is
        meant for TensorFlow and PyTorch data loaders that support generators.

        Args:
            N: Number of samples in the dataset. Avoid large N as the coordinates
                on each epoch are pre-stored for speed. On each new epoch the
                samples are randomized again (new random coordinates are generated)
                therefore you can easily use small `N` to avoid memory issues.
            num_points: Number of points in each sample. For 3D samples these samples
                will be randomly split between the planes.
            radius: Radius of the disks to be drawn on each plane.
            shape: Shape of the output image. Dimensions are [h w n_z] where n_z is
                the number of planes in 3D. For 2D samples use z=1.
            z_range: list
                Minimum and Maximum values for z values [min, max]. The returned
                coordinates are [x y z] and this parameter determines the range of
                the z coordinates.
        """

        assert (
            len(shape) == 3
        ), "Shape must specify three dimensions, shape parameter is: {}".format(
            len(shape)
        )
        self.N = N
        self.radius = radius
        self.shape = shape
        self.z_range = z_range
        self.num_points = num_points
        self.num_planes = shape[-1]
        self.reset()

    def reset(self):
        """
        Generate all the random coordinates. This is called when generator
        is instantiated or the last sample in the generator is reached.
        """
        self.y = np.random.randint(
            low=self.radius,
            high=self.shape[0] - self.radius,
            size=(self.N, self.num_points),
        )
        self.x = np.random.randint(
            low=self.radius,
            high=self.shape[1] - self.radius,
            size=(self.N, self.num_points),
        )

        if self.num_planes > 1:
            self.z_indices = np.random.randint(
                low=0, high=self.num_planes, size=(self.N, self.num_points)
            )
            self.z_values = np.random.rand(self.N, self.num_planes) * (
                self.z_range[1] - self.z_range[0]
            )
            self.z_values += self.z_range[0]
            self.z_values.sort(axis=1)
            self.z = np.zeros_like(self.x).astype(np.float32)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Get a new sample.

        Args:
            idx: Index of the current sample that needs to be generated.

        Returns:
            numpy.ndarray
                A (num_points 2) or (num_points 3) array containing the
                    coordinates.
            numpy.ndarray
                2D or 3D Image corresponding to the coordinates.
        """
        coords = np.array([self.x[idx], self.y[idx]]).T

        if self.num_planes > 1:
            canvas = np.zeros(self.shape)
            for i in range(self.num_planes):
                canvas[:, :, i] = draw_disks(
                    self.shape[:-1],
                    coords[self.z_indices[idx] == i, :],
                    self.radius,
                    color=255,
                )  # TODO add weight
                print(self.z_values[idx, i])
                self.z[idx, self.z_indices[idx] == i] = self.z_values[idx, i]
            return (
                np.array([self.x[idx], self.y[idx], self.z[idx]]).T,
                canvas,
            )  # TODO add weight

        else:
            image = draw_disks(self.shape[:-1], coords, self.radius, color=255)[
                ..., None
            ]  # TODO add weight
            return coords, image

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a new sample. Automatically iterates through samples of coordinates
        with every call. Will cause the random coordinates to be regenerated
        when the last sample is reached.

        Returns:
            numpy.ndarray
                A (num_points 2) or (num_points 3) array containing the
                    coordinates.
            numpy.ndarray
                2D or 3D Image corresponding to the coordinates.
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.reset()

def min_hari_phantom(shape: Tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape)
    # dots
    num_dots = 50
    vmin = 500
    vmax = 4000
    values = rng.integers(low=vmin, high=vmax, size=num_dots)
    z = rng.integers(low=0, high=shape[0], size=num_dots)
    y = rng.integers(low=0, high=shape[1], size=num_dots)
    x = rng.integers(low=0, high=shape[2], size=num_dots)
    volume[z, y, x] = values
    # lines
    num_lines = 150
    vmin = 200
    vmax = 1500
    values = rng.integers(low=vmin, high=vmax, size=num_lines)
    max_length = np.round(0.8 * shape[2])
    lengths = rng.integers(low=1, high=max_length, size=num_lines)
    theta = rng.uniform(low=0, high=2 * np.pi, size=num_lines)
    alpha = rng.uniform(low=0, high=2 * np.pi, size=num_lines)
    start_z = rng.integers(low=0, high=shape[0], size=num_lines)
    start_y = rng.integers(low=0, high=shape[1], size=num_lines)
    start_x = rng.integers(low=0, high=shape[2], size=num_lines)
    for i in range(num_lines):
        length_yx = np.abs(lengths[i] * np.sin(alpha[i]))
        length_zx = np.abs(lengths[i] * np.cos(alpha[i]))
        end_z = np.minimum(np.maximum(np.round(start_z[i] + length_zx), 0), shape[0] - 1)
        end_y = np.minimum(np.maximum(np.round(start_y[i] + length_yx * np.sin(theta[i])), 0), shape[1] - 1)
        end_x = np.minimum(np.maximum(np.round(start_x[i] + length_yx * np.cos(theta[i])), 0), shape[2] - 1)
        volume[
            np.linspace(start_z[i], end_z, num=lengths[i], dtype=int),
            np.linspace(start_y[i], end_y, num=lengths[i], dtype=int),
            np.linspace(start_x[i], end_x, num=lengths[i], dtype=int),
        ] = values[i]
    # horizontal lines
    num_lines = 30
    vmin = 200
    vmax = 1500
    values = rng.integers(low=vmin, high=vmax, size=num_lines)
    max_length = np.round(0.8 * shape[2])
    lengths = rng.integers(low=1, high=max_length, size=num_lines)
    theta = rng.uniform(low=0, high=2 * np.pi, size=num_lines)
    alpha = np.pi / 2
    start_z = rng.integers(low=0, high=shape[0], size=num_lines)
    start_y = rng.integers(low=0, high=shape[1], size=num_lines)
    start_x = rng.integers(low=0, high=shape[2], size=num_lines)
    for i in range(num_lines):
        length_yx = np.abs(lengths[i] * np.sin(alpha))
        length_zx = np.abs(lengths[i] * np.cos(alpha))
        end_z = np.minimum(np.maximum(np.round(start_z[i] + length_zx), 0), shape[0] - 1)
        end_y = np.minimum(np.maximum(np.round(start_y[i] + length_yx * np.sin(theta[i])), 0), shape[1] - 1)
        end_x = np.minimum(np.maximum(np.round(start_x[i] + length_yx * np.cos(theta[i])), 0), shape[2] - 1)
        volume[
            np.linspace(start_z[i], end_z, num=lengths[i], dtype=int),
            np.linspace(start_y[i], end_y, num=lengths[i], dtype=int),
            np.linspace(start_x[i], end_x, num=lengths[i], dtype=int),
        ] = values[i]
    # horizontal lines
    num_lines = 30
    vmin = 200
    vmax = 1500
    values = rng.integers(low=vmin, high=vmax, size=num_lines)
    max_length = np.round(0.8 * shape[2])
    lengths = rng.integers(low=1, high=max_length, size=num_lines)
    theta = np.pi / 2
    alpha = rng.uniform(low=0, high=2 * np.pi, size=num_lines)
    start_z = rng.integers(low=0, high=shape[0], size=num_lines)
    start_y = rng.integers(low=0, high=shape[1], size=num_lines)
    start_x = rng.integers(low=0, high=shape[2], size=num_lines)
    for i in range(num_lines):
        length_yx = np.abs(lengths[i] * np.sin(alpha[i]))
        length_zx = np.abs(lengths[i] * np.cos(alpha[i]))
        end_z = np.minimum(np.maximum(np.round(start_z[i] + length_zx), 0), shape[0] - 1)
        end_y = np.minimum(np.maximum(np.round(start_y[i] + length_yx * np.sin(theta)), 0), shape[1] - 1)
        end_x = np.minimum(np.maximum(np.round(start_x[i] + length_yx * np.cos(theta)), 0), shape[2] - 1)
        volume[
            np.linspace(start_z[i], end_z, num=lengths[i], dtype=int),
            np.linspace(start_y[i], end_y, num=lengths[i], dtype=int),
            np.linspace(start_x[i], end_x, num=lengths[i], dtype=int),
        ] = values[i]
    # balls
    num_balls = 100
    vmin = 100
    vmax = 800
    vmid = np.round((vmin + vmax) / 2)
    radii = rng.random(num_balls)
    max_radius = 6
    values = np.where(radii <= 0.5, rng.integers(low=vmid, high=vmax, size=num_balls), rng.integers(low=vmin, high=vmid, size=num_balls))
    radii *= max_radius
    z = rng.integers(low=max_radius + 1, high=shape[0] - max_radius, size=num_balls)
    y = rng.integers(low=max_radius + 1, high=shape[1] - max_radius, size=num_balls)
    x = rng.integers(low=max_radius + 1, high=shape[2] - max_radius, size=num_balls)
    for i in range(num_balls):
        radius = radii[i]
        length = int(2 * np.round(radius) + 1)
        Z, Y, X = np.meshgrid(
            np.linspace(-radius, radius, num=length),
            np.linspace(-radius, radius, num=length),
            np.linspace(-radius, radius, num=length),
            indexing="ij"
        )
        Z += 0.5
        Y += 0.5
        X += 0.5
        ball = Z**2 + Y**2 + X**2
        ball = (ball <= radius**2) * values[i]
        volume[
            z[i] - (length - 1) // 2:z[i] + (length + 1) // 2,
            y[i] - (length - 1) // 2:y[i] + (length + 1) // 2,
            x[i] - (length - 1) // 2:x[i] + (length + 1) // 2,
        ] += ball
    # shells
    num_shells = 100
    vmin = 100
    vmax = 1000
    max_radius = 12
    shell_thickness = 1
    radii = rng.random(num_shells)
    radii *= max_radius
    values = rng.integers(low=vmin, high=vmax, size=num_shells)
    z = rng.integers(low=max_radius + 1, high=shape[0] - max_radius, size=num_shells)
    y = rng.integers(low=max_radius + 1, high=shape[1] - max_radius, size=num_shells)
    x = rng.integers(low=max_radius + 1, high=shape[2] - max_radius, size=num_shells)
    for i in range(num_shells):
        radius = radii[i]
        length = int(2 * np.round(radius) + 1)
        Z, Y, X = np.meshgrid(
            np.linspace(-radius, radius, num=length),
            np.linspace(-radius, radius, num=length),
            np.linspace(-radius, radius, num=length),
            indexing="ij"
        )
        Z += 0.5
        Y += 0.5
        X += 0.5
        shell = Z**2 + Y**2 + X**2
        shell = ((shell < radius**2) & (shell >= (radius - shell_thickness)**2)) * values[i]
        volume[
            z[i] - (length - 1) // 2:z[i] + (length + 1) // 2,
            y[i] - (length - 1) // 2:y[i] + (length + 1) // 2,
            x[i] - (length - 1) // 2:x[i] + (length + 1) // 2,
        ] += shell
    # rings
    num_rings = 100
    vmin = 200
    vmax = 1500
    max_radius = 12
    ring_thickness = 1
    radii = rng.random(num_rings)
    radii *= max_radius
    values = rng.integers(low=vmin, high=vmax, size=num_rings)
    z = rng.integers(low=max_radius + 1, high=shape[0] - max_radius, size=num_rings)
    y = rng.integers(low=max_radius + 1, high=shape[1] - max_radius, size=num_rings)
    x = rng.integers(low=max_radius + 1, high=shape[2] - max_radius, size=num_rings)
    for i in range(num_rings):
        radius = radii[i]
        length = int(2 * np.round(radius) + 1)
        Y, X = np.meshgrid(
            np.linspace(-radius, radius, num=length),
            np.linspace(-radius, radius, num=length),
            indexing="ij"
        )
        Y += 0.5
        X += 0.5
        ring = Y**2 + X**2
        ring = ((ring < radius**2) & (ring >= (radius - ring_thickness)**2)) * values[i]
        volume[
            z[i],
            y[i] - (length - 1) // 2:y[i] + (length + 1) // 2,
            x[i] - (length - 1) // 2:x[i] + (length + 1) // 2,
        ] += ring
    return volume

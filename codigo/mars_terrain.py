"""Procedural Mars terrain: heightmap, impact craters, dune field and scattered rocks.

Pure numpy, no MuJoCo import, deterministic per seed (every random stream is an independent
``np.random.default_rng([seed, stream])``; no global state is touched).

Conventions
-----------
* The terrain is a square of side ``extent_m`` centred on the origin, x and y in
  ``[-extent_m / 2, +extent_m / 2]``, z up, all lengths in meters.
* ``heights[row, col]``: the row runs along +y and the column along +x, which is exactly the
  memory layout of a MuJoCo ``hfield``.
* A smooth, gentle spawn zone of radius ``SPAWN_RADIUS_M`` around the origin is reserved for the
  future rover; the terrain blends smoothly into the full relief outside of it.
* The border of the map is tapered to exactly zero (separable smoothstep, so there is no diagonal
  crease at the corners). A second, coarse "far terrain" heightfield continues the world to the
  horizon: its relief fades in from z = 0 at the border, and it sinks well below the playable map
  so it never shows through craters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SPAWN_RADIUS_M = 6.0
SPAWN_BLEND_M = 14.0  # distance over which the relief fades in outside the spawn zone
EDGE_TAPER_M = 24.0  # width of the border band that fades to z = 0

# Random stream ids (keeps the layers independent of each other).
_STREAM_ROLLING, _STREAM_CRATERS, _STREAM_DUNES, _STREAM_ROCKS, _STREAM_ROUGH = range(5)
_STREAM_DUNE_WARP = 5
_STREAM_FAR = 6

CRATER_RADIUS_MIN_M = 2.0
CRATER_RADIUS_MAX_M = 15.0

FAR_EXTENT_M = 4000.0  # side of the far terrain that continues the world to the horizon
FAR_CELL_M = 10.0
FAR_FADE_IN_M = 150.0  # distance from the map border over which the far relief fades in
FAR_HOLE_DEPTH_M = -10.0  # far surface under the playable map
FAR_LEVEL_M = -0.05  # far surface just below the inner border: no z-fighting at the map corners

ROCK_SEMI_AXIS_MIN_M = 0.04
ROCK_SEMI_AXIS_MAX_M = 0.5


@dataclass(frozen=True)
class Crater:
    x: float
    y: float
    radius: float
    depth: float


@dataclass(frozen=True)
class DuneField:
    x: float
    y: float
    radius: float
    wind_angle: float  # direction the wind blows towards, radians from +x
    wavelength: float  # crest-to-crest distance along the wind
    amplitude: float  # crest height above trough, meters


@dataclass(frozen=True)
class Rock:
    """Ellipsoid rock: centre (x, y, z), semi-axes (sx, sy, sz) in meters, yaw in radians.

    ``pitch`` / ``roll`` (radians, small) tilt it after the yaw and ``shade`` scales its colour;
    the defaults leave the sandbox rocks untilted and unshaded.
    """

    x: float
    y: float
    z: float
    sx: float
    sy: float
    sz: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0
    shade: float = 1.0


# -- helpers -------------------------------------------------------------------------------


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream])


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _grid(size: int, extent_m: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-extent_m / 2, extent_m / 2, size)
    return np.meshgrid(axis, axis)  # X varies along columns, Y along rows


def _spectral_noise(
    rng: np.random.Generator,
    size: int,
    extent_m: float,
    wavelength_min_m: float,
    wavelength_max_m: float,
    exponent: float,
) -> np.ndarray:
    """Band-limited fractal noise with zero mean and unit standard deviation.

    Spectral synthesis: white noise shaped by ``k**-exponent`` and smooth low/high-pass
    rolloffs at the given wavelengths. Larger ``exponent`` means smoother terrain.
    """
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    freq = np.fft.fftfreq(size, d=extent_m / size)
    k = np.hypot(*np.meshgrid(freq, freq))
    with np.errstate(divide="ignore"):
        shape = np.where(k > 0, k**-exponent, 0.0)
    shape *= np.exp(-((k * wavelength_min_m) ** 2))  # low-pass: drop detail below the minimum
    shape *= 1.0 - np.exp(-((k * wavelength_max_m) ** 2))  # high-pass: drop very long waves
    field = np.fft.ifft2(spectrum * shape).real
    field -= field.mean()
    return field / field.std()


# -- layout planning (craters, dunes) ------------------------------------------------------


def plan_craters(extent_m: float, seed: int) -> tuple[Crater, ...]:
    """Deterministic crater layout: 6-12 craters, the last two small and overlapping the biggest."""
    rng = _rng(seed, _STREAM_CRATERS)
    half = extent_m / 2
    radius_max = min(CRATER_RADIUS_MAX_M, extent_m / 8)
    total = int(rng.integers(6, 13))
    dunes = plan_dune_field(extent_m, seed)
    craters: list[Crater] = []

    def new_crater(x: float, y: float, radius: float) -> Crater:
        return Crater(float(x), float(y), float(radius), float(radius * rng.uniform(0.15, 0.2)))

    def fits(x: float, y: float, radius: float) -> bool:
        margin = half - 0.6 * EDGE_TAPER_M
        if abs(x) + radius > margin or abs(y) + radius > margin:
            return False
        if np.hypot(x - dunes.x, y - dunes.y) < dunes.radius + radius:
            return False  # craters never overlap the dune field
        return bool(np.hypot(x, y) - radius > SPAWN_RADIUS_M + 8.0)

    for index in range(total - 2):
        # The first crater is always a big one; the others are log-uniform.
        if index == 0:
            radius = rng.uniform(0.5, 1.0) * radius_max
        else:
            radius = CRATER_RADIUS_MIN_M * (radius_max / CRATER_RADIUS_MIN_M) ** (rng.uniform() ** 0.8)
        while True:
            placed = False
            for _ in range(300):
                x, y = rng.uniform(-half, half, size=2)
                if fits(x, y, radius) and all(
                    np.hypot(x - c.x, y - c.y) > 1.1 * (radius + c.radius) for c in craters
                ):
                    craters.append(new_crater(x, y, radius))
                    placed = True
                    break
            if placed or radius <= CRATER_RADIUS_MIN_M:
                break
            radius = max(CRATER_RADIUS_MIN_M, 0.8 * radius)  # no room for this size: try a smaller one

    parent = max(craters, key=lambda c: c.radius)
    for _ in range(2):
        radius = rng.uniform(CRATER_RADIUS_MIN_M, 4.0)
        for _ in range(100):
            angle = rng.uniform(0.0, 2 * np.pi)
            dist = parent.radius * rng.uniform(0.8, 1.05)  # centred on the parent's rim
            x, y = parent.x + dist * np.cos(angle), parent.y + dist * np.sin(angle)
            if fits(x, y, radius):
                craters.append(new_crater(x, y, radius))
                break
    return tuple(craters)


def plan_dune_field(extent_m: float, seed: int) -> DuneField:
    """Deterministic dune field placement, away from the spawn zone and inside the map."""
    rng = _rng(seed, _STREAM_DUNES)
    half = extent_m / 2
    radius = 0.2 * extent_m
    wind_angle = float(rng.uniform(0.0, 2 * np.pi))
    wavelength = float(rng.uniform(7.0, 10.0))
    amplitude = float(rng.uniform(0.7, 1.2))
    for _ in range(300):
        angle = rng.uniform(0.0, 2 * np.pi)
        dist = rng.uniform(radius + SPAWN_RADIUS_M + 1.0, half)
        x, y = dist * np.cos(angle), dist * np.sin(angle)
        if abs(x) + radius <= half - 6.0 and abs(y) + radius <= half - 6.0:
            return DuneField(float(x), float(y), radius, wind_angle, wavelength, amplitude)
    corner = half - radius - 6.0  # fallback: a corner of the map (always valid)
    return DuneField(corner, corner, radius, wind_angle, wavelength, amplitude)


# -- relief layers -------------------------------------------------------------------------


def _crater_profile(s: np.ndarray, depth: float) -> np.ndarray:
    """Radial profile in units of crater radius: parabolic bowl, raised rim, faint ejecta blanket."""
    bowl = np.where(s < 1.0, -depth * (1.0 - s**2), 0.0)
    rim = 0.4 * depth * np.exp(-(((s - 1.0) / 0.22) ** 2))
    ejecta = 0.15 * depth * np.exp(-((np.maximum(s - 1.0, 0.0) / 1.6) ** 2))
    return bowl + rim + ejecta


def _crater_layer(X: np.ndarray, Y: np.ndarray, craters: Sequence[Crater]) -> np.ndarray:
    layer = np.zeros_like(X)
    for c in craters:
        s = np.hypot(X - c.x, Y - c.y) / c.radius
        near = s < 3.5
        layer[near] += _crater_profile(s[near], c.depth)
    return layer


def _dune_profile(phase: np.ndarray, lee_fraction: float = 0.3) -> np.ndarray:
    """Asymmetric periodic ridge in [0, 1]: gentle windward slope, steeper lee slope."""
    q = np.mod(phase, 1.0)
    rise = 1.0 - lee_fraction
    windward = 0.5 * (1.0 - np.cos(np.pi * q / rise))
    lee = 0.5 * (1.0 + np.cos(np.pi * (q - rise) / lee_fraction))
    return np.where(q < rise, windward, lee)


def _dune_layer(
    X: np.ndarray, Y: np.ndarray, dunes: DuneField, seed: int, size: int, extent_m: float
) -> np.ndarray:
    dx, dy = X - dunes.x, Y - dunes.y
    along = dx * np.cos(dunes.wind_angle) + dy * np.sin(dunes.wind_angle)
    across = -dx * np.sin(dunes.wind_angle) + dy * np.cos(dunes.wind_angle)
    warp = _spectral_noise(_rng(seed, _STREAM_DUNE_WARP), size, extent_m, 12.0, 50.0, 2.0)
    # Curved crests (barchan-like) plus domain warping so the ridges are not perfectly regular.
    phase = along / dunes.wavelength + 0.4 * (across / (0.9 * dunes.radius)) ** 2 + 0.2 * warp
    modulation = 0.7 + 0.3 * np.tanh(warp)
    mask = 1.0 - _smoothstep(0.55, 1.0, np.hypot(dx, dy) / dunes.radius)
    return mask * modulation * dunes.amplitude * (_dune_profile(phase) - 0.4)


def edge_weight(x: np.ndarray | float, y: np.ndarray | float, extent_m: float) -> np.ndarray | float:
    """Border taper: 1 in the interior, 0 on the border. Separable (product of two 1-D smoothsteps),
    so it has no diagonal crease at the corners, unlike a max-norm distance."""
    half = extent_m / 2
    return _smoothstep(0.0, EDGE_TAPER_M, half - np.abs(x)) * _smoothstep(0.0, EDGE_TAPER_M, half - np.abs(y))


def generate_heightmap(
    size: int = 257, extent_m: float = 120.0, seed: int = 0, include_craters: bool = True
) -> np.ndarray:
    """Square heightmap in meters (``heights[row=y, col=x]``), deterministic per ``seed``.

    ``include_craters=False`` leaves the craters out (same plain and dunes), handy for tests and debugging.
    """
    X, Y = _grid(size, extent_m)
    radius = np.hypot(X, Y)

    rolling = 0.6 * _spectral_noise(_rng(seed, _STREAM_ROLLING), size, extent_m, 8.0, 90.0, 1.5)
    rolling -= rolling[size // 2, size // 2]  # the spawn point sits at z = 0
    rough = 0.012 * _spectral_noise(_rng(seed, _STREAM_ROUGH), size, extent_m, 1.0, 4.0, 0.5)

    # 0 inside the spawn zone, 1 beyond spawn + blend: everything but a whisper of the rolling
    # plain is faded out near the origin so the rover starts on smooth ground.
    relief = _smoothstep(SPAWN_RADIUS_M, SPAWN_RADIUS_M + SPAWN_BLEND_M, radius)

    heights = rolling * (0.1 + 0.9 * relief)
    heights += relief * (
        _dune_layer(X, Y, plan_dune_field(extent_m, seed), seed, size, extent_m)
        + (_crater_layer(X, Y, plan_craters(extent_m, seed)) if include_craters else 0.0)
        + rough
    )

    heights *= edge_weight(X, Y, extent_m)
    return heights


def generate_far_heightmap(inner_extent_m: float = 120.0, seed: int = 0) -> np.ndarray:
    """Coarse heightmap of the world beyond the playable map (``FAR_EXTENT_M`` square, ``FAR_CELL_M`` cells).

    Gentle low-amplitude rolling relief that fades in from ``FAR_LEVEL_M`` (a hidden 5 cm lip below
    the inner border, z = 0) so the two heightfields join without a visible seam. Under the inner map the surface is
    ``FAR_HOLE_DEPTH_M`` deep (a linear ramp over one cell), hidden beneath the inner terrain.
    Node spacing divides typical map half-extents (60 m / 10 m), so the border falls on nodes.
    """
    size = int(round(FAR_EXTENT_M / FAR_CELL_M)) + 1
    X, Y = _grid(size, FAR_EXTENT_M)
    outside = np.maximum(np.abs(X), np.abs(Y)) - inner_extent_m / 2  # > 0 beyond the inner border
    relief = 1.8 * _spectral_noise(_rng(seed, _STREAM_FAR), size, FAR_EXTENT_M, 60.0, 1500.0, 1.6)
    relief *= _smoothstep(0.0, FAR_FADE_IN_M, outside)
    hole = np.clip(-outside / FAR_CELL_M, 0.0, 1.0)
    return relief + FAR_LEVEL_M + FAR_HOLE_DEPTH_M * hole


# -- sampling and rocks --------------------------------------------------------------------


def sample_height(heights: np.ndarray, extent_m: float, x: float, y: float) -> float:
    """Bilinear height at world (x, y); queries outside the map are clamped to the border."""
    n_rows, n_cols = heights.shape
    col = float(np.clip((x + extent_m / 2) / extent_m * (n_cols - 1), 0, n_cols - 1))
    row = float(np.clip((y + extent_m / 2) / extent_m * (n_rows - 1), 0, n_rows - 1))
    c0, r0 = min(int(col), n_cols - 2), min(int(row), n_rows - 2)
    fc, fr = col - c0, row - r0
    top = heights[r0, c0] * (1 - fc) + heights[r0, c0 + 1] * fc
    bottom = heights[r0 + 1, c0] * (1 - fc) + heights[r0 + 1, c0 + 1] * fc
    return float(top * (1 - fr) + bottom * fr)


def scatter_rocks(
    heights: np.ndarray,
    extent_m: float,
    seed: int,
    count: int = 150,
    craters: Sequence[Crater] | None = None,
) -> list[Rock]:
    """Scatter ``count`` ellipsoid rocks over the terrain.

    Sizes follow a truncated power law (many pebbles, few boulders), 60 % of the rocks cluster
    around crater rims, none lies in the spawn zone, and each is partly embedded in the ground.
    ``craters`` defaults to ``plan_craters(extent_m, seed)``, the layout ``generate_heightmap`` uses.
    """
    rng = _rng(seed, _STREAM_ROCKS)
    if craters is None:
        craters = plan_craters(extent_m, seed)
    weights = np.array([c.radius for c in craters], dtype=float)
    weights /= weights.sum()
    limit = extent_m / 2 - 1.0
    alpha = 2.0
    lo, hi = ROCK_SEMI_AXIS_MIN_M ** (1 - alpha), ROCK_SEMI_AXIS_MAX_M ** (1 - alpha)

    rocks: list[Rock] = []
    while len(rocks) < count:
        if len(craters) > 0 and rng.random() < 0.6:
            crater = craters[int(rng.choice(len(craters), p=weights))]
            angle = rng.uniform(0.0, 2 * np.pi)
            dist = crater.radius * rng.normal(1.05, 0.22)
            x, y = crater.x + dist * np.cos(angle), crater.y + dist * np.sin(angle)
        else:
            x, y = rng.uniform(-limit, limit, size=2)
        if abs(x) > limit or abs(y) > limit or np.hypot(x, y) < SPAWN_RADIUS_M + 1.0:
            continue

        s = (lo + rng.uniform() * (hi - lo)) ** (1.0 / (1 - alpha))
        sx, sy, sz = s, s * rng.uniform(0.6, 1.0), s * rng.uniform(0.4, 0.75)
        yaw = rng.uniform(0.0, 2 * np.pi)

        # Rest the rock on the lowest point under its footprint, then sink it a little.
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        footprint = [
            (x, y),
            (x + sx * cos_y, y + sx * sin_y),
            (x - sx * cos_y, y - sx * sin_y),
            (x - sy * sin_y, y + sy * cos_y),
            (x + sy * sin_y, y - sy * cos_y),
        ]
        lowest = min(sample_height(heights, extent_m, px, py) for px, py in footprint)
        rocks.append(Rock(float(x), float(y), float(lowest + 0.45 * sz), float(sx), float(sy), float(sz), float(yaw)))
    return rocks

"""
mars_terrain.py: el terreno marciano procedural del mapa de pruebas (sandbox).

Para qué existe
    Antes de la pista real de Jezero había un mapa cuadrado generado por completo. Este archivo lo
    conserva (la escena con kind="sandbox") y aporta piezas que la pista reutiliza, como el tipo Rock.

Qué contiene
    - generate_heightmap: mapa de alturas cuadrado con cráteres y dunas, determinista por semilla.
    - generate_far_heightmap: terreno lejano y de baja resolución que continúa el mapa hasta el horizonte.
    - scatter_rocks y Rock: rocas elipsoidales repartidas sobre el terreno.
    - sample_height: altura del terreno en un punto (x, y).

Cómo funciona
    Solo usa numpy, sin MuJoCo. Cada flujo aleatorio sale de su propio generador derivado de la
    semilla, así no se modifica ningún estado global.

Dónde encaja
    Etapa: simulación (escena). No importa otros archivos del proyecto.
    Lo usan scene.py, headless.py y track_rocks.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SPAWN_RADIUS_M = 6.0
SPAWN_BLEND_M = 14.0
EDGE_TAPER_M = 24.0

_STREAM_ROLLING, _STREAM_CRATERS, _STREAM_DUNES, _STREAM_ROCKS, _STREAM_ROUGH = range(5)
_STREAM_DUNE_WARP = 5
_STREAM_FAR = 6

CRATER_RADIUS_MIN_M = 2.0
CRATER_RADIUS_MAX_M = 15.0

FAR_EXTENT_M = 4000.0
FAR_CELL_M = 10.0
FAR_FADE_IN_M = 150.0
FAR_HOLE_DEPTH_M = -10.0
FAR_LEVEL_M = -0.05

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
    wind_angle: float
    wavelength: float
    amplitude: float


@dataclass(frozen=True)
class Rock:
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


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream])


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _grid(size: int, extent_m: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-extent_m / 2, extent_m / 2, size)
    return np.meshgrid(axis, axis)


def _spectral_noise(
    rng: np.random.Generator,
    size: int,
    extent_m: float,
    wavelength_min_m: float,
    wavelength_max_m: float,
    exponent: float,
) -> np.ndarray:
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    freq = np.fft.fftfreq(size, d=extent_m / size)
    k = np.hypot(*np.meshgrid(freq, freq))
    with np.errstate(divide="ignore"):
        shape = np.where(k > 0, k**-exponent, 0.0)
    shape *= np.exp(-((k * wavelength_min_m) ** 2))
    shape *= 1.0 - np.exp(-((k * wavelength_max_m) ** 2))
    field = np.fft.ifft2(spectrum * shape).real
    field -= field.mean()
    return field / field.std()


def plan_craters(extent_m: float, seed: int) -> tuple[Crater, ...]:
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
            return False
        return bool(np.hypot(x, y) - radius > SPAWN_RADIUS_M + 8.0)

    for index in range(total - 2):
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
            radius = max(CRATER_RADIUS_MIN_M, 0.8 * radius)

    parent = max(craters, key=lambda c: c.radius)
    for _ in range(2):
        radius = rng.uniform(CRATER_RADIUS_MIN_M, 4.0)
        for _ in range(100):
            angle = rng.uniform(0.0, 2 * np.pi)
            dist = parent.radius * rng.uniform(0.8, 1.05)
            x, y = parent.x + dist * np.cos(angle), parent.y + dist * np.sin(angle)
            if fits(x, y, radius):
                craters.append(new_crater(x, y, radius))
                break
    return tuple(craters)


def plan_dune_field(extent_m: float, seed: int) -> DuneField:
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
    corner = half - radius - 6.0
    return DuneField(corner, corner, radius, wind_angle, wavelength, amplitude)


def _crater_profile(s: np.ndarray, depth: float) -> np.ndarray:
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
    phase = along / dunes.wavelength + 0.4 * (across / (0.9 * dunes.radius)) ** 2 + 0.2 * warp
    modulation = 0.7 + 0.3 * np.tanh(warp)
    mask = 1.0 - _smoothstep(0.55, 1.0, np.hypot(dx, dy) / dunes.radius)
    return mask * modulation * dunes.amplitude * (_dune_profile(phase) - 0.4)


def edge_weight(x: np.ndarray | float, y: np.ndarray | float, extent_m: float) -> np.ndarray | float:
    half = extent_m / 2
    return _smoothstep(0.0, EDGE_TAPER_M, half - np.abs(x)) * _smoothstep(0.0, EDGE_TAPER_M, half - np.abs(y))


def generate_heightmap(
    size: int = 257, extent_m: float = 120.0, seed: int = 0, include_craters: bool = True
) -> np.ndarray:
    X, Y = _grid(size, extent_m)
    radius = np.hypot(X, Y)

    rolling = 0.6 * _spectral_noise(_rng(seed, _STREAM_ROLLING), size, extent_m, 8.0, 90.0, 1.5)
    rolling -= rolling[size // 2, size // 2]
    rough = 0.012 * _spectral_noise(_rng(seed, _STREAM_ROUGH), size, extent_m, 1.0, 4.0, 0.5)

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
    size = int(round(FAR_EXTENT_M / FAR_CELL_M)) + 1
    X, Y = _grid(size, FAR_EXTENT_M)
    outside = np.maximum(np.abs(X), np.abs(Y)) - inner_extent_m / 2
    relief = 1.8 * _spectral_noise(_rng(seed, _STREAM_FAR), size, FAR_EXTENT_M, 60.0, 1500.0, 1.6)
    relief *= _smoothstep(0.0, FAR_FADE_IN_M, outside)
    hole = np.clip(-outside / FAR_CELL_M, 0.0, 1.0)
    return relief + FAR_LEVEL_M + FAR_HOLE_DEPTH_M * hole


def sample_height(heights: np.ndarray, extent_m: float, x: float, y: float) -> float:
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

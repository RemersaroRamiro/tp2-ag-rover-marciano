"""
track_rocks.py: el campo de rocas de la pista, según el modelo de Golombek.

Para qué existe
    Las rocas son el principal obstáculo del rover. Este archivo decide cuántas rocas hay, de qué tamaño
    y dónde, con una distribución publicada para los sitios de aterrizaje en Marte.

Qué contiene
    - Rock: una roca elipsoidal (posición, semiejes y orientación).
    - cumulative_rocks_per_m2: cantidad de rocas por m² con diámetro mayor o igual que D.
    - sample_diameters: sortea diámetros que siguen esa distribución.
    - plan_track_rocks: genera las rocas de la pista y de sus márgenes para una semilla.
    - unsurmountable: marca las rocas del carril más altas que la rueda más grande posible; se descartan.

Cómo funciona
    El área cubierta por rocas de diámetro ≥ D es F(D) = k · exp(-q · D), con q = 1.79 + 0.152 / k.
    La abundancia k depende de la sección: 5 % en el piso, hasta 15 % en el campo de rocas y menos en
    la roca firme. Las rocas más pequeñas que un diámetro mínimo no se modelan: las representa la
    rugosidad del terreno.

Dónde encaja
    Etapa: simulación (pista). Usa track_sections, track_terrain, mars_params y genome_spec.
    Lo usa headless.py.

Fuentes
    - Golombek et al.: modelo de fracción de área acumulada cubierta por rocas en Marte.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import genome_spec
import mars_params as mp
import track_sections as ts
import track_terrain as tt

_STREAM_ROCKS = 103
_EULER_GAMMA = 0.5772156649015329
SLAB_M = 2.0

LANE_ROCK_MAX_HEIGHT_M = next(g for g in genome_spec.GENES if g.name == "wheel_diameter_m").high
LANE_ROCK_BAND_M = mp.LANE_WIDTH_M / 2 + mp.ROVER_REACH_M


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


def expint_e1(x):
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, 1.0)
    series = np.zeros_like(x)
    term = np.ones_like(x)
    for n in range(1, 40):
        term = term * (-small) / n
        series = series - term / n
    e1_small = -_EULER_GAMMA - np.log(small) + series

    large = np.maximum(x, 1.0)
    depth = 120
    fraction = large + 2 * depth + 1
    for n in range(depth, 0, -1):
        fraction = large + 2 * (n - 1) + 1 - n * n / fraction
    e1_large = np.exp(-large) / fraction
    return np.where(x <= 1.0, e1_small, e1_large)[()]


def golombek_q(k: float) -> float:
    return 1.79 + 0.152 / k


def cumulative_rocks_per_m2(diameter_m, k: float = mp.ROCK_FRACTIONAL_AREA_K):
    d = np.asarray(diameter_m, dtype=float)
    q = golombek_q(k)
    return (4.0 * k * q / np.pi) * (np.exp(-q * d) / d - q * expint_e1(q * d))


def sample_diameters(
    rng: np.random.Generator, n: int, d_min: float, d_max: float, k: float = mp.ROCK_FRACTIONAL_AREA_K
) -> np.ndarray:
    survival, log_grid = _survival_table(round(d_min, 6), round(d_max, 6), round(k, 6))
    u = rng.uniform(size=n)
    return np.exp(np.interp(u, survival, log_grid))


@lru_cache(maxsize=256)
def _survival_table(d_min: float, d_max: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.geomspace(d_min, d_max, 512)
    survival = cumulative_rocks_per_m2(grid, k)
    survival = (survival - survival[-1]) / (survival[0] - survival[-1])
    return survival[::-1].copy(), np.log(grid)[::-1].copy()


def _smoothstep(edge0: float, edge1: float, x):
    t = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _minimum_diameter(abs_y: np.ndarray, d_lane: float = mp.ROCK_D_MIN_LANE_M) -> np.ndarray:
    lo, hi = mp.ROCK_GRADED_BAND_M
    blend = _smoothstep(lo, hi, abs_y)
    return np.exp(np.log(d_lane) + blend * np.log(mp.ROCK_D_MIN_OUTSIDE_M / d_lane))


def semi_height_m(visible_height_m: float):
    return 0.5 * visible_height_m / (1.0 - mp.ROCK_EMBEDDED_FRACTION)


def visible_height_m(rock: Rock) -> float:
    return 2.0 * rock.sz * (1.0 - mp.ROCK_EMBEDDED_FRACTION)


def unsurmountable(abs_y, visible_height_m, max_height_m: float | None = None) -> np.ndarray:
    limit = LANE_ROCK_MAX_HEIGHT_M if max_height_m is None else max_height_m
    return (np.asarray(abs_y, dtype=float) <= LANE_ROCK_BAND_M) & (np.asarray(visible_height_m, dtype=float) > limit)


def _build_rocks(
    rng: np.random.Generator, terrain: tt.TrackTerrain, x: np.ndarray, y: np.ndarray, diameters: np.ndarray,
    max_lane_height_m: float | None = None,
) -> list[Rock]:
    count = len(x)
    yaw = rng.uniform(0.0, 2 * np.pi, size=count)
    aspect = rng.uniform(0.65, 1.0, size=count)
    ratio = np.clip(rng.normal(mp.ROCK_HEIGHT_TO_DIAMETER, 0.05, size=count), 0.25, 0.55)
    tilt = np.clip(rng.normal(0.0, np.radians(6.0), size=(2, count)), -np.radians(15.0), np.radians(15.0))
    shade = np.clip(rng.normal(1.0, 0.12, size=count), 0.75, 1.25)

    sx = diameters / 2
    sy = sx * aspect
    visible_height = ratio * diameters
    sz = semi_height_m(visible_height)
    c, s = np.cos(yaw), np.sin(yaw)
    footprint = ((0 * x, 0 * x), (sx * c, sx * s), (-sx * c, -sx * s), (-sy * s, sy * c), (sy * s, -sy * c))
    ground = np.mean([terrain.sample(x + dx, y + dy) for dx, dy in footprint], axis=0)
    z = ground + visible_height - sz
    doomed = unsurmountable(np.abs(y), visible_height, max_lane_height_m)
    return [
        Rock(
            float(x[i]), float(y[i]), float(z[i]), float(sx[i]), float(sy[i]), float(sz[i]), float(yaw[i]),
            pitch=float(tilt[0, i]), roll=float(tilt[1, i]), shade=float(shade[i]),
        )
        for i in range(count)
        if not doomed[i]
    ]


def plan_track_rocks(terrain: tt.TrackTerrain, seed: int, max_lane_rock_height_m: float | None = None) -> list[Rock]:
    rng = np.random.default_rng([seed, _STREAM_ROCKS])
    sections = terrain.sections or ts.default_sections(terrain.length_m)
    length, half_lane, (band_lo, band_hi) = terrain.length_m, mp.LANE_WIDTH_M / 2, mp.ROCK_GRADED_BAND_M
    x_lane = (mp.SPAWN_CLEAR_X_M, length - 0.5)
    x_lo, x_hi, y_lo, y_hi = terrain.x_min + 1.0, terrain.x_max - 1.0, terrain.y_min + 1.0, terrain.y_max - 1.0
    xs, ys, ds = [], [], []

    def slabs(lo: float, hi: float):
        edges = np.arange(lo, hi, SLAB_M)
        return [(a, min(a + SLAB_M, hi)) for a in edges]

    for a, b in slabs(*x_lane):
        k, d_lane = float(ts.rock_fractional_area(0.5 * (a + b), sections)), float(ts.rock_d_min_lane(0.5 * (a + b), sections))
        density = cumulative_rocks_per_m2(d_lane, k)
        n = int(rng.poisson(density * (b - a) * 2 * half_lane))
        d = sample_diameters(rng, n, d_lane, mp.ROCK_D_MAX_M, k)
        xs.append(rng.uniform(a, b, size=n)), ys.append(rng.uniform(-half_lane, half_lane, size=n)), ds.append(d)

        n = int(rng.poisson(density * (b - a) * 2 * (band_hi - band_lo)))
        d = sample_diameters(rng, n, d_lane, mp.ROCK_D_MAX_M, k)
        x = rng.uniform(a, b, size=n)
        y = rng.uniform(band_lo, band_hi, size=n) * rng.choice([-1.0, 1.0], size=n)
        keep = d >= _minimum_diameter(np.abs(y), d_lane)
        xs.append(x[keep]), ys.append(y[keep]), ds.append(d[keep])

    fx, fy, fd = [], [], []
    for a, b in slabs(x_lo, x_hi):
        k = float(ts.rock_fractional_area(0.5 * (a + b), sections))
        n = int(rng.poisson(cumulative_rocks_per_m2(mp.ROCK_D_MIN_OUTSIDE_M, k) * (b - a) * (y_hi - y_lo)))
        fd.append(sample_diameters(rng, n, mp.ROCK_D_MIN_OUTSIDE_M, mp.ROCK_D_MAX_M, k))
        fx.append(rng.uniform(a, b, size=n)), fy.append(rng.uniform(y_lo, y_hi, size=n))
    x, y, d = np.concatenate(fx), np.concatenate(fy), np.concatenate(fd)
    before_finish = x <= x_lane[1]
    excluded = (before_finish & (np.abs(y) < half_lane)) | (
        (x >= x_lane[0]) & before_finish & (np.abs(y) < band_hi)
    )
    outside = np.hypot(np.maximum(np.maximum(-x, x - length), 0.0), np.maximum(np.abs(y) - terrain.width_m / 2, 0.0))
    survive = rng.uniform(size=len(x)) < 1.0 - _smoothstep(0.0, mp.ROCK_FADE_OUT_M, outside)
    keep = ~excluded & survive
    xs.append(x[keep]), ys.append(y[keep]), ds.append(d[keep])

    return _build_rocks(rng, terrain, np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), max_lane_rock_height_m)

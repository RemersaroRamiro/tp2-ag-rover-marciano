"""
track_terrain.py: el terreno de la pista de Jezero, construido a partir de un modelo de elevación real.

Para qué existe
    La pista se apoya en una franja real del cráter Jezero (un DTM de HiRISE). Este archivo carga esa
    franja, le agrega la rugosidad fina que el DTM no resuelve y el relieve de cada sección, y produce
    el mapa de alturas que usa la física.

Qué contiene
    - load_strip: carga la franja real (data/track/) y sus datos de origen.
    - generate_track_heightmap: el mapa de alturas de la pista con sus márgenes (TrackTerrain).
    - finish_x: la posición x de la meta.
    - plan_friction_tiles: corta el terreno de colisión en franjas para variar la fricción en la arena.

Cómo funciona
    1. Se suaviza el DTM (filtro gaussiano de 4 m), porque su detalle de 1 m es en parte ruido.
    2. Se agrega rugosidad fractal (Hurst 0.7, pendiente rms de 2.4° a 1 m) solo en las escalas quitadas.
    3. Se suman las formas de cada sección (track_features). El relieve real es el mismo para todas las
       semillas; la semilla solo cambia la rugosidad y pequeños detalles de esas formas.

Dónde encaja
    Etapa: simulación (pista). Usa track_features, track_sections y mars_params.
    Lo usan headless, track_rocks y rover_run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mars_params as mp
import track_features as tf
import track_sections as ts

DATA_DIR = Path(__file__).resolve().parent / "data" / "track"
STRIP_STEM = "jezero_floor_strip"

_STREAM_ROUGHNESS = 101

FAR_CELL_M = 3.8


@dataclass(frozen=True)
class Strip:
    elevation: np.ndarray
    px_x_m: float
    px_y_m: float
    center_row: int
    provenance: dict

    @property
    def rows(self) -> int:
        return self.elevation.shape[0]

    @property
    def cols(self) -> int:
        return self.elevation.shape[1]

    @property
    def length_m(self) -> float:
        return self.cols * self.px_x_m

    @property
    def width_m(self) -> float:
        return self.rows * self.px_y_m


def load_strip(directory: Path = DATA_DIR) -> Strip:
    elevation = np.load(directory / f"{STRIP_STEM}.npy")
    info = json.loads((directory / f"{STRIP_STEM}.json").read_text(encoding="utf-8"))
    assert elevation.shape == (info["shape"]["rows_y"], info["shape"]["cols_x"])
    return Strip(
        elevation=elevation,
        px_x_m=float(info["true_ground_m_per_pixel_x"]),
        px_y_m=float(info["true_ground_m_per_pixel_y"]),
        center_row=int(info["line"]["center_row_index_in_strip"]),
        provenance=info,
    )


def _gaussian_filter_axis(a: np.ndarray, sigma_px: float, axis: int) -> np.ndarray:
    radius = int(np.ceil(4.0 * sigma_px))
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (t / sigma_px) ** 2)
    kernel /= kernel.sum()
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    return np.apply_along_axis(np.convolve, axis, np.pad(a, pad, mode="edge"), kernel, mode="valid")


def lowpass_dtm(strip: Strip, sigma_m: float = mp.DTM_LOWPASS_SIGMA_M) -> np.ndarray:
    field = strip.elevation.astype(np.float64)
    field = _gaussian_filter_axis(field, sigma_m / strip.px_y_m, axis=0)
    return _gaussian_filter_axis(field, sigma_m / strip.px_x_m, axis=1)


def _bilinear(field: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    n_rows, n_cols = field.shape
    r0 = np.minimum(np.floor(row).astype(int), n_rows - 2)
    c0 = np.minimum(np.floor(col).astype(int), n_cols - 2)
    fr, fc = row - r0, col - c0
    top = field[r0, c0] * (1 - fc) + field[r0, c0 + 1] * fc
    bottom = field[r0 + 1, c0] * (1 - fc) + field[r0 + 1, c0 + 1] * fc
    return top * (1 - fr) + bottom * fr


def sample_strip(strip: Strip, field: np.ndarray, x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    col = np.clip(x / strip.px_x_m - 0.5, 0, strip.cols - 1)
    row = np.clip((y + strip.width_m / 2) / strip.px_y_m - 0.5, 0, strip.rows - 1)
    return _bilinear(field, row, col)[()]


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream])


def _smoothstep(edge0: float, edge1: float, x):
    t = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wavenumber(shape: tuple[int, int], cell_x: float, cell_y: float) -> np.ndarray:
    ny, nx = shape
    kx = np.fft.fftfreq(nx, d=cell_x)
    ky = np.fft.fftfreq(ny, d=cell_y)
    return np.hypot(*np.meshgrid(kx, ky))


def _roughness_amplitude(shape: tuple[int, int], cell_x: float, cell_y: float) -> np.ndarray:
    k = _wavenumber(shape, cell_x, cell_y)
    with np.errstate(divide="ignore"):
        amplitude = np.where(k > 0, k ** -(mp.HURST + 1.0), 0.0)
    highpass = 1.0 - np.exp(-2.0 * (np.pi * mp.DTM_LOWPASS_SIGMA_M * k) ** 2)
    return amplitude * highpass * np.exp(-((k * mp.ROUGHNESS_CUTOFF_M) ** 2))


def _expected_rms_lag_difference(amplitude: np.ndarray, lag_x: int, lag_y: int) -> float:
    ny, nx = amplitude.shape
    power = amplitude**2
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    var_x = np.sum(power * (2.0 * (1.0 - np.cos(2.0 * np.pi * kx * lag_x / nx)))[None, :]) / power.size
    var_y = np.sum(power * (2.0 * (1.0 - np.cos(2.0 * np.pi * ky * lag_y / ny)))[:, None]) / power.size
    return float(np.sqrt(0.5 * (var_x + var_y)))


def synthetic_roughness(shape: tuple[int, int], cell_m: float, seed: int, cell_y_m: float | None = None) -> np.ndarray:
    cell_y = cell_m if cell_y_m is None else cell_y_m
    amplitude = _roughness_amplitude(shape, cell_m, cell_y)
    lag_x, lag_y = max(1, round(1.0 / cell_m)), max(1, round(1.0 / cell_y))
    baseline = 0.5 * (lag_x * cell_m + lag_y * cell_y)
    target = np.tan(np.radians(mp.ROUGHNESS_RMS_SLOPE_1M_DEG)) * baseline**mp.HURST
    scale = target / _expected_rms_lag_difference(amplitude, lag_x, lag_y)
    white = _rng(seed, _STREAM_ROUGHNESS).standard_normal(shape)
    return np.fft.ifft2(np.fft.fft2(white) * amplitude).real * scale


def _bilinear_on_axes(field: np.ndarray, x0: float, y0: float, cell_x: float, cell_y: float, x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    col = np.clip((x - x0) / cell_x, 0, field.shape[1] - 1)
    row = np.clip((y - y0) / cell_y, 0, field.shape[0] - 1)
    return _bilinear(field, row, col)[()]


def far_grid_cells(extent_x_m: float, extent_y_m: float) -> tuple[int, int]:
    return max(1, round(extent_x_m / FAR_CELL_M)), max(1, round(extent_y_m / FAR_CELL_M))


@dataclass(frozen=True)
class TrackTerrain:
    heights: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    length_m: float
    width_m: float
    offset_m: float
    features: np.ndarray | None = None
    sections: tuple[ts.Section, ...] = ()

    def sample(self, x, y):
        n_rows, n_cols = self.heights.shape
        return _bilinear_on_axes(
            self.heights, self.x_min, self.y_min,
            (self.x_max - self.x_min) / (n_cols - 1), (self.y_max - self.y_min) / (n_rows - 1), x, y,
        )


def _coarse_surface(strip: Strip, bounds: tuple[float, float, float, float], x, y):
    return _coarse_field(
        bounds, lambda nx, ny: sample_strip(strip, lowpass_dtm(strip, mp.DTM_BORDER_SIGMA_M), nx, ny), x, y
    )


def _coarse_field(bounds: tuple[float, float, float, float], values, x, y):
    x_min, x_max, y_min, y_max = bounds
    cells_x, cells_y = far_grid_cells(x_max - x_min, y_max - y_min)
    nodes = np.meshgrid(np.linspace(x_min, x_max, cells_x + 1), np.linspace(y_min, y_max, cells_y + 1))
    grid = values(*nodes)
    return _bilinear_on_axes(grid, x_min, y_min, (x_max - x_min) / cells_x, (y_max - y_min) / cells_y, x, y)


def _terrain_bounds(strip: Strip) -> tuple[float, float, float, float]:
    margin = mp.TERRAIN_MARGIN_M
    return -margin, strip.length_m + margin, -strip.width_m / 2 - margin, strip.width_m / 2 + margin


def generate_track_heightmap(
    strip: Strip | None = None,
    cell_m: float = mp.TRACK_CELL_M,
    seed: int = 0,
    sections: tuple[ts.Section, ...] | None = None,
    with_features: bool = True,
) -> TrackTerrain:
    strip = strip or load_strip()
    sections = sections or ts.default_sections(strip.length_m)
    bounds = _terrain_bounds(strip)
    x_min, x_max, y_min, y_max = bounds
    far_cells_x, far_cells_y = far_grid_cells(x_max - x_min, y_max - y_min)
    nx = far_cells_x * max(1, round((x_max - x_min) / (cell_m * far_cells_x))) + 1
    ny = far_cells_y * max(1, round((y_max - y_min) / (cell_m * far_cells_y))) + 1
    cell_x, cell_y = (x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1)
    x, y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    to_border = np.minimum.reduce([x - x_min, x_max - x, y - y_min, y_max - y])
    fade = _smoothstep(0.0, mp.TERRAIN_MARGIN_M, to_border)

    detailed = sample_strip(strip, lowpass_dtm(strip, mp.DTM_LOWPASS_SIGMA_M), x, y)
    smooth = _coarse_surface(strip, bounds, x, y)
    roughness = synthetic_roughness((ny, nx), cell_x, seed, cell_y_m=cell_y)
    if with_features:
        generator = tf.TrackFeatures(sections, seed)
        row, column = x[:1, :], y[:, :1]
        coarse_climb = _coarse_field(bounds, generator.climb, x, y)
        features = generator.local_height(row, column) * fade + fade * generator.climb(row, column) + (1.0 - fade) * coarse_climb
    else:
        features = np.zeros_like(fade)
    absolute = smooth + (detailed - smooth) * fade + roughness * fade + features

    offset = float(_bilinear_on_axes(absolute, x_min, y_min, cell_x, cell_y, mp.START_X_M, 0.0))
    return TrackTerrain(
        absolute - offset, x_min, x_max, y_min, y_max, strip.length_m, strip.width_m, offset, features, sections
    )


def finish_x(terrain: TrackTerrain) -> float:
    return terrain.length_m - mp.FINISH_MARGIN_M


@dataclass(frozen=True)
class Tile:
    col0: int
    col1: int
    friction: float


def plan_friction_tiles(terrain: TrackTerrain) -> list[Tile]:
    n_cols = terrain.heights.shape[1]
    cell = (terrain.x_max - terrain.x_min) / (n_cols - 1)
    sand = next(s for s in terrain.sections if s.name == "soft_sand")
    lo, hi = sand.x_start - mp.SECTION_TRANSITION_M / 2, sand.x_end + mp.SECTION_TRANSITION_M / 2
    pieces = max(1, math.ceil((hi - lo) / mp.FRICTION_TILE_M))
    edges = np.round((np.linspace(lo, hi, pieces + 1) - terrain.x_min) / cell).astype(int)
    edges = np.unique(np.clip(np.concatenate([[0], edges, [n_cols - 1]]), 0, n_cols - 1))
    tiles = []
    for a, b in zip(edges[:-1], edges[1:]):
        centre = terrain.x_min + 0.5 * (a + b) * cell
        inside = lo <= centre <= hi
        friction = float(ts.ground_friction(centre, terrain.sections)) if inside else mp.SOIL_FRICTION
        tiles.append(Tile(int(a), int(b), friction))
    return tiles

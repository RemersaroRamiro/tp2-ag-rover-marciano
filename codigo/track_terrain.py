"""
track_terrain.py: el terreno de la pista de Jezero, construido a partir de un modelo de elevación real.

Para qué existe
    La pista se apoya en una franja real del cráter Jezero (un DTM de HiRISE). Este archivo carga esa
    franja, le agrega la rugosidad fina que el DTM no resuelve y el relieve de cada sección, y produce
    los mapas de alturas que usan la física y la escena.

Qué contiene
    - load_strip: carga la franja real (data/track/) y sus datos de origen.
    - generate_track_heightmap: el mapa de alturas de la pista con sus márgenes (TrackTerrain).
    - generate_far_terrain: el terreno lejano, que continúa el mundo sin escalones.
    - make_visual_terrains, plan_markers y plan_sand_decals: versiones más livianas para dibujar,
      postes de distancia y el tinte de la arena (solo visual).
    - plan_friction_tiles: corta el terreno de colisión en franjas para variar la fricción en la arena.

Cómo funciona
    1. Se suaviza el DTM (filtro gaussiano de 4 m), porque su detalle de 1 m es en parte ruido.
    2. Se agrega rugosidad fractal (Hurst 0.7, pendiente rms de 2.4° a 1 m) solo en las escalas quitadas.
    3. Se suman las formas de cada sección (track_features). El relieve real es el mismo para todas las
       semillas; la semilla solo cambia la rugosidad, el relieve lejano y las rocas.

Dónde encaja
    Etapa: simulación (pista). Usa track_features, track_sections y mars_params.
    Lo usan scene, headless, track_rocks y rover_run.
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
_STREAM_FAR = 102

FAR_CELL_M = 3.8
FAR_DEFAULT_PAD_CELLS = 300
FAR_RELAX_M = 300.0
FAR_RELIEF_AMPLITUDE_M = 1.8
FAR_RELIEF_FADE_IN_M = 150.0
FAR_HOLE_DEPTH_M = 4.0
FAR_HOLE_RAMP_M = 10.0

VISUAL_TERRAIN_MAX_DECIMATION = (3, 3)
VISUAL_FLANK_MAX_DECIMATION_Y = 6
VISUAL_LANE_BAND_HALF_M = 20.0
VISUAL_FINE_SECTIONS = ("ripples", "soft_sand", "bedrock", "cross_slope", "crater")
VISUAL_ALIGN_X_M = 50.0
VISUAL_ALIGN_Y_M = 5 * 5.0 * (224 / 416)
VISUAL_FAR_MAX_DECIMATION = 4
VISUAL_SEAM_BAND_M = 15.0

MARKER_LANE_MARGIN_M = 0.5
START_POST_HEIGHT_M = 0.30
DISTANCE_POST_HEIGHT_M = 1.2
SECTION_POST_HEIGHT_M = 0.8

SAND_DECAL_X_LEVELS = (0.1, 0.35, 0.6, 0.85)
SAND_DECAL_Y_HALF_WIDTHS_M = (12.0, 18.0, 24.0)


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


def _spectral_field(
    rng: np.random.Generator,
    shape: tuple[int, int],
    cell_x: float,
    cell_y: float,
    wavelength_min_m: float,
    wavelength_max_m: float,
    exponent: float,
) -> np.ndarray:
    k = _wavenumber(shape, cell_x, cell_y)
    with np.errstate(divide="ignore"):
        shaping = np.where(k > 0, k**-exponent, 0.0)
    shaping = shaping * np.exp(-((k * wavelength_min_m) ** 2)) * (1.0 - np.exp(-((k * wavelength_max_m) ** 2)))
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * shaping).real
    return (field - field.mean()) / field.std()


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

    @property
    def x_axis(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.heights.shape[1])

    @property
    def y_axis(self) -> np.ndarray:
        return np.linspace(self.y_min, self.y_max, self.heights.shape[0])

    @property
    def far_cells(self) -> tuple[int, int]:
        return far_grid_cells(self.x_max - self.x_min, self.y_max - self.y_min)

    def sample(self, x, y):
        n_rows, n_cols = self.heights.shape
        return _bilinear_on_axes(
            self.heights, self.x_min, self.y_min,
            (self.x_max - self.x_min) / (n_cols - 1), (self.y_max - self.y_min) / (n_rows - 1), x, y,
        )

    def sample_features(self, x, y):
        if self.features is None:
            return np.zeros(np.broadcast(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).shape)[()]
        n_rows, n_cols = self.features.shape
        return _bilinear_on_axes(
            self.features, self.x_min, self.y_min,
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


@dataclass(frozen=True)
class FarTerrain:
    heights: np.ndarray
    x0: float
    y0: float
    cell_x: float
    cell_y: float

    @property
    def x_axis(self) -> np.ndarray:
        return self.x0 + self.cell_x * np.arange(self.heights.shape[1])

    @property
    def y_axis(self) -> np.ndarray:
        return self.y0 + self.cell_y * np.arange(self.heights.shape[0])

    def sample(self, x, y):
        return _bilinear_on_axes(self.heights, self.x0, self.y0, self.cell_x, self.cell_y, x, y)


def generate_far_terrain(
    strip: Strip, terrain: TrackTerrain, seed: int = 0, pad_cells: int = FAR_DEFAULT_PAD_CELLS
) -> FarTerrain:
    x_min, x_max, y_min, y_max = terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max
    cells_x, cells_y = terrain.far_cells
    cell_x, cell_y = (x_max - x_min) / cells_x, (y_max - y_min) / cells_y
    nx, ny = cells_x + 2 * pad_cells + 1, cells_y + 2 * pad_cells + 1
    x0, y0 = x_min - pad_cells * cell_x, y_min - pad_cells * cell_y
    x, y = np.meshgrid(x0 + cell_x * np.arange(nx), y0 + cell_y * np.arange(ny))

    outside_x = np.maximum(np.maximum(x_min - x, x - x_max), 0.0)
    outside_y = np.maximum(np.maximum(y_min - y, y - y_max), 0.0)
    d_out = np.hypot(outside_x, outside_y)
    d_in = np.maximum(np.minimum.reduce([x - x_min, x_max - x, y - y_min, y_max - y]), 0.0)

    extension = sample_strip(strip, lowpass_dtm(strip, mp.DTM_BORDER_SIGMA_M), x, y) - terrain.offset_m
    noise = _spectral_field(_rng(seed, _STREAM_FAR), (ny, nx), cell_x, cell_y, 60.0, 1500.0, 1.6)
    climb = tf.TrackFeatures(terrain.sections, seed).climb(x, y)
    heights = (
        extension * (1.0 - _smoothstep(0.0, FAR_RELAX_M, d_out))
        + FAR_RELIEF_AMPLITUDE_M * noise * _smoothstep(0.0, FAR_RELIEF_FADE_IN_M, d_out)
        - FAR_HOLE_DEPTH_M * _smoothstep(0.0, FAR_HOLE_RAMP_M, d_in)
        + climb
    )
    return FarTerrain(heights, x0, y0, cell_x, cell_y)


def decimation_factor(cells: int, maximum: int) -> int:
    return max((n for n in range(1, max(1, maximum) + 1) if cells % n == 0), default=1)


def _decimate_far(far: FarTerrain, terrain: TrackTerrain, max_factor: int) -> FarTerrain:
    cells_x, cells_y = terrain.far_cells
    step = decimation_factor(math.gcd(cells_x, cells_y), max_factor)
    pad_x = round((terrain.x_min - far.x0) / far.cell_x)
    pad_y = round((terrain.y_min - far.y0) / far.cell_y)
    start_x, start_y = pad_x % step, pad_y % step
    return FarTerrain(
        far.heights[start_y::step, start_x::step].copy(),
        far.x0 + start_x * far.cell_x,
        far.y0 + start_y * far.cell_y,
        far.cell_x * step,
        far.cell_y * step,
    )


def _decimate_terrain(terrain: TrackTerrain, visual_far: FarTerrain, max_factor: int | tuple[int, int]) -> TrackTerrain:
    rows, cols = terrain.heights.shape
    max_x, max_y = (max_factor, max_factor) if isinstance(max_factor, int) else max_factor
    step_x, step_y = decimation_factor(cols - 1, max_x), decimation_factor(rows - 1, max_y)
    coarse = terrain.heights[::step_y, ::step_x]
    xs = np.linspace(terrain.x_min, terrain.x_max, coarse.shape[1])
    ys = np.linspace(terrain.y_min, terrain.y_max, coarse.shape[0])

    bottom = visual_far.sample(xs, np.full_like(xs, terrain.y_min)) - coarse[0, :]
    top = visual_far.sample(xs, np.full_like(xs, terrain.y_max)) - coarse[-1, :]
    left = visual_far.sample(np.full_like(ys, terrain.x_min), ys) - coarse[:, 0]
    right = visual_far.sample(np.full_like(ys, terrain.x_max), ys) - coarse[:, -1]
    u = ((xs - terrain.x_min) / (terrain.x_max - terrain.x_min))[None, :]
    v = ((ys - terrain.y_min) / (terrain.y_max - terrain.y_min))[:, None]
    blend = (
        (1 - v) * bottom[None, :] + v * top[None, :] + (1 - u) * left[:, None] + u * right[:, None]
        - ((1 - u) * (1 - v) * bottom[0] + u * (1 - v) * bottom[-1] + (1 - u) * v * top[0] + u * v * top[-1])
    )
    x, y = np.meshgrid(xs, ys)
    to_border = np.minimum.reduce([x - terrain.x_min, terrain.x_max - x, y - terrain.y_min, terrain.y_max - y])
    weight = 1.0 - _smoothstep(0.0, VISUAL_SEAM_BAND_M, to_border)
    return TrackTerrain(
        coarse + weight * blend, terrain.x_min, terrain.x_max, terrain.y_min, terrain.y_max,
        terrain.length_m, terrain.width_m, terrain.offset_m, None, terrain.sections,
    )


@dataclass(frozen=True)
class VisualTerrain:
    segments: tuple[TrackTerrain, ...]
    bands: int = 1

    @property
    def x_min(self) -> float:
        return self.segments[0].x_min

    @property
    def x_max(self) -> float:
        return self.segments[-1].x_max

    @property
    def y_min(self) -> float:
        return self.segments[0].y_min

    @property
    def y_max(self) -> float:
        return self.segments[self.bands - 1].y_max

    @property
    def length_m(self) -> float:
        return self.segments[0].length_m

    @property
    def width_m(self) -> float:
        return self.segments[0].width_m

    @property
    def offset_m(self) -> float:
        return self.segments[0].offset_m

    @property
    def sections(self) -> tuple[ts.Section, ...]:
        return self.segments[0].sections

    @property
    def nodes(self) -> int:
        return sum(s.heights.size for s in self.segments)

    def _indices(self, x, y):
        columns = len(self.segments) // self.bands
        x_edges = np.array([self.segments[k * self.bands].x_min for k in range(1, columns)])
        y_edges = np.array([self.segments[k].y_min for k in range(1, self.bands)])
        return np.searchsorted(x_edges, x, side="left") * self.bands + np.searchsorted(y_edges, y, side="left")

    def segment_at(self, x: float, y: float = 0.0) -> TrackTerrain:
        return self.segments[int(self._indices(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))]

    def sample(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        index, out = self._indices(x, y), np.empty(x.shape)
        for k in np.unique(index):
            mask = index == k
            out[mask] = self.segments[int(k)].sample(x[mask], y[mask])
        return out[()]


def _relief_span(terrain: TrackTerrain) -> tuple[float, float] | None:
    fine = [s for s in terrain.sections if s.name in VISUAL_FINE_SECTIONS]
    if not fine:
        return None
    half = mp.SECTION_TRANSITION_M / 2
    return min(s.x_start for s in fine) - half, max(s.x_end for s in fine) + half


def _aligned_index(origin: float, cell: float, last: int, target: float, period: float, multiple: int, up: bool) -> int:
    steps = (math.ceil if up else math.floor)((target - origin) / period)
    index = round(steps * period / cell)
    index = -(-index // multiple) * multiple if up else index // multiple * multiple
    return int(min(max(index, 0), last))


def _split_visual(
    fine: TrackTerrain, coarse_x: int, lane_y: int, flank_y: int, span: tuple[float, float] | None
) -> VisualTerrain:
    n_rows, n_cols = fine.heights.shape
    last_r, last_c = n_rows - 1, n_cols - 1
    cell_x, cell_y = (fine.x_max - fine.x_min) / last_c, (fine.y_max - fine.y_min) / last_r
    if span is None:
        x_pieces = [(0, last_c, coarse_x)]
    else:
        c0 = _aligned_index(fine.x_min, cell_x, last_c, span[0], VISUAL_ALIGN_X_M, coarse_x, up=False)
        c1 = _aligned_index(fine.x_min, cell_x, last_c, span[1], VISUAL_ALIGN_X_M, coarse_x, up=True)
        x_pieces = [(0, c0, coarse_x)] * (c0 > 0) + [(c0, c1, 1)] + [(c1, last_c, coarse_x)] * (c1 < last_c)
    both = math.lcm(lane_y, flank_y)
    r0 = _aligned_index(fine.y_min, cell_y, last_r, -VISUAL_LANE_BAND_HALF_M, VISUAL_ALIGN_Y_M, both, up=False)
    r1 = _aligned_index(fine.y_min, cell_y, last_r, VISUAL_LANE_BAND_HALF_M, VISUAL_ALIGN_Y_M, both, up=True)
    y_pieces = [(0, r0, flank_y)] * (r0 > 0) + [(r0, r1, lane_y)] + [(r1, last_r, flank_y)] * (r1 < last_r)
    patches = []
    for col0, col1, step_x in x_pieces:
        for row0, row1, step_y in y_pieces:
            patches.append(TrackTerrain(
                fine.heights[row0 : row1 + 1 : step_y, col0 : col1 + 1 : step_x].copy(),
                fine.x_min + col0 * cell_x, fine.x_min + col1 * cell_x,
                fine.y_min + row0 * cell_y, fine.y_min + row1 * cell_y,
                fine.length_m, fine.width_m, fine.offset_m, None, fine.sections,
            ))
    return VisualTerrain(tuple(patches), len(y_pieces))


def make_visual_terrains(
    terrain: TrackTerrain,
    far: FarTerrain,
    terrain_max_factor: int | tuple[int, int] = VISUAL_TERRAIN_MAX_DECIMATION,
    far_max_factor: int = VISUAL_FAR_MAX_DECIMATION,
    flank_max_factor: int = VISUAL_FLANK_MAX_DECIMATION_Y,
) -> tuple[VisualTerrain, FarTerrain]:
    max_x, max_y = (terrain_max_factor, terrain_max_factor) if isinstance(terrain_max_factor, int) else terrain_max_factor
    visual_far = _decimate_far(far, terrain, far_max_factor)
    fine = _decimate_terrain(terrain, visual_far, (1, 1))
    rows, cols = terrain.heights.shape[0] - 1, terrain.heights.shape[1] - 1
    patches = _split_visual(
        fine, decimation_factor(cols, max_x), decimation_factor(rows, max_y), decimation_factor(rows, flank_max_factor),
        _relief_span(terrain),
    )
    return patches, visual_far


@dataclass(frozen=True)
class Marker:
    name: str
    kind: str
    x: float
    y: float
    z_base: float
    height: float
    radius: float
    section: str = ""


def finish_x(terrain: TrackTerrain) -> float:
    return terrain.length_m - mp.FINISH_MARGIN_M


def plan_markers(terrain: TrackTerrain) -> list[Marker]:
    half_lane = mp.LANE_WIDTH_M / 2
    markers: list[Marker] = []

    def add(name: str, kind: str, x: float, y: float, height: float, radius: float, section: str = "") -> None:
        markers.append(Marker(name, kind, x, y, float(terrain.sample(x, y)), height, radius, section))

    line_y = np.linspace(-half_lane, half_lane, 9)
    for i, y in enumerate(line_y):
        add(f"marker_start_{i}", "start", mp.START_X_M, float(y), START_POST_HEIGHT_M, 0.03)
        add(f"marker_finish_{i}", "finish", finish_x(terrain), float(y), START_POST_HEIGHT_M, 0.03)
    step = 1
    while (x := mp.START_X_M + step * mp.MARKER_SPACING_M) < finish_x(terrain) - 1.0:
        for side, y in (("s", -half_lane - MARKER_LANE_MARGIN_M), ("n", half_lane + MARKER_LANE_MARGIN_M)):
            add(f"marker_50m_{int(step * mp.MARKER_SPACING_M)}_{side}", "distance", x, y, DISTANCE_POST_HEIGHT_M, 0.04)
        step += 1
    for section in terrain.sections[1:]:
        for side, y in (("s", -half_lane - MARKER_LANE_MARGIN_M), ("n", half_lane + MARKER_LANE_MARGIN_M)):
            add(f"marker_section_{section.name}_{side}", "section", section.x_start, y, SECTION_POST_HEIGHT_M, 0.04, section.name)
    return markers


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


@dataclass(frozen=True)
class Decal:
    x0: float
    x1: float
    half_width: float
    alpha: float


def plan_sand_decals(sections: tuple[ts.Section, ...]) -> list[Decal]:
    sand = next(s for s in sections if s.name == "soft_sand")
    half = mp.SECTION_TRANSITION_M / 2
    xs = np.arange(sand.x_start - half, sand.x_end + half + 0.125, 0.25)
    weight = ts.section_weight("soft_sand", xs, sections)
    layers = len(SAND_DECAL_X_LEVELS) * len(SAND_DECAL_Y_HALF_WIDTHS_M)
    alpha = 1.0 - (1.0 - mp.SAND_TINT_ALPHA) ** (1.0 / layers)
    decals = []
    for level in SAND_DECAL_X_LEVELS:
        inside = xs[weight >= level]
        for width in SAND_DECAL_Y_HALF_WIDTHS_M:
            if inside.size:
                decals.append(Decal(float(inside[0]), float(inside[-1]), width, float(alpha)))
    return decals

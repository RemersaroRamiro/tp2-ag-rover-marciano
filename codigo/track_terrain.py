"""Terrain of the Jezero track: a real HiRISE DTM strip, low-passed, plus calibrated fractal roughness.

Pure numpy, no MuJoCo import, deterministic per seed. The seed only drives the synthetic parts
(roughness, far backdrop relief, rocks); the real relief of the DTM never changes.

Coordinates (true ground meters, z up)
--------------------------------------
* x runs along the track, from 0 at the strip's west edge to ``length_m`` (~379.5 m) at its east
  edge. Pixels of the DTM are not square on the ground: an x pixel is 0.9486 m, a y pixel 1 m,
  so sizes always come from the provenance JSON, never from the pixel counts.
* y runs across the track, 0 on the strip's centre row (the 2D demo's ``jezero_floor`` line),
  +y = north, ``+-width_m / 2`` at the strip's borders.
* z is relative to the spawn point: the terrain is exactly 0 at ``(START_X_M, 0)``.
* ``heights[row, col]``: rows run along +y, columns along +x (the memory layout of a MuJoCo hfield).

Realism pipeline
----------------
1. The DTM's effective horizontal resolution is ~10-20 px, so 1 m-scale content is partly noise:
   keep only the relief a Gaussian low-pass (sigma 4 m) leaves.
2. Add synthetic fractal roughness (Hurst 0.7, rms slope 2.4 degrees at a 1 m baseline) only at the
   scales the low-pass removed (its spectrum is the complement of the low-pass).
3. The heightfield extends ``TERRAIN_MARGIN_M`` (30 m) beyond the strip on every side. The roughness
   and the DTM detail are at full strength up to and across the strip border and fade to a smooth
   copy of the DTM only over that margin, so the strip has no visible border; the heightfield's own
   border meets a coarse far-field heightfield exactly, without a seam.
4. The mixed track: the synthetic features of the named sections (``track_features``: ripples,
   ledges, cross slope, crater, climb) are added on top, with the same margin fade, so the real DTM
   stays the base relief everywhere. ``plan_friction_tiles`` splits the collision terrain along x so
   that each tile can carry its own friction (soft sand); ``plan_sand_decals`` plans the visual tint.
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

FAR_CELL_M = 3.8  # far-grid cell size; the terrain's border falls exactly on far-grid nodes
FAR_DEFAULT_PAD_CELLS = 300  # far cells beyond each side of the strip (~1.1 km)
FAR_RELAX_M = 300.0  # the DTM extension relaxes to spawn level over this distance
FAR_RELIEF_AMPLITUDE_M = 1.8
FAR_RELIEF_FADE_IN_M = 150.0
FAR_HOLE_DEPTH_M = 4.0  # far surface sinks this deep under the strip (hidden beneath it)
FAR_HOLE_RAMP_M = 10.0

# Visual copies (see ``make_visual_terrains``): the drawn ground is a decimated version of the
# physical one. Decimation keeps every n-th node, n being a divisor of the cell counts.
# MuJoCo shades a heightfield flat, one tone per cell, so cells of 0.5 m facet the 4 m ripples and the
# crater walls into visible blocks. The drawn terrain is therefore a grid of patches, each with its own
# cell size: the lane band (|y| <= ~20 m) over the stretch with relief (ripples .. crater) keeps every
# column (0.25 m along x); the rest of the lane band keeps every 3rd (0.75 m); the flanks beyond the lane
# band keep every 6th row (1.5 m across). Neighbouring patches share whole rows / columns of nodes and
# the heights on them, so the surface is continuous.
VISUAL_TERRAIN_MAX_DECIMATION = (3, 3)  # coarse (x, y): outside the relief zone, and the lane band rows
VISUAL_FLANK_MAX_DECIMATION_Y = 6  # rows of the flanks beyond the lane band
VISUAL_LANE_BAND_HALF_M = 20.0  # half width of the finely drawn band (snapped outwards to the texture period)
VISUAL_FINE_SECTIONS = ("ripples", "soft_sand", "bedrock", "cross_slope", "crater")  # drawn with every column
# Every patch maps the ground texture from its own west / south edge, which is seamless only where that edge
# lies a whole number of texture periods from the terrain's border: scene.py's texture period, in x
# 10 tiles x 5 m, in y 5 tiles x 5 m x tile aspect (a test keeps them equal).
VISUAL_ALIGN_X_M = 50.0
VISUAL_ALIGN_Y_M = 5 * 5.0 * (224 / 416)
VISUAL_FAR_MAX_DECIMATION = 4  # ~15 m cells instead of 3.8 m; the backdrop is hidden by fog
VISUAL_SEAM_BAND_M = 15.0  # the visual terrain blends into the coarse visual far surface over this band

MARKER_LANE_MARGIN_M = 0.5
START_POST_HEIGHT_M = 0.30
DISTANCE_POST_HEIGHT_M = 1.2
SECTION_POST_HEIGHT_M = 0.8  # the stake pairs at section starts: shorter than the 50 m posts, subtle

# Soft-sand tint: translucent layers stacked over the visual terrain (MuJoCo ignores per-pixel texture
# alpha, so the blend is a staircase of uniform layers). A layer covers the x extent where the sand
# weight is at least one of these levels and the |y| extent below one of these half widths; the
# stacked opacity is the product of an along-track and an across-track staircase.
SAND_DECAL_X_LEVELS = (0.1, 0.35, 0.6, 0.85)
SAND_DECAL_Y_HALF_WIDTHS_M = (12.0, 18.0, 24.0)


# -- DTM strip -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Strip:
    """The real DTM strip: ``elevation[row, col]`` in meters (row 0 = south, col 0 = west)."""

    elevation: np.ndarray
    px_x_m: float  # true ground size of one pixel along x
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
    """Load the committed strip and its provenance (true-meter pixel sizes come from the JSON)."""
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
    """Gaussian low-pass of the DTM with ``sigma_m`` in true meters (borders replicate)."""
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
    """Bilinear sample of a strip-shaped ``field`` at true-meter (x, y); queries clamp to the border."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    col = np.clip(x / strip.px_x_m - 0.5, 0, strip.cols - 1)
    row = np.clip((y + strip.width_m / 2) / strip.px_y_m - 0.5, 0, strip.rows - 1)
    return _bilinear(field, row, col)[()]


# -- calibrated fractal roughness ------------------------------------------------------------------


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
    """Amplitude spectrum of a 2-D fractional Brownian surface, ``k**-(H+1)``, high-passed with the
    complement of the DTM low-pass so it only fills the scales the low-pass removed."""
    k = _wavenumber(shape, cell_x, cell_y)
    with np.errstate(divide="ignore"):
        amplitude = np.where(k > 0, k ** -(mp.HURST + 1.0), 0.0)
    highpass = 1.0 - np.exp(-2.0 * (np.pi * mp.DTM_LOWPASS_SIGMA_M * k) ** 2)
    return amplitude * highpass * np.exp(-((k * mp.ROUGHNESS_CUTOFF_M) ** 2))


def _expected_rms_lag_difference(amplitude: np.ndarray, lag_x: int, lag_y: int) -> float:
    """Expected rms of ``f[n + lag] - f[n]`` for white noise shaped by ``amplitude`` (no sampling).

    For ``f = ifft2(fft2(white) * A)``: Var = (1 / N) * sum_k |A_k|^2 * 2 * (1 - cos(2 pi k lag / n)).
    """
    ny, nx = amplitude.shape
    power = amplitude**2
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    var_x = np.sum(power * (2.0 * (1.0 - np.cos(2.0 * np.pi * kx * lag_x / nx)))[None, :]) / power.size
    var_y = np.sum(power * (2.0 * (1.0 - np.cos(2.0 * np.pi * ky * lag_y / ny)))[:, None]) / power.size
    return float(np.sqrt(0.5 * (var_x + var_y)))


def synthetic_roughness(shape: tuple[int, int], cell_m: float, seed: int, cell_y_m: float | None = None) -> np.ndarray:
    """Fractal roughness on a ``shape = (rows, cols)`` grid, in meters, zero mean.

    Calibrated analytically (independent of the seed): the expected rms height difference over a
    1 m baseline equals ``tan(rms slope) * 1 m**H`` with the slope and H from ``mars_params``, i.e.
    rms dh(L) ~ 0.042 * L**0.7 m.
    """
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
    """Band-limited fractal noise with zero mean and unit std on a (rows, cols) grid of rectangular cells."""
    k = _wavenumber(shape, cell_x, cell_y)
    with np.errstate(divide="ignore"):
        shaping = np.where(k > 0, k**-exponent, 0.0)
    shaping = shaping * np.exp(-((k * wavelength_min_m) ** 2)) * (1.0 - np.exp(-((k * wavelength_max_m) ** 2)))
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * shaping).real
    return (field - field.mean()) / field.std()


# -- the track heightmap ---------------------------------------------------------------------------


def _bilinear_on_axes(field: np.ndarray, x0: float, y0: float, cell_x: float, cell_y: float, x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    col = np.clip((x - x0) / cell_x, 0, field.shape[1] - 1)
    row = np.clip((y - y0) / cell_y, 0, field.shape[0] - 1)
    return _bilinear(field, row, col)[()]


def far_grid_cells(extent_x_m: float, extent_y_m: float) -> tuple[int, int]:
    """Far-grid cells across a rectangle of the given size (about ``FAR_CELL_M`` each)."""
    return max(1, round(extent_x_m / FAR_CELL_M)), max(1, round(extent_y_m / FAR_CELL_M))


@dataclass(frozen=True)
class TrackTerrain:
    """Track heightfield over the real strip plus a synthetic margin on every side.

    It covers ``x in [x_min, x_max]`` and ``y in [y_min, y_max]``; the real strip is
    ``[0, length_m] x [-width_m / 2, +width_m / 2]`` inside it.
    """

    heights: np.ndarray  # (rows=y, cols=x), meters relative to the spawn point
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    length_m: float  # of the real strip (not of the whole heightfield)
    width_m: float
    offset_m: float  # absolute DTM elevation that was subtracted so that z = 0 at the spawn point
    features: np.ndarray | None = None  # the synthetic section relief already included in ``heights``
    sections: tuple[ts.Section, ...] = ()  # named sections of the strip (``track_sections``)

    @property
    def x_axis(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.heights.shape[1])

    @property
    def y_axis(self) -> np.ndarray:
        return np.linspace(self.y_min, self.y_max, self.heights.shape[0])

    @property
    def far_cells(self) -> tuple[int, int]:
        """Far-grid cells across this terrain: their nodes are a subset of the terrain's nodes."""
        return far_grid_cells(self.x_max - self.x_min, self.y_max - self.y_min)

    def sample(self, x, y):
        """Bilinear height at world (x, y); accepts arrays; queries clamp to the terrain border."""
        n_rows, n_cols = self.heights.shape
        return _bilinear_on_axes(
            self.heights, self.x_min, self.y_min,
            (self.x_max - self.x_min) / (n_cols - 1), (self.y_max - self.y_min) / (n_rows - 1), x, y,
        )

    def sample_features(self, x, y):
        """Bilinear height of the synthetic section features alone (zero when the terrain has none)."""
        if self.features is None:
            return np.zeros(np.broadcast(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).shape)[()]
        n_rows, n_cols = self.features.shape
        return _bilinear_on_axes(
            self.features, self.x_min, self.y_min,
            (self.x_max - self.x_min) / (n_cols - 1), (self.y_max - self.y_min) / (n_rows - 1), x, y,
        )


def _coarse_surface(strip: Strip, bounds: tuple[float, float, float, float], x, y):
    """Smoothed DTM (sigma ``DTM_BORDER_SIGMA_M``, extended past the strip by replicating its border)
    interpolated bilinearly between the nodes of the far grid over ``bounds = (x_min, x_max, y_min,
    y_max)``. The terrain's border takes exactly these values, and the far heightfield uses them at
    its border nodes: both surfaces are then piecewise-linear along the border with the same values,
    so they meet without any step (a step of even 1 cm exposes the hfield side wall)."""
    return _coarse_field(
        bounds, lambda nx, ny: sample_strip(strip, lowpass_dtm(strip, mp.DTM_BORDER_SIGMA_M), nx, ny), x, y
    )


def _coarse_field(bounds: tuple[float, float, float, float], values, x, y):
    """``values(x, y)`` evaluated at the nodes of the far grid over ``bounds`` and interpolated bilinearly
    between them: on the border this is exactly what the far heightfield (which stores ``values`` at its
    nodes) reproduces."""
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
    """Heightmap of the strip plus its margins at about ``cell_m`` resolution (1740 x 480 cells at
    the default 0.25 m), deterministic per ``seed``.

    Inside the strip: the low-passed real DTM plus full-strength synthetic roughness. Over the
    ``TERRAIN_MARGIN_M`` beyond it, the DTM detail blends into its smooth extension and the roughness
    fades to zero, so the border of the heightfield meets the far backdrop exactly. The section
    features (``with_features``) are added on top with the same fade.
    """
    strip = strip or load_strip()
    sections = sections or ts.default_sections(strip.length_m)
    bounds = _terrain_bounds(strip)
    x_min, x_max, y_min, y_max = bounds
    # The far grid's nodes across the terrain are a subset of these nodes (see ``_coarse_surface``).
    far_cells_x, far_cells_y = far_grid_cells(x_max - x_min, y_max - y_min)
    nx = far_cells_x * max(1, round((x_max - x_min) / (cell_m * far_cells_x))) + 1
    ny = far_cells_y * max(1, round((y_max - y_min) / (cell_m * far_cells_y))) + 1
    cell_x, cell_y = (x_max - x_min) / (nx - 1), (y_max - y_min) / (ny - 1)
    x, y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    to_border = np.minimum.reduce([x - x_min, x_max - x, y - y_min, y_max - y])  # 0 on the heightfield border
    fade = _smoothstep(0.0, mp.TERRAIN_MARGIN_M, to_border)  # 1 from the strip border inwards

    detailed = sample_strip(strip, lowpass_dtm(strip, mp.DTM_LOWPASS_SIGMA_M), x, y)
    smooth = _coarse_surface(strip, bounds, x, y)
    roughness = synthetic_roughness((ny, nx), cell_x, seed, cell_y_m=cell_y)
    if with_features:  # row / column vectors: the feature weights only depend on x
        generator = tf.TrackFeatures(sections, seed)
        row, column = x[:1, :], y[:, :1]
        # The local features (ripples, ledges, crater, ...) fade out towards the border. The climb is a regional
        # rise: it continues across the border (the far backdrop carries it too), so near the border it is blended
        # into its interpolation between the far nodes, which is exactly what the far heightfield stores there.
        coarse_climb = _coarse_field(bounds, generator.climb, x, y)
        features = generator.local_height(row, column) * fade + fade * generator.climb(row, column) + (1.0 - fade) * coarse_climb
    else:
        features = np.zeros_like(fade)
    absolute = smooth + (detailed - smooth) * fade + roughness * fade + features

    offset = float(_bilinear_on_axes(absolute, x_min, y_min, cell_x, cell_y, mp.START_X_M, 0.0))
    return TrackTerrain(
        absolute - offset, x_min, x_max, y_min, y_max, strip.length_m, strip.width_m, offset, features, sections
    )


# -- far backdrop --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FarTerrain:
    """Coarse heightfield that continues the world beyond the track terrain. Rectangular cells: the
    terrain's border lies exactly on grid nodes, so the two heightfields join without a step."""

    heights: np.ndarray  # (rows=y, cols=x)
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
    """Far backdrop: the smoothed DTM border extended outwards, relaxing to gentle procedural relief.

    Under the track terrain the surface sinks by ``FAR_HOLE_DEPTH_M`` so it never shows through it.
    """
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
    climb = tf.TrackFeatures(terrain.sections, seed).climb(x, y)  # the regional rise never stops: up to the horizon
    heights = (
        extension * (1.0 - _smoothstep(0.0, FAR_RELAX_M, d_out))
        + FAR_RELIEF_AMPLITUDE_M * noise * _smoothstep(0.0, FAR_RELIEF_FADE_IN_M, d_out)
        - FAR_HOLE_DEPTH_M * _smoothstep(0.0, FAR_HOLE_RAMP_M, d_in)
        + climb
    )
    return FarTerrain(heights, x0, y0, cell_x, cell_y)


# -- visual copies: decimated terrain pair for rendering ----------------------------------------------------


def decimation_factor(cells: int, maximum: int) -> int:
    """Largest divisor of ``cells`` that is at most ``maximum`` (1 when there is none): keeping every
    n-th node then reproduces both ends of the grid exactly."""
    return max((n for n in range(1, max(1, maximum) + 1) if cells % n == 0), default=1)


def _decimate_far(far: FarTerrain, terrain: TrackTerrain, max_factor: int) -> FarTerrain:
    """Every ``n``-th node of the far grid, chosen so the terrain border still falls on kept nodes."""
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
    """Every ``n``-th node of the terrain, with its border pulled onto the coarse visual far surface.

    The collision terrain's border lies on the fine far grid, but the visual far grid is coarser, so
    between its nodes it is a straight line that the fine border does not follow (a few cm off). A
    step of even 1 cm shows the hfield's side wall, so the visual border is set to that line exactly
    and the correction (a Coons blend of the four edge differences) fades to zero over
    ``VISUAL_SEAM_BAND_M``: the strip and the lane are never touched.
    """
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
    """The drawn ground: a grid of ``TrackTerrain`` patches, stored x-major (all ``bands`` patches across y for
    the first x piece, then the next x piece...). Neighbours share their edge nodes and edge heights, so
    together they are one continuous surface at mixed resolution."""

    segments: tuple[TrackTerrain, ...]
    bands: int = 1  # patches across y in every x piece

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
        """Patch index of every (x, y): the x piece by the west edges, the band by the south edges (clamped)."""
        columns = len(self.segments) // self.bands
        x_edges = np.array([self.segments[k * self.bands].x_min for k in range(1, columns)])
        y_edges = np.array([self.segments[k].y_min for k in range(1, self.bands)])
        return np.searchsorted(x_edges, x, side="left") * self.bands + np.searchsorted(y_edges, y, side="left")

    def segment_at(self, x: float, y: float = 0.0) -> TrackTerrain:
        """The patch that contains world (x, y) (a shared edge belongs to the western / southern patch)."""
        return self.segments[int(self._indices(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))]

    def sample(self, x, y):
        """Bilinear height of the drawn surface at world (x, y); accepts arrays."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        index, out = self._indices(x, y), np.empty(x.shape)
        for k in np.unique(index):
            mask = index == k
            out[mask] = self.segments[int(k)].sample(x[mask], y[mask])
        return out[()]


def _relief_span(terrain: TrackTerrain) -> tuple[float, float] | None:
    """x extent, transition zones included, of the sections that need finely drawn cells (None when unknown)."""
    fine = [s for s in terrain.sections if s.name in VISUAL_FINE_SECTIONS]
    if not fine:
        return None
    half = mp.SECTION_TRANSITION_M / 2
    return min(s.x_start for s in fine) - half, max(s.x_end for s in fine) + half


def _aligned_index(origin: float, cell: float, last: int, target: float, period: float, multiple: int, up: bool) -> int:
    """Node index nearest below / above ``target`` that lies on ``origin + k * period`` (texture period) and is a
    multiple of ``multiple`` (so every patch's decimated grid contains it), clamped to [0, last]."""
    steps = (math.ceil if up else math.floor)((target - origin) / period)
    index = round(steps * period / cell)
    index = -(-index // multiple) * multiple if up else index // multiple * multiple
    return int(min(max(index, 0), last))


def _split_visual(
    fine: TrackTerrain, coarse_x: int, lane_y: int, flank_y: int, span: tuple[float, float] | None
) -> VisualTerrain:
    """Cut ``fine`` (every node kept) into the patch grid: x pieces (every column inside ``span``, every
    ``coarse_x``-th outside) times y bands (every ``lane_y``-th row in the lane band, ``flank_y``-th beyond)."""
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
    """Decimated (drawn) copies of the collision terrain pair, meeting each other without a step.

    Rendering cost is proportional to the hfield triangle count (drawn twice: shadow pass), so the
    viewer draws these lighter copies and hides the full-resolution collision heightfields. The drawn
    terrain is a ``VisualTerrain`` patch grid: the lane band over the relief zone keeps every column, the
    rest of the lane band every ``max_x``-th, and the flanks beyond it every ``flank_max_factor``-th row.
    """
    max_x, max_y = (terrain_max_factor, terrain_max_factor) if isinstance(terrain_max_factor, int) else terrain_max_factor
    visual_far = _decimate_far(far, terrain, far_max_factor)
    # the seam blend is evaluated on every node, so the patches keep exact values on the far line
    fine = _decimate_terrain(terrain, visual_far, (1, 1))
    rows, cols = terrain.heights.shape[0] - 1, terrain.heights.shape[1] - 1
    patches = _split_visual(
        fine, decimation_factor(cols, max_x), decimation_factor(rows, max_y), decimation_factor(rows, flank_max_factor),
        _relief_span(terrain),
    )
    return patches, visual_far


# -- lane and markers ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Marker:
    """A thin visual-only post standing on the terrain (cylinder of ``radius``, ``height``)."""

    name: str
    kind: str  # "start" | "finish" | "distance" | "section"
    x: float
    y: float
    z_base: float
    height: float
    radius: float
    section: str = ""  # for kind == "section": the section that starts at this stake pair


def finish_x(terrain: TrackTerrain) -> float:
    return terrain.length_m - mp.FINISH_MARGIN_M


def plan_markers(terrain: TrackTerrain) -> list[Marker]:
    """Start and finish lines (rows of low stakes across the lane) and a pair of posts every 50 m."""
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
    for section in terrain.sections[1:]:  # a subtle pair of stakes where each section (after the first) starts
        for side, y in (("s", -half_lane - MARKER_LANE_MARGIN_M), ("n", half_lane + MARKER_LANE_MARGIN_M)):
            add(f"marker_section_{section.name}_{side}", "section", section.x_start, y, SECTION_POST_HEIGHT_M, 0.04, section.name)
    return markers


# -- friction tiles and sand tint ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """A slice of the collision terrain: columns ``col0..col1`` inclusive (neighbours share an edge column)
    with the sliding friction its geom carries."""

    col0: int
    col1: int
    friction: float


def plan_friction_tiles(terrain: TrackTerrain) -> list[Tile]:
    """Split the collision terrain along x so the soft-sand zone can change friction gradually.

    MuJoCo friction is per geom, so the ground is cut into a west remainder, ``FRICTION_TILE_M``-wide
    tiles across the whole transition-sand-transition zone (each with the friction of its centre) and
    an east remainder. Tiles share their edge column, hence identical heights on the shared edge: no
    step between them.
    """
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
    """A translucent visual layer over ``x0 <= x <= x1`` and ``|y| <= half_width`` (true meters)."""

    x0: float
    x1: float
    half_width: float
    alpha: float


def plan_sand_decals(sections: tuple[ts.Section, ...]) -> list[Decal]:
    """Nested translucent layers that tint the ground in the soft-sand zone, fading in along the track
    (following the sand weight) and out across it. Their stacked opacity peaks at ``SAND_TINT_ALPHA``."""
    sand = next(s for s in sections if s.name == "soft_sand")
    half = mp.SECTION_TRANSITION_M / 2
    xs = np.arange(sand.x_start - half, sand.x_end + half + 0.125, 0.25)
    weight = ts.section_weight("soft_sand", xs, sections)
    layers = len(SAND_DECAL_X_LEVELS) * len(SAND_DECAL_Y_HALF_WIDTHS_M)
    alpha = 1.0 - (1.0 - mp.SAND_TINT_ALPHA) ** (1.0 / layers)  # `layers` of them stack to SAND_TINT_ALPHA
    decals = []
    for level in SAND_DECAL_X_LEVELS:
        inside = xs[weight >= level]
        for width in SAND_DECAL_Y_HALF_WIDTHS_M:
            if inside.size:
                decals.append(Decal(float(inside[0]), float(inside[-1]), width, float(alpha)))
    return decals

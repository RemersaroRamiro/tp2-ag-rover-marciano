"""Rock field of the Jezero track from the Golombek et al. size-frequency model.

Golombek's model gives the cumulative fractional area covered by rocks of diameter >= D::

    F_k(D) = k * exp(-q * D),    q = 1.79 + 0.152 / k

Converting the area to a number of circles of area pi D^2 / 4 gives the cumulative number per m^2::

    N(>=D) = (4 k q / pi) * [exp(-q D) / D - q * E1(q D)]

with E1 the exponential integral (implemented below with numpy only, no scipy). For k = 5 %:
N(>=0.10 m) = 1.03, N(>=0.25 m) = 0.135, N(>=0.5 m) = 0.0136 per m^2.

Rocks smaller than the cut-off diameter are not modelled as geoms (MuJoCo budget); the fractal
roughness of the terrain stands for them.

The abundance ``k`` follows the track sections (``track_sections``): the base value on the floor,
ramping up to ``ROCK_K_DENSE`` in the rock field and down on the bedrock. The rocks are drawn slab by
slab along x (``SLAB_M``), each slab with the local ``k``, its own Golombek size distribution and a
lane cut-off diameter that grows with ``k`` so the dense field stays inside the geom budget.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import genome_spec
import mars_params as mp
import mars_terrain as mt
import track_sections as ts
import track_terrain as tt

_STREAM_ROCKS = 103
_EULER_GAMMA = 0.5772156649015329
SLAB_M = 2.0  # rocks are drawn in slabs of this length along x, each with the local abundance

# Unsurmountable rocks (DECLARED SIMPLIFICATION). A rocker-bogie climbs obstacles up to about one wheel diameter in
# height (T13 research; NASA's Curiosity/Perseverance rule of thumb), and the largest wheel any genome can have is the
# upper bound of ``wheel_diameter_m``. A rock whose visible height above the ground exceeds that, and that a rover of
# the lane could touch (``|y| <= LANE_ROCK_BAND_M``: the lane half-width plus the reach of any rover), is a wall for
# EVERY possible design: no evolution could ever get past it, and every rover would die at that x whatever it is.
# Real rovers path-plan around such boulders; our controller drives straight, so the rock is left out of the plan.
# Nothing else changes: the random stream is consumed as before (other rocks keep their exact place and size) and
# rocks outside the band are kept however tall. Read from the genome spec, not hard-coded.
LANE_ROCK_MAX_HEIGHT_M = next(g for g in genome_spec.GENES if g.name == "wheel_diameter_m").high
LANE_ROCK_BAND_M = mp.LANE_WIDTH_M / 2 + mp.ROVER_REACH_M


def expint_e1(x):
    """Exponential integral E1(x) for x > 0 (scalar or array): power series below 1, continued fraction above."""
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, 1.0)
    series = np.zeros_like(x)
    term = np.ones_like(x)
    for n in range(1, 40):
        term = term * (-small) / n
        series = series - term / n  # sum_{n>=1} (-1)^(n+1) x^n / (n n!)
    e1_small = -_EULER_GAMMA - np.log(small) + series

    large = np.maximum(x, 1.0)
    depth = 120  # backward evaluation of E1 = e^-x / (x+1 - 1/(x+3 - 4/(x+5 - ...)))
    fraction = large + 2 * depth + 1
    for n in range(depth, 0, -1):
        fraction = large + 2 * (n - 1) + 1 - n * n / fraction
    e1_large = np.exp(-large) / fraction
    return np.where(x <= 1.0, e1_small, e1_large)[()]


def golombek_q(k: float) -> float:
    return 1.79 + 0.152 / k


def cumulative_rocks_per_m2(diameter_m, k: float = mp.ROCK_FRACTIONAL_AREA_K):
    """Number of rocks per m^2 with diameter >= ``diameter_m`` (Golombek model, fractional area ``k``)."""
    d = np.asarray(diameter_m, dtype=float)
    q = golombek_q(k)
    return (4.0 * k * q / np.pi) * (np.exp(-q * d) / d - q * expint_e1(q * d))


def sample_diameters(
    rng: np.random.Generator, n: int, d_min: float, d_max: float, k: float = mp.ROCK_FRACTIONAL_AREA_K
) -> np.ndarray:
    """``n`` diameters in [d_min, d_max] following N(>=D) (inverse-transform sampling on a log grid)."""
    survival, log_grid = _survival_table(round(d_min, 6), round(d_max, 6), round(k, 6))
    u = rng.uniform(size=n)
    # survival decreases with D: interpolate log D as a function of the (reversed) survival values
    return np.exp(np.interp(u, survival, log_grid))


@lru_cache(maxsize=256)
def _survival_table(d_min: float, d_max: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Reversed normalised survival function and log-diameter grid used by the inverse-transform sampler."""
    grid = np.geomspace(d_min, d_max, 512)
    survival = cumulative_rocks_per_m2(grid, k)
    survival = (survival - survival[-1]) / (survival[0] - survival[-1])  # 1 at d_min, 0 at d_max
    return survival[::-1].copy(), np.log(grid)[::-1].copy()


def _smoothstep(edge0: float, edge1: float, x):
    t = np.clip((np.asarray(x, dtype=float) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _minimum_diameter(abs_y: np.ndarray, d_lane: float = mp.ROCK_D_MIN_LANE_M) -> np.ndarray:
    """Smallest modelled diameter at lateral distance |y|: the lane's cut-off inside the lane, rising
    smoothly (in log space) to the boulder-only cut-off across the graded band."""
    lo, hi = mp.ROCK_GRADED_BAND_M
    blend = _smoothstep(lo, hi, abs_y)
    return np.exp(np.log(d_lane) + blend * np.log(mp.ROCK_D_MIN_OUTSIDE_M / d_lane))


def semi_height_m(visible_height_m: float):
    """Vertical semi-axis of a rock that stands ``visible_height_m`` above the ground with its buried share
    (``ROCK_EMBEDDED_FRACTION``) below it. The builder, ``visible_height_m`` and the tests share this one formula."""
    return 0.5 * visible_height_m / (1.0 - mp.ROCK_EMBEDDED_FRACTION)


def visible_height_m(rock: mt.Rock) -> float:
    """Height of ``rock`` above the ground it stands on: the inverse of ``semi_height_m``."""
    return 2.0 * rock.sz * (1.0 - mp.ROCK_EMBEDDED_FRACTION)


def unsurmountable(abs_y, visible_height_m, max_height_m: float | None = None) -> np.ndarray:
    """Mask of the rocks no rover could get over and that a rover of the lane could touch: visible height above the
    ground strictly over ``max_height_m`` (default ``LANE_ROCK_MAX_HEIGHT_M``) at ``|y| <= LANE_ROCK_BAND_M``."""
    limit = LANE_ROCK_MAX_HEIGHT_M if max_height_m is None else max_height_m
    return (np.asarray(abs_y, dtype=float) <= LANE_ROCK_BAND_M) & (np.asarray(visible_height_m, dtype=float) > limit)


def _build_rocks(
    rng: np.random.Generator, terrain: tt.TrackTerrain, x: np.ndarray, y: np.ndarray, diameters: np.ndarray,
    max_lane_height_m: float | None = None,
) -> list[mt.Rock]:
    """Turn positions and diameters into partly buried, slightly irregular ellipsoids on the ground. Every random
    draw is made for every rock, then the unsurmountable ones are dropped (see ``unsurmountable``)."""
    count = len(x)
    yaw = rng.uniform(0.0, 2 * np.pi, size=count)
    aspect = rng.uniform(0.65, 1.0, size=count)  # plan aspect ratio (slightly irregular)
    ratio = np.clip(rng.normal(mp.ROCK_HEIGHT_TO_DIAMETER, 0.05, size=count), 0.25, 0.55)
    tilt = np.clip(rng.normal(0.0, np.radians(6.0), size=(2, count)), -np.radians(15.0), np.radians(15.0))
    shade = np.clip(rng.normal(1.0, 0.12, size=count), 0.75, 1.25)  # individual albedo variation

    sx = diameters / 2
    sy = sx * aspect
    visible_height = ratio * diameters
    sz = semi_height_m(visible_height)  # the buried share makes the full height taller than the visible one
    # Rest on the mean ground height under the footprint, then stand `visible_height` above it.
    c, s = np.cos(yaw), np.sin(yaw)
    footprint = ((0 * x, 0 * x), (sx * c, sx * s), (-sx * c, -sx * s), (-sy * s, sy * c), (sy * s, -sy * c))
    ground = np.mean([terrain.sample(x + dx, y + dy) for dx, dy in footprint], axis=0)
    z = ground + visible_height - sz
    doomed = unsurmountable(np.abs(y), visible_height, max_lane_height_m)
    return [
        mt.Rock(
            float(x[i]), float(y[i]), float(z[i]), float(sx[i]), float(sy[i]), float(sz[i]), float(yaw[i]),
            pitch=float(tilt[0, i]), roll=float(tilt[1, i]), shade=float(shade[i]),
        )
        for i in range(count)
        if not doomed[i]
    ]


def plan_track_rocks(terrain: tt.TrackTerrain, seed: int, max_lane_rock_height_m: float | None = None) -> list[mt.Rock]:
    """Deterministic rocks for the strip and its margins, continuous across every border.

    * Lane (|y| <= 10 m, from ``SPAWN_CLEAR_X_M`` to the strip end): diameters from the local lane
      cut-off (``ROCK_D_MIN_LANE_M`` on the floor, larger in the dense rock field), abundance ``k``
      of the section under each 2 m slab.
    * Graded band (10 to 16 m from the centre line): the cut-off diameter rises smoothly from the
      lane's to the boulder cut-off, so the lane edge is not a wall of rocks.
    * Everywhere else, including the terrain margins around the strip: boulders from
      ``ROCK_D_MIN_OUTSIDE_M`` with the local abundance, thinned to zero over ``ROCK_FADE_OUT_M``
      beyond the strip border, so the rocks fade out with distance instead of stopping at the border.

    Nothing in the spawn zone (x < ``SPAWN_CLEAR_X_M``) or behind the start line inside the lane.

    Rocks that no rover can surmount and that one in the lane could touch are left out (``unsurmountable``): taller
    than ``max_lane_rock_height_m`` (default ``LANE_ROCK_MAX_HEIGHT_M``, read at call time; ``math.inf`` = keep all)
    within ``LANE_ROCK_BAND_M`` of the centre line. Both the viewer's scene and the headless model call this function.
    """
    rng = np.random.default_rng([seed, _STREAM_ROCKS])
    sections = terrain.sections or ts.default_sections(terrain.length_m)
    length, half_lane, (band_lo, band_hi) = terrain.length_m, mp.LANE_WIDTH_M / 2, mp.ROCK_GRADED_BAND_M
    x_lane = (mp.SPAWN_CLEAR_X_M, length - 0.5)
    x_lo, x_hi, y_lo, y_hi = terrain.x_min + 1.0, terrain.x_max - 1.0, terrain.y_min + 1.0, terrain.y_max - 1.0
    xs, ys, ds = [], [], []

    def slabs(lo: float, hi: float):
        edges = np.arange(lo, hi, SLAB_M)
        return [(a, min(a + SLAB_M, hi)) for a in edges]

    # 1. lane and 2. graded band, slab by slab (local abundance and cut-off)
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

    # 3. boulders over the whole terrain, thinned outside the strip, excluding lane and band
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

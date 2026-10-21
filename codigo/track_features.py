"""Synthetic relief of the mixed track: what each section adds on top of the real DTM floor.

Pure numpy, no MuJoCo. ``TrackFeatures.height(x, y)`` is the height added at true-ground (x, y); the
track terrain adds it to the real DTM + roughness. Every feature is scaled by the blend weight of its
section (``track_sections``), so it is exactly zero where that weight is zero and ramps in and out
over ``SECTION_TRANSITION_M`` without a jump. All amplitudes are named constants in ``mars_params``.

=============  ==============================================================================
ripples        aeolian ripples / megaripples: asymmetric ridges (steep lee face towards +x),
               roughly perpendicular to the lane, slightly oblique and wavy; amplitude follows
               the weight of ``ripples`` plus a fraction (small ripples) in ``soft_sand``
bedrock        outcrop: plates separated by ledges that alternate up and down, so the level
               never drifts; step heights ramp up along the section to ``BEDROCK_STEP_M``;
               plates tilt slightly along the track; ledges are slightly oblique
cross_slope    lateral tilt across the lane (+y side higher); bounded away from the lane
crater         one small crater on the lane: cosine bowl plus a gaussian raised rim
climb          a REGIONAL rise, not a mesa: the slope ramps from 0 to ``CLIMB_SLOPE_DEG`` over
               the transition, holds to the end of the strip, eases to zero over ``CLIMB_TOP_EASE_M``
               and the height gained is kept for ever. It does not depend on y except through a wavy
               scarp line (zero shift inside the lane), so it has no lateral walls: the terrain and
               the far backdrop both carry it across their whole width
=============  ==============================================================================

The seed only moves the small-scale forms (crest phase, obliquity and waviness of the ripples,
ledge positions, plate tilts); the layout, the amplitudes and the crater are the same for every seed.
"""

from __future__ import annotations

import numpy as np

import mars_params as mp
import track_sections as ts

_STREAM_FEATURES = 106
_LEDGE_Y_SCALE_M = 15.0  # ledge obliquity acts on tanh(y / scale) * scale: ledges can never cross far off-lane
_CROSS_LINEAR_HALF_WIDTH_M = 11.0  # the cross slope is exactly linear in y up to here (the lane is +-10 m)
_CROSS_SATURATION_M = 6.0  # beyond it the tilt eases out (tanh) so the flanks stay bounded
_RAMP_FROM, _RAMP_TO = 0.05, 0.55  # ledge heights grow over this share of the bedrock section
# The cross slope is linear in y up to _CROSS_LINEAR_HALF_WIDTH_M, eases out (tanh) and is tapered back to zero
# between these |y|: a hillside that returns to the base level well inside the terrain's margin, not a wedge.
_CROSS_TAPER_FROM_M, _CROSS_TAPER_TO_M = 16.0, 50.0
_SCARP_TERMS = ((0.30, 60.0), (0.15, 160.0), (0.55, 500.0))  # (share of the wander, wavelength along y in m)
_SCARP_FADE_IN_M = 20.0  # the wander grows from 0 to full over this |y| beyond the lane edge
_BROADENING_FADE_IN_M = 150.0  # the scarp broadens from 1x (lane) to CLIMB_SCARP_BROADENING x over this |y|


def _smoothstep_integral(t):
    """Integral from 0 to t of smoothstep(s): 0 below 0, t^3 - t^4/2 on [0, 1], 1/2 + (t - 1) beyond."""
    t = np.asarray(t, dtype=float)
    inner = np.clip(t, 0.0, 1.0)
    return inner**3 - 0.5 * inner**4 + np.maximum(t - 1.0, 0.0) * (t > 1.0)


class TrackFeatures:
    """Relief added by the sections of ``sections``; deterministic per ``seed``."""

    def __init__(self, sections: tuple[ts.Section, ...], seed: int = 0):
        self.sections = tuple(sections)
        self._index = {s.name: i for i, s in enumerate(self.sections)}
        rng = np.random.default_rng([seed, _STREAM_FEATURES])
        self._init_ripples(rng)
        self._init_bedrock(rng)
        self._init_scarp()

    # -- setup ---------------------------------------------------------------------------------------

    def _init_ripples(self, rng: np.random.Generator) -> None:
        self._ripple_theta = np.radians(mp.RIPPLE_OBLIQUITY_DEG) * rng.uniform(-1.0, 1.0)
        self._ripple_phase0 = rng.uniform(0.0, 2 * np.pi)
        # crest lines wander across the lane (terms in y) and the spacing breathes along it (terms in x)
        self._warp_y = [(a * mp.RIPPLE_WARP_RAD, lam, rng.uniform(0, 2 * np.pi))
                        for a, lam in zip((0.6, 0.3, 0.1), (17.0, 29.0, 43.0))]
        self._warp_x = [(a * mp.RIPPLE_WARP_RAD, lam, rng.uniform(0, 2 * np.pi))
                        for a, lam in zip((0.8, 0.4), (61.0, 113.0))]

    def _init_bedrock(self, rng: np.random.Generator) -> None:
        s = self.sections[self._index["bedrock"]]
        lo, hi = mp.BEDROCK_PLATE_LENGTH_M
        first, last = s.x_start - mp.SECTION_TRANSITION_M / 2, s.x_end + mp.SECTION_TRANSITION_M / 2
        ledges = []
        x = first + rng.uniform(lo, hi)
        while x < last:
            ledges.append(x)
            x += rng.uniform(lo, hi)
        self.ledge_x = np.array(ledges)  # nominal ledge positions (on the centre line y = 0)
        self._ledge_tan = np.tan(np.radians(mp.BEDROCK_LEDGE_OBLIQUITY_DEG) * rng.uniform(-1.0, 1.0, len(ledges)))
        bounds = np.concatenate([[self.ledge_x[0] - 0.5 * (lo + hi)], self.ledge_x, [self.ledge_x[-1] + 0.5 * (lo + hi)]])
        centres = 0.5 * (bounds[:-1] + bounds[1:])  # one plate before, between and after the ledges
        u = (centres - s.x_start) / s.length
        ramp = ts.smoothstep((u - _RAMP_FROM) / (_RAMP_TO - _RAMP_FROM))
        sign = np.where(np.arange(len(centres)) % 2 == 0, 1.0, -1.0)
        self._plate_centre = centres
        self._plate_level = sign * 0.5 * mp.BEDROCK_STEP_M * ramp  # +-half a step: ledges alternate up / down
        self._plate_dip = ramp * np.tan(np.radians(mp.BEDROCK_PLATE_TILT_DEG)) * rng.uniform(-1.0, 1.0, len(centres))

    def _init_scarp(self) -> None:
        # fixed phases, not seeded: the scarp is part of the layout, which is the same for every seed
        self._scarp = [(share * mp.CLIMB_SCARP_WANDER_M, lam, phase) for (share, lam), phase in zip(_SCARP_TERMS, (0.7, 2.9, 4.1))]

    # -- weights -------------------------------------------------------------------------------------

    def _weights(self, x) -> np.ndarray:
        return ts.section_weights(x, self.sections)

    def _w(self, weights: np.ndarray, name: str) -> np.ndarray:
        return weights[self._index[name]]

    # -- the features ----------------------------------------------------------------------------------

    def _ripples(self, x, y, weights) -> np.ndarray:
        amplitude = 0.5 * mp.RIPPLE_HEIGHT_M * (
            self._w(weights, "ripples") + mp.SAND_RIPPLE_FRACTION * self._w(weights, "soft_sand")
        )
        warp = sum(a * np.sin(2 * np.pi * y / lam + p) for a, lam, p in self._warp_y)
        warp = warp + sum(a * np.sin(2 * np.pi * x / lam + p) for a, lam, p in self._warp_x)
        along = x * np.cos(self._ripple_theta) + y * np.sin(self._ripple_theta)
        phase = 2 * np.pi * along / mp.RIPPLE_WAVELENGTH_M + self._ripple_phase0 + warp
        # sin(phase - a sin(phase)): the derivative is steepest (1 + a) on the descending face, so the
        # short steep face looks downwind (+x) and the long gentle face is the stoss side
        return amplitude * np.sin(phase - mp.RIPPLE_ASYMMETRY * np.sin(phase))

    def _bedrock(self, x, y, weights) -> np.ndarray:
        edge = mp.BEDROCK_LEDGE_EDGE_M
        y_eff = _LEDGE_Y_SCALE_M * np.tanh(y / _LEDGE_Y_SCALE_M)
        below = np.ones(np.broadcast(x, y).shape)  # weight of the plate that ends at the current ledge
        total = np.zeros_like(below)
        for i, (ledge, tan) in enumerate(zip(self.ledge_x, self._ledge_tan)):
            passed = ts.smoothstep((x - (ledge + y_eff * tan)) / edge + 0.5)
            total = total + (below - passed) * self._plate(i, x)
            below = passed
        total = total + below * self._plate(len(self.ledge_x), x)
        return self._w(weights, "bedrock") * total

    def _plate(self, j: int, x):
        return self._plate_level[j] + self._plate_dip[j] * (x - self._plate_centre[j])

    def _cross_slope(self, y, weights) -> np.ndarray:
        half, sat = _CROSS_LINEAR_HALF_WIDTH_M, _CROSS_SATURATION_M
        a = np.abs(y)
        taper = 1.0 - ts.smoothstep((a - _CROSS_TAPER_FROM_M) / (_CROSS_TAPER_TO_M - _CROSS_TAPER_FROM_M))
        lateral = np.sign(y) * np.where(a <= half, a, half + sat * np.tanh((a - half) / sat)) * taper
        return self._w(weights, "cross_slope") * np.tan(np.radians(mp.CROSS_SLOPE_DEG)) * lateral

    def _crater(self, x, y, weights) -> np.ndarray:
        section = self.sections[self._index["crater"]]
        radius = 0.5 * mp.CRATER_DIAMETER_M
        r = np.hypot(x - section.center, y - mp.CRATER_CENTER_Y_M)
        depth = mp.CRATER_DEPTH_TO_DIAMETER * mp.CRATER_DIAMETER_M
        rim = mp.CRATER_RIM_HEIGHT_TO_DIAMETER * mp.CRATER_DIAMETER_M
        width = mp.CRATER_RIM_WIDTH_FRACTION * radius
        bowl = np.where(r < radius, -0.5 * depth * (1.0 + np.cos(np.pi * np.minimum(r, radius) / radius)), 0.0)
        return self._w(weights, "crater") * (bowl + rim * np.exp(-(((r - radius) / width) ** 2)))

    def _scarp_shift(self, y) -> np.ndarray:
        """How far the start of the rise is pushed towards +x at ``y``: exactly 0 inside the lane."""
        fade = ts.smoothstep((np.abs(y) - mp.LANE_WIDTH_M / 2) / _SCARP_FADE_IN_M)
        return fade * sum(a * np.sin(2 * np.pi * y / lam + p) for a, lam, p in self._scarp)

    def _climb(self, x, y) -> np.ndarray:
        """Height gained: slope ramps in over the transition, holds to the strip's end, eases out over
        ``CLIMB_TOP_EASE_M`` (``W I((x-a)/W) - E I((x-L)/E)``: the integrals of the ramp-in and of the ease-out)."""
        width, ease = mp.SECTION_TRANSITION_M, mp.CLIMB_TOP_EASE_M
        start = self.sections[-1].x_start - width / 2  # where the slope begins to ramp in
        top = self.sections[-1].x_end - start  # distance from there to where it begins to ease out
        # away from the lane the whole profile is stretched about the scarp's start: the same rise over a longer
        # distance (gentler slope, same plateau); the lane (|y| <= 10 m) has a factor of exactly 1
        fade = ts.smoothstep((np.abs(y) - mp.LANE_WIDTH_M / 2) / _BROADENING_FADE_IN_M)
        d = (x - self._scarp_shift(y) - start) / (1.0 + (mp.CLIMB_SCARP_BROADENING - 1.0) * fade)
        gain = width * _smoothstep_integral(d / width) - ease * _smoothstep_integral((d - top) / ease)
        return np.tan(np.radians(mp.CLIMB_SLOPE_DEG)) * gain

    # -- public ------------------------------------------------------------------------------------------

    def components(self, x, y) -> dict[str, np.ndarray]:
        """Height added by each feature at ``(x, y)`` (true meters, broadcast together)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        shape = np.broadcast(x, y).shape
        weights = self._weights(x)
        parts = {
            "ripples": self._ripples(x, y, weights),
            "bedrock": self._bedrock(x, y, weights),
            "cross_slope": self._cross_slope(y, weights),
            "crater": self._crater(x, y, weights),
            "climb": self._climb(x, y),
        }
        return {name: np.broadcast_to(part, shape) for name, part in parts.items()}

    def height(self, x, y) -> np.ndarray:
        """Total synthetic height at ``(x, y)``."""
        return sum(self.components(x, y).values())

    def climb(self, x, y) -> np.ndarray:
        """The regional rise alone (it is the only feature that continues beyond the terrain)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return np.broadcast_to(self._climb(x, y), np.broadcast(x, y).shape)

    def local_height(self, x, y) -> np.ndarray:
        """Everything but the regional rise: the features that fade out towards the terrain's border."""
        parts = self.components(x, y)
        return sum(v for k, v in parts.items() if k != "climb")

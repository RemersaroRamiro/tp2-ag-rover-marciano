"""Named sections along the mixed track and their smooth blend weights.

Pure numpy, no MuJoCo. The track is ONE continuous traverse made of eight sections along +x, from
gentle to hard::

    floor -> rock_field -> ripples -> soft_sand -> bedrock -> cross_slope -> crater -> climb

Section ``i`` nominally spans ``[x_start, x_end)``; the boundary between two sections is not a line
but a transition zone of ``SECTION_TRANSITION_M`` centred on it, over which their weights cross with
a smoothstep. The weights of all sections sum to exactly 1 and every weight is 0 beyond half a
transition from its section, so a feature scaled by its weight "is present only where the weight is
positive" and ramps in and out without a single jump. The first section extends to -infinity and
the last one to +infinity (the strip's ends and the margins beyond them).

The same weights drive every per-section quantity: rock abundance (``rock_fractional_area``), the
smallest modelled lane rock (``rock_d_min_lane``) and the ground friction (``ground_friction``);
``track_features`` uses them for the relief. Future fitness / analysis code reports per-section
progress with ``section_at``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mars_params as mp

SECTION_NAMES = ("floor", "rock_field", "ripples", "soft_sand", "bedrock", "cross_slope", "crater", "climb")
# Spanish names for the viewer overlay. ASCII only: MuJoCo's font has no accents or "n with tilde".
SECTION_LABELS = {
    "floor": "piso",
    "rock_field": "campo de rocas",
    "ripples": "ondulaciones",
    "soft_sand": "arena blanda",
    "bedrock": "roca madre",
    "cross_slope": "pendiente lateral",
    "crater": "crater",
    "climb": "subida",
}


@dataclass(frozen=True)
class Section:
    """One named stretch of the track: nominal extent ``[x_start, x_end)`` in true ground meters."""

    name: str
    label: str
    x_start: float
    x_end: float

    @property
    def length(self) -> float:
        return self.x_end - self.x_start

    @property
    def center(self) -> float:
        return 0.5 * (self.x_start + self.x_end)


def default_sections(length_m: float) -> tuple[Section, ...]:
    """The eight sections tiling ``[0, length_m]`` (boundaries from ``mars_params.SECTION_BOUNDARIES_M``)."""
    edges = (0.0, *mp.SECTION_BOUNDARIES_M, float(length_m))
    return tuple(
        Section(name, SECTION_LABELS[name], edges[i], edges[i + 1]) for i, name in enumerate(SECTION_NAMES)
    )


def smoothstep(t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def section_weights(x, sections: tuple[Section, ...], transition_m: float = mp.SECTION_TRANSITION_M) -> np.ndarray:
    """Blend weights, shape ``(len(sections), *x.shape)``: non-negative, in [0, 1], summing to 1.

    ``c_i`` rises from 0 to 1 across boundary ``i``; the weight of section ``i`` is ``c_{i-1} - c_i``
    (a telescoping partition of unity, non-negative because boundaries are further apart than the
    transition zone).
    """
    x = np.asarray(x, dtype=float)
    rises = [smoothstep((x - (s.x_end - transition_m / 2)) / transition_m) for s in sections[:-1]]
    ends = [np.ones_like(x), *rises]
    starts = [*rises, np.zeros_like(x)]
    return np.stack([a - b for a, b in zip(ends, starts)])


def section_index(name: str, sections: tuple[Section, ...]) -> int:
    for i, s in enumerate(sections):
        if s.name == name:
            return i
    raise KeyError(name)


def section_weight(name: str, x, sections: tuple[Section, ...], transition_m: float = mp.SECTION_TRANSITION_M):
    """Blend weight of the section called ``name`` at ``x`` (scalar or array)."""
    return section_weights(x, sections, transition_m)[section_index(name, sections)][()]


def section_at(x: float, sections: tuple[Section, ...]) -> Section:
    """The section whose nominal extent contains ``x``; positions beyond the ends belong to the first / last."""
    return next((s for s in sections[:-1] if x < s.x_end), sections[-1])


# -- per-section laws ----------------------------------------------------------------------------------


def rock_fractional_area(x, sections: tuple[Section, ...]):
    """Golombek abundance ``k`` at ``x``: base on the floor, ``ROCK_K_DENSE`` in the rock field, reduced on bedrock."""
    base = mp.ROCK_FRACTIONAL_AREA_K
    dense = section_weight("rock_field", x, sections)
    bare = section_weight("bedrock", x, sections)
    return base + (mp.ROCK_K_DENSE - base) * dense + (base * mp.ROCK_K_BEDROCK_FACTOR - base) * bare


def rock_d_min_lane(x, sections: tuple[Section, ...]):
    """Smallest modelled lane rock at ``x``: it grows with the abundance (never below the floor's cut-off)
    so that a dense field stays inside the rock-geom budget."""
    base = mp.ROCK_FRACTIONAL_AREA_K
    dense = np.clip((rock_fractional_area(x, sections) - base) / (mp.ROCK_K_DENSE - base), 0.0, 1.0)
    return mp.ROCK_D_MIN_LANE_M + (mp.ROCK_D_MIN_DENSE_M - mp.ROCK_D_MIN_LANE_M) * dense


def ground_friction(x, sections: tuple[Section, ...]):
    """Sliding friction coefficient of the soil at ``x``: ``SOIL_FRICTION``, easing to ``SOFT_SAND_FRICTION``."""
    sand = section_weight("soft_sand", x, sections)
    return mp.SOIL_FRICTION + (mp.SOFT_SAND_FRICTION - mp.SOIL_FRICTION) * sand

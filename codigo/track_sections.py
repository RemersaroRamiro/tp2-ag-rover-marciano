"""
track_sections.py: las ocho secciones de la pista y cómo se mezclan entre sí.

Para qué existe
    La pista es un único recorrido que va de lo fácil a lo difícil. Este archivo define dónde empieza y
    termina cada sección y cuánto pesa cada una en cada punto, sin saltos bruscos.

Qué contiene
    - default_sections: las ocho secciones: floor, rock_field, ripples, soft_sand, bedrock, cross_slope,
      crater y climb.
    - section_weights: el peso de cada sección en una posición x (siempre suman 1).
    - section_at: la sección que corresponde a una posición x.
    - rock_fractional_area, rock_d_min_lane y ground_friction: densidad de rocas, tamaño mínimo de roca
      modelada y fricción del suelo según la sección.

Cómo funciona
    Entre dos secciones hay una transición de 16 m en la que los pesos se cruzan suavemente (smoothstep).
    Así ninguna característica aparece de golpe y la distancia recorrida siempre puede mejorar de a poco.

Dónde encaja
    Etapa: simulación (pista). Usa mars_params. Lo usan track_terrain, track_features, track_rocks
    y headless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mars_params as mp

SECTION_NAMES = ("floor", "rock_field", "ripples", "soft_sand", "bedrock", "cross_slope", "crater", "climb")
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
    edges = (0.0, *mp.SECTION_BOUNDARIES_M, float(length_m))
    return tuple(
        Section(name, SECTION_LABELS[name], edges[i], edges[i + 1]) for i, name in enumerate(SECTION_NAMES)
    )


def smoothstep(t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def section_weights(x, sections: tuple[Section, ...], transition_m: float = mp.SECTION_TRANSITION_M) -> np.ndarray:
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
    return section_weights(x, sections, transition_m)[section_index(name, sections)][()]


def section_at(x: float, sections: tuple[Section, ...]) -> Section:
    return next((s for s in sections[:-1] if x < s.x_end), sections[-1])


def rock_fractional_area(x, sections: tuple[Section, ...]):
    base = mp.ROCK_FRACTIONAL_AREA_K
    dense = section_weight("rock_field", x, sections)
    bare = section_weight("bedrock", x, sections)
    return base + (mp.ROCK_K_DENSE - base) * dense + (base * mp.ROCK_K_BEDROCK_FACTOR - base) * bare


def rock_d_min_lane(x, sections: tuple[Section, ...]):
    base = mp.ROCK_FRACTIONAL_AREA_K
    dense = np.clip((rock_fractional_area(x, sections) - base) / (mp.ROCK_K_DENSE - base), 0.0, 1.0)
    return mp.ROCK_D_MIN_LANE_M + (mp.ROCK_D_MIN_DENSE_M - mp.ROCK_D_MIN_LANE_M) * dense


def ground_friction(x, sections: tuple[Section, ...]):
    sand = section_weight("soft_sand", x, sections)
    return mp.SOIL_FRICTION + (mp.SOFT_SAND_FRICTION - mp.SOIL_FRICTION) * sand

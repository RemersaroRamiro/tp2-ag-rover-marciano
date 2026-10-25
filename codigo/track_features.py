"""
track_features.py: el relieve sintético que cada sección le agrega a la pista.

Para qué existe
    El relieve real de Jezero es suave. Para que la pista ponga a prueba distintas capacidades del rover,
    cada sección suma una forma propia sobre ese relieve.

Qué contiene
    - TrackFeatures: calcula la altura agregada en cada punto (x, y), de forma determinista por semilla.
    - TrackFeatures.components: la altura que aporta cada forma por separado.
    - TrackFeatures.climb: solo la subida final, la única forma que continúa más allá del terreno.

Cómo funciona
    Las formas son: ondulaciones de arena (ripples), escalones de roca (bedrock), pendiente lateral
    (cross_slope), un cráter pequeño (crater) y una subida regional final (climb). Cada forma se
    multiplica por el peso de su sección, así aparece y desaparece de forma gradual. La semilla solo
    mueve detalles pequeños (fase y ondulación de las crestas, posición de los escalones).

Dónde encaja
    Etapa: simulación (pista). Usa track_sections y mars_params, donde están todas las amplitudes.
    Lo usa track_terrain.py.
"""

from __future__ import annotations

import numpy as np

import mars_params as mp
import track_sections as ts

_STREAM_FEATURES = 106
_LEDGE_Y_SCALE_M = 15.0
_CROSS_LINEAR_HALF_WIDTH_M = 11.0
_CROSS_SATURATION_M = 6.0
_RAMP_FROM, _RAMP_TO = 0.05, 0.55
_CROSS_TAPER_FROM_M, _CROSS_TAPER_TO_M = 16.0, 50.0
_SCARP_TERMS = ((0.30, 60.0), (0.15, 160.0), (0.55, 500.0))
_SCARP_FADE_IN_M = 20.0
_BROADENING_FADE_IN_M = 150.0


def _smoothstep_integral(t):
    t = np.asarray(t, dtype=float)
    inner = np.clip(t, 0.0, 1.0)
    return inner**3 - 0.5 * inner**4 + np.maximum(t - 1.0, 0.0) * (t > 1.0)


class TrackFeatures:
    def __init__(self, sections: tuple[ts.Section, ...], seed: int = 0):
        self.sections = tuple(sections)
        self._index = {s.name: i for i, s in enumerate(self.sections)}
        rng = np.random.default_rng([seed, _STREAM_FEATURES])
        self._init_ripples(rng)
        self._init_bedrock(rng)
        self._init_scarp()

    def _init_ripples(self, rng: np.random.Generator) -> None:
        self._ripple_theta = np.radians(mp.RIPPLE_OBLIQUITY_DEG) * rng.uniform(-1.0, 1.0)
        self._ripple_phase0 = rng.uniform(0.0, 2 * np.pi)
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
        self.ledge_x = np.array(ledges)
        self._ledge_tan = np.tan(np.radians(mp.BEDROCK_LEDGE_OBLIQUITY_DEG) * rng.uniform(-1.0, 1.0, len(ledges)))
        bounds = np.concatenate([[self.ledge_x[0] - 0.5 * (lo + hi)], self.ledge_x, [self.ledge_x[-1] + 0.5 * (lo + hi)]])
        centres = 0.5 * (bounds[:-1] + bounds[1:])
        u = (centres - s.x_start) / s.length
        ramp = ts.smoothstep((u - _RAMP_FROM) / (_RAMP_TO - _RAMP_FROM))
        sign = np.where(np.arange(len(centres)) % 2 == 0, 1.0, -1.0)
        self._plate_centre = centres
        self._plate_level = sign * 0.5 * mp.BEDROCK_STEP_M * ramp
        self._plate_dip = ramp * np.tan(np.radians(mp.BEDROCK_PLATE_TILT_DEG)) * rng.uniform(-1.0, 1.0, len(centres))

    def _init_scarp(self) -> None:
        self._scarp = [(share * mp.CLIMB_SCARP_WANDER_M, lam, phase) for (share, lam), phase in zip(_SCARP_TERMS, (0.7, 2.9, 4.1))]

    def _weights(self, x) -> np.ndarray:
        return ts.section_weights(x, self.sections)

    def _w(self, weights: np.ndarray, name: str) -> np.ndarray:
        return weights[self._index[name]]

    def _ripples(self, x, y, weights) -> np.ndarray:
        amplitude = 0.5 * mp.RIPPLE_HEIGHT_M * (
            self._w(weights, "ripples") + mp.SAND_RIPPLE_FRACTION * self._w(weights, "soft_sand")
        )
        warp = sum(a * np.sin(2 * np.pi * y / lam + p) for a, lam, p in self._warp_y)
        warp = warp + sum(a * np.sin(2 * np.pi * x / lam + p) for a, lam, p in self._warp_x)
        along = x * np.cos(self._ripple_theta) + y * np.sin(self._ripple_theta)
        phase = 2 * np.pi * along / mp.RIPPLE_WAVELENGTH_M + self._ripple_phase0 + warp
        return amplitude * np.sin(phase - mp.RIPPLE_ASYMMETRY * np.sin(phase))

    def _bedrock(self, x, y, weights) -> np.ndarray:
        edge = mp.BEDROCK_LEDGE_EDGE_M
        y_eff = _LEDGE_Y_SCALE_M * np.tanh(y / _LEDGE_Y_SCALE_M)
        below = np.ones(np.broadcast(x, y).shape)
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
        fade = ts.smoothstep((np.abs(y) - mp.LANE_WIDTH_M / 2) / _SCARP_FADE_IN_M)
        return fade * sum(a * np.sin(2 * np.pi * y / lam + p) for a, lam, p in self._scarp)

    def _climb(self, x, y) -> np.ndarray:
        width, ease = mp.SECTION_TRANSITION_M, mp.CLIMB_TOP_EASE_M
        start = self.sections[-1].x_start - width / 2
        top = self.sections[-1].x_end - start
        fade = ts.smoothstep((np.abs(y) - mp.LANE_WIDTH_M / 2) / _BROADENING_FADE_IN_M)
        d = (x - self._scarp_shift(y) - start) / (1.0 + (mp.CLIMB_SCARP_BROADENING - 1.0) * fade)
        gain = width * _smoothstep_integral(d / width) - ease * _smoothstep_integral((d - top) / ease)
        return np.tan(np.radians(mp.CLIMB_SLOPE_DEG)) * gain

    def components(self, x, y) -> dict[str, np.ndarray]:
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
        return sum(self.components(x, y).values())

    def climb(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return np.broadcast_to(self._climb(x, y), np.broadcast(x, y).shape)

    def local_height(self, x, y) -> np.ndarray:
        parts = self.components(x, y)
        return sum(v for k, v in parts.items() if k != "climb")

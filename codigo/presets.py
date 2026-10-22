"""
presets.py: los tamaños de corrida del AG.

Para qué existe
    Da nombre a las combinaciones de población y generaciones, para lanzar una corrida con una sola
    opción (--preset) y que las corridas comparables usen siempre los mismos tamaños.

Qué contiene
    - Preset: población, generaciones y terrenos por generación (2 por defecto: la distancia se
      promedia), y la cantidad de simulaciones que cuesta una corrida.
    - PRESETS: rapido (12 individuos × 10 generaciones), normal (30 × 30) y completo (50 × 60).
    - SMOKE: el preset "prueba" (4 × 2), solo para verificar en segundos que todo funciona.
    - resolve: devuelve un preset a partir de su nombre.

Dónde encaja
    Etapa: corrida. No importa otros archivos del proyecto. Lo usa evolve.py para leer la opción
    --preset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    population: int
    generations: int
    description: str
    seeds_per_generation: int = 2

    def __post_init__(self):
        if self.seeds_per_generation < 1:
            raise ValueError(f"seeds_per_generation must be at least 1, got {self.seeds_per_generation}")

    @property
    def evaluations(self) -> int:
        return self.population * self.generations * self.seeds_per_generation


PRESETS: dict[str, Preset] = {
    "rapido": Preset("rapido", 12, 10, "prueba rapida: se ve como evoluciona en pocos minutos"),
    "normal": Preset("normal", 30, 30, "corrida normal de evolucion"),
    "completo": Preset("completo", 50, 60, "corrida completa, para las conclusiones del TP"),
}

SMOKE = Preset("prueba", 4, 2, "prueba minima: solo para verificar que todo funciona")


def resolve(name: str) -> Preset:
    available = {**PRESETS, SMOKE.name: SMOKE}
    if name not in available:
        raise KeyError(f"unknown preset {name!r}; choose one of {', '.join(available)}")
    return available[name]

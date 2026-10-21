"""Evolution presets: how big a GA run is. The GA loop (T18) and the benchmark share these constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    population: int
    generations: int
    description: str
    seeds_per_generation: int = 2  # terrain seeds every rover of a generation is scored on (the distance is averaged)

    def __post_init__(self):
        if self.seeds_per_generation < 1:
            raise ValueError(f"seeds_per_generation must be at least 1, got {self.seeds_per_generation}")

    @property
    def evaluations(self) -> int:
        """Rover simulations in one GA run (every generation re-evaluates its whole population, elites included, on each
        of its terrain seeds: the track seeds change every generation)."""
        return self.population * self.generations * self.seeds_per_generation


PRESETS: dict[str, Preset] = {
    "rapido": Preset("rapido", 12, 10, "prueba rapida: se ve como evoluciona en pocos minutos"),
    "normal": Preset("normal", 30, 30, "corrida normal de evolucion"),
    "completo": Preset("completo", 50, 60, "corrida completa, para las conclusiones del TP"),
}

# A tiny preset for smoke tests and for checking a change end to end in seconds; not part of PRESETS (the benchmark
# estimates only the real ones), but ``evolve.py --preset prueba`` accepts it.
SMOKE = Preset("prueba", 4, 2, "prueba minima: solo para verificar que todo funciona")


def resolve(name: str) -> Preset:
    """The preset called ``name`` (the real ones and ``prueba``); ``KeyError`` with the valid names otherwise."""
    available = {**PRESETS, SMOKE.name: SMOKE}
    if name not in available:
        raise KeyError(f"unknown preset {name!r}; choose one of {', '.join(available)}")
    return available[name]

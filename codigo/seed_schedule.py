"""
seed_schedule.py: elige las semillas de terreno de una corrida.

Para qué existe
    Cada generación se evalúa en terrenos distintos, para que el AG no se adapte a una sola disposición
    de rocas, y al final se valida en terrenos que nunca vio. Este archivo decide esas semillas a partir
    de la semilla de la corrida, así la corrida es reproducible.

Qué contiene
    - SeedSchedule: las semillas de entrenamiento de cada generación y las semillas de validación.
    - SeedSchedule.for_run: arma las semillas de una corrida.
    - seeds_for_generation: las semillas en las que se evalúa una generación.

Cómo funciona
    La semilla solo cambia las rocas y la rugosidad fina; el relieve y las secciones son fijos. Las
    semillas salen de numpy.random.SeedSequence, en flujos separados del generador del AG. Ninguna
    semilla de entrenamiento se repite, y las de validación nunca coinciden con ellas.

Dónde encaja
    Etapa: corrida. Usa evolution_params. Lo usa evolve.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import evolution_params as params


def _stream(run_seed: int, purpose: int):
    rng = np.random.default_rng(np.random.SeedSequence([int(run_seed), params.SCHEDULE_TAG, purpose]))
    while True:
        yield int(rng.integers(1, params.TERRAIN_SEED_LIMIT))


def _draw(run_seed: int, purpose: int, count: int, taken: set[int]) -> tuple[int, ...]:
    drawn: list[int] = []
    for seed in _stream(run_seed, purpose):
        if len(drawn) == count:
            break
        if seed not in taken:
            taken.add(seed)
            drawn.append(seed)
    return tuple(drawn)


@dataclass(frozen=True)
class SeedSchedule:
    run_seed: int
    training: tuple[int, ...]
    validation: tuple[int, ...]
    seeds_per_generation: int = 1
    extra: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def for_run(cls, run_seed: int, generations: int, validation_seeds: int = params.VALIDATION_SEEDS,
                seeds_per_generation: int = 1) -> "SeedSchedule":
        if generations < 1:
            raise ValueError(f"generations must be at least 1, got {generations}")
        if validation_seeds < 0:
            raise ValueError(f"validation_seeds must be non-negative, got {validation_seeds}")
        if seeds_per_generation < 1:
            raise ValueError(f"seeds_per_generation must be at least 1, got {seeds_per_generation}")
        taken: set[int] = set()
        training = _draw(run_seed, 0, generations, taken)
        more = seeds_per_generation - 1
        drawn = _draw(run_seed, 2, generations * more, taken)
        extra = tuple(drawn[g * more:(g + 1) * more] for g in range(generations)) if more else ()
        validation = _draw(run_seed, 1, validation_seeds, taken)
        return cls(int(run_seed), training, validation, seeds_per_generation, extra)

    def _check(self, generation: int) -> None:
        if not 0 <= generation < len(self.training):
            raise IndexError(f"generation {generation} is outside this schedule ({len(self.training)} generations)")

    def seeds_for_generation(self, generation: int) -> tuple[int, ...]:
        self._check(generation)
        return (self.training[generation], *(self.extra[generation] if self.extra else ()))

    def to_dict(self) -> dict:
        return {"run_seed": self.run_seed, "seeds_per_generation": self.seeds_per_generation, "training": list(self.training),
                "training_per_generation": [list(self.seeds_for_generation(g)) for g in range(len(self.training))],
                "validation": list(self.validation)}

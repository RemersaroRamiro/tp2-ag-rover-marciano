"""The terrain seeds of one evolution run.

Terrain relief and sections are fixed; the seed only changes the rocks and the sub-metre roughness (``track_rocks``,
``track_terrain``). A run uses

* ``seeds_per_generation`` TRAINING seeds per generation (1 by default here, 2 in the presets): every rover of a
  generation is scored on all of them, so a rover cannot be evolved to one particular arrangement of rocks, and one
  boulder that stops the whole population cannot make the selection of that generation blind, and
* a few VALIDATION seeds, never used during the run, on which the best genomes are judged at the end.

Everything derives from the run seed alone (through ``numpy.random.SeedSequence``, in streams separate from the GA's
own rng, which uses the run seed directly), so a run is reproducible and two runs with different seeds see different
terrains. The training seeds are drawn one by one, so the seeds of the first generations do not depend on how many
generations follow; validation seeds skip any value a training seed took, so they are disjoint by construction.

The FIRST seed of each generation (``training``, the seed that is recorded for replay) comes from the same stream as
when a generation had a single seed, so a schedule with ``seeds_per_generation=1`` is exactly the one it always was, and
the validation seeds are the same whatever the number of training seeds (runs stay comparable). The extra seeds of a
generation come from a stream of their own. Every training seed differs from every other one (of any generation) and
from every validation seed.
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
    training: tuple[int, ...]  # the FIRST terrain seed of generation g is training[g] (the one recorded for replay)
    validation: tuple[int, ...]  # unseen during the run
    seeds_per_generation: int = 1
    extra: tuple[tuple[int, ...], ...] = ()  # per generation, its seeds after the first (empty without extra seeds)

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

    def for_generation(self, generation: int) -> int:
        """The terrain seed of the generation that is recorded for replay: its first seed."""
        self._check(generation)
        return self.training[generation]

    def seeds_for_generation(self, generation: int) -> tuple[int, ...]:
        """Every terrain seed the generation is scored on, the recorded one first."""
        self._check(generation)
        return (self.training[generation], *(self.extra[generation] if self.extra else ()))

    def to_dict(self) -> dict:
        return {"run_seed": self.run_seed, "seeds_per_generation": self.seeds_per_generation, "training": list(self.training),
                "training_per_generation": [list(self.seeds_for_generation(g)) for g in range(len(self.training))],
                "validation": list(self.validation)}

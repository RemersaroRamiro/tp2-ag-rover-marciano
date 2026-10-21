"""Baselines a GA run is compared with: the reference design and random search with the GA's budget.

    uv run python evolve.py --preset normal --seed 1 --out runs/ga-seed1                         # the GA
    uv run python evolve.py --random-search --preset normal --seed 1 --out runs/random-seed1     # the same budget, at random
    uv run python evolve.py --summarize runs/ga-seed1 runs/random-seed1                          # side by side

* **Reference design** (Tabla 4 of the partial delivery): every run, GA or baseline, validates the Perseverance-like
  genome (``reference_rovers.PERSEVERANCE_LIKE``, about 1 t, legal) on its own validation terrains, in the same batch
  as its best designs (``evolve.validate_designs``; ``reference`` in ``run.json``). A run's validation terrains depend
  only on its seed, generations, terrains per generation and number of validation terrains, so the GA run and the
  random search of the same command see the same ones, and the reference's result is the same in both.
* **Random search** (``run_random_search``, ``evolve.py --random-search``): the GA's evaluation budget spent on
  uniform random designs. Batch ``g`` holds ``population`` genomes uniform in [0, 1] per gene, scored with the same
  Deb fitness on the same terrain seeds as GA generation ``g`` of the same run seed, through the same worker pool. The
  genomes come from ``numpy.random.default_rng(run seed)``, the GA's own stream, so batch 0 is the GA's initial
  population (without ``--seed-reference``) and the two methods start from the same designs. At the end its best
  designs over all batches by training fitness (``random_best`` = the best ever, then ``random_2``, ``random_3``)
  and the reference are validated on the run's validation terrains, and the success flag is computed for
  ``random_best``, exactly as for the GA's ``final_best``.

A random-search folder is a run folder: ``run.json`` (``kind: random_search``, the budget under ``random_search``),
``generations.csv`` with the columns of ``ga.CSV_COLUMNS`` (per batch: best, mean, median and std of the fitness,
diversity of the batch; ``mutation_sigma`` is ``nan``: there is no mutation), ``best_genomes.json``,
``populations.npz`` and the two charts of ``plot_run`` (titles suffixed "búsqueda aleatoria"; the sigma line says
"sin datos"). It records no replay. ``evolve.py --summarize`` and ``--runs N`` work on it as on a GA run.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import numpy as np

import evolve
import ga
import replay as replay_files


class RandomSearchRun(evolve._Run):
    """A random-search run: ``evolve._Run`` with its search, its candidates and its manifest replaced."""

    kind = "random_search"
    unit = "lote"
    records_replay = False
    chart_title = "búsqueda aleatoria"

    def __init__(self, config: evolve.EvolveConfig, out_dir: Path, pool, log):
        # The batches are scored by the GA's ``GenerationEvaluator``; with this policy it asks for no replay frames.
        super().__init__(dataclasses.replace(config, replay=replay_files.ReplayPolicy(mode="none")), out_dir, pool, log)

    def method_manifest(self) -> dict:
        p = self.config.preset
        return {"random_search": {
            "method": "random search: every batch is population genomes uniform in [0, 1] per gene; nothing is learnt "
                      "from one batch to the next",
            "designs_per_batch": p.population, "batches": p.generations, "design_evaluations": p.population * p.generations,
            "seeds_per_batch": p.seeds_per_generation, "simulations": p.evaluations,
            "genomes": "numpy.random.default_rng(run seed), the GA's own stream: batch 0 is the GA's initial population "
                       "(without --seed-reference)",
            "training_seeds": "batch g is scored on the terrain seeds of GA generation g of the same run seed "
                              "(seeds.training_per_generation), with the same fitness",
            "validated": "the validation genomes best designs of all batches by training fitness (random_best = the best "
                         "ever) and the reference design, on the run's validation seeds"}}

    def announce(self) -> None:
        cfg, p = self.config, self.config.preset
        per_batch = f" x {p.seeds_per_generation} semillas por lote" if p.seeds_per_generation > 1 else ""
        self.say(f"Busqueda aleatoria (linea de base): preset {p.name} ({p.population} disenos x {p.generations} lotes = "
                 f"{p.population * p.generations} evaluaciones de disenos{per_batch} = {p.evaluations} simulaciones), "
                 f"semilla {cfg.seed}, {self.manifest['environment']['workers']} procesos")
        self.say("Mismo presupuesto que el AG con estos parametros: genes uniformes en [0, 1]; el lote g se puntua en los "
                 "terrenos de la generacion g del AG")
        self.say(f"Carpeta: {self.out_dir}")

    def search(self) -> ga.RunHistory:
        cfg = self.config
        ga_config = cfg.ga_config()  # only its population size and seed are used
        rng = np.random.default_rng(cfg.seed)  # the GA's own stream (``ga.run``): batch 0 is its initial population
        no_mutation = ga.GAState(sigma=math.nan)
        start, records = time.perf_counter(), []
        for g in range(cfg.preset.generations):
            batch = ga.initial_population(ga_config, rng)  # population x genes, uniform in [0, 1]
            record = ga.make_record(g, batch, self.batches(batch), no_mutation, time.perf_counter() - start)
            records.append(record)
            self.log(record)
        return ga.RunHistory(config=ga_config, records=tuple(records))

    def candidates(self, history: ga.RunHistory) -> list[tuple[str, np.ndarray, float]]:
        """The best designs of all batches by training fitness; the first one is the best ever."""
        genomes = np.concatenate([r.population for r in history.records])
        fitness = np.concatenate([r.fitness for r in history.records])
        return evolve.best_designs(genomes, fitness, self.config.validation_genomes, "random")

    def final_summary(self, history: ga.RunHistory) -> dict:
        """A random search has no final population: its result is its best design ever."""
        ever = history.best_record
        return {"generation": history.records[-1].generation, "best_fitness": ever.best_fitness,
                "best_genome": ever.best_genome.tolist(), "best_details": ever.best_details,
                "best_ever_generation": ever.generation, "best_ever_fitness": ever.best_fitness}


def run_random_search(config: evolve.EvolveConfig, out_dir: str | Path, evaluator=None, log=evolve.say) -> evolve.RunResult:
    """One random-search run into ``out_dir`` (new or empty) with the budget, terrains and validation of the GA run of
    ``config``. ``evaluator`` is a worker pool to reuse; by default one is made for this run and closed at its end."""
    if config.seed_reference:
        raise ValueError("random search has no initial population to seed: seed_reference must be False")
    return evolve.execute(RandomSearchRun, config, out_dir, evaluator, log)

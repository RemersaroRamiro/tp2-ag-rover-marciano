"""
baselines.py: la búsqueda aleatoria, la línea de base con la que se compara el AG.

Para qué existe
    Para afirmar que el AG aporta algo hay que compararlo con un método más simple que use el mismo
    presupuesto. Este archivo gasta la misma cantidad de simulaciones en diseños elegidos al azar.

Qué contiene
    - RandomSearchRun: una corrida de búsqueda aleatoria, con la misma estructura que una del AG.
    - run_random_search: ejecuta esa corrida y guarda su carpeta de resultados.

Cómo funciona
    En cada lote se generan al azar (uniformes en [0, 1]) tantos genomas como individuos tiene la
    población del AG, y se evalúan con la misma aptitud y en los mismos terrenos que la generación
    equivalente del AG. Al final, los 3 mejores diseños de todos los lotes y el rover de referencia se
    validan en los mismos terrenos no vistos que usa el AG. No guarda grabaciones (replay).
    uv run python codigo/evolve.py --random-search --preset normal --seed 1

Dónde encaja
    Etapa: resultados. Reutiliza evolve (el ciclo de una corrida), ga y replay.
    evolve.py lo llama con la opción --random-search.
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
    kind = "random_search"
    unit = "lote"
    records_replay = False
    chart_title = "búsqueda aleatoria"

    def __init__(self, config: evolve.EvolveConfig, out_dir: Path, pool, log):
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
        ga_config = cfg.ga_config()
        rng = np.random.default_rng(cfg.seed)
        no_mutation = ga.GAState(sigma=math.nan)
        start, records = time.perf_counter(), []
        for g in range(cfg.preset.generations):
            batch = ga.initial_population(ga_config, rng)
            record = ga.make_record(g, batch, self.batches(batch), no_mutation, time.perf_counter() - start)
            records.append(record)
            self.log(record)
        return ga.RunHistory(config=ga_config, records=tuple(records))

    def candidates(self, history: ga.RunHistory) -> list[tuple[str, np.ndarray, float]]:
        genomes = np.concatenate([r.population for r in history.records])
        fitness = np.concatenate([r.fitness for r in history.records])
        return evolve.best_designs(genomes, fitness, self.config.validation_genomes, "random")

    def final_summary(self, history: ga.RunHistory) -> dict:
        ever = history.best_record
        return {"generation": history.records[-1].generation, "best_fitness": ever.best_fitness,
                "best_genome": ever.best_genome.tolist(), "best_details": ever.best_details,
                "best_ever_generation": ever.generation, "best_ever_fitness": ever.best_fitness}


def run_random_search(config: evolve.EvolveConfig, out_dir: str | Path, evaluator=None, log=evolve.say) -> evolve.RunResult:
    if config.seed_reference:
        raise ValueError("random search has no initial population to seed: seed_reference must be False")
    return evolve.execute(RandomSearchRun, config, out_dir, evaluator, log)

"""Evolve rovers headless, record the run for replay, validate the best on unseen terrains.

    uv run python evolve.py --preset rapido --seed 1
    uv run python evolve.py --preset normal --seed 1 --runs 3 --out runs/tp      # three independent runs, seeds 1 2 3
    uv run python evolve.py --summarize runs/tp                                  # compare finished runs again

A RUN is a preset (population x generations) plus a run seed and the GA's methods (``--selection``, ``--crossover``,
``--mutation-rate``, ``--sigma-min`` ...; the defaults are Tabla 1 of the partial delivery, see ``ga.GAConfig``; with
``--preset normal`` the whole of Tabla 1). Everything random derives from the run seed (the GA's
own rng, and the terrain seeds through ``seed_schedule``), so the same run is reproducible, and it gives the same
history for any number of workers. Per generation:

1. the generation's terrain seeds (``seeds_per_generation``, 2 in the presets) pick the rocks and the roughness
   (relief and sections are fixed). One seed can be a wall: a single boulder stopped the whole population at the same x
   in 5 of 20 generations, every rover scored the same, and that generation's selection was blind;
2. every genome of the population, elites included, is decoded, simulated on EACH of those tracks (population x seeds
   jobs as one batch for the worker pool) and scored by ``fitness.fitness_from_summaries`` (Deb's feasibility rules:
   legal designs by their MEAN distance over the seeds, illegal ones by how little they violate; legality is the
   design's, the same on every seed). Nothing is cached between generations, so the elites are judged afresh on each
   new set of tracks;
3. the GA (DEAP operators, ``ga.next_generation``) breeds the next population; the mutation step adapts to the
   population's convergence (its diversity against generation 0's), never to the fitness history, which a new track
   every generation turns into noise;
4. the generation is recorded: the GA logs, and the replay files of ``replay.py`` (frames of every rover on the
   generation's FIRST seed, summaries only on the others, so a generation's files stay the size they had with one
   seed).

At the end the final population's best genomes are validated on the run's unseen validation seeds (mean / min / max
distance and statuses), which says how far a rover evolved on many tracks drives on a track it never met. The
Perseverance-like reference design is validated with them, on the same terrains and in the same batch (``reference``
in ``run.json``: the baseline a good design has to beat), and the run's success flag is computed (partial delivery,
section 6.3: the best design, the final population's best, is legal and finishes on at least 3 of the 5 validation
terrains; ``success`` in ``run.json``, with the finishes of the final top 3 and the same flag for the reference).

``--random-search`` runs the other baseline (``baselines.py``) instead of the GA: the same budget, terrains and
validation, with uniform random designs.

Files of a run directory: ``run.json`` (manifest: rules, seeds, timings, per-generation summary, validation, success;
refreshed after every generation), ``generations.csv``, ``best_genomes.json``, ``populations.npz`` (``ga.RunLogger``),
``replay/gen_NNNN.npz|json`` and, once the run is finished, its charts ``generations_fitness.png`` and
``generations_diversity.png`` (``plot_run``; ``uv run python plot_run.py RUN_DIR`` draws them again from the CSV). Fitness values of different generations are scored on different tracks: compare them as
a trend, not as a monotone curve (the best of one generation may be lower than the previous one's).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import evolution_params as params
import ga
import genome_spec
import presets
import reference_rovers
import replay as replay_files
from evaluate import SHARED_TRACKS_KEPT, EvalConfig, Evaluator, resolve_workers
from fitness import FitnessParams, fitness_from_summary, fitness_from_summaries, stop_cluster_fraction
from seed_schedule import SeedSchedule


# -- configuration ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolveConfig:
    """One run: the preset, the run seed and every rule. ``workers=None`` = one process per CPU core."""

    preset: presets.Preset = presets.PRESETS["rapido"]
    seed: int = 1
    workers: int | None = None
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    fitness_params: FitnessParams = field(default_factory=FitnessParams)
    replay: replay_files.ReplayPolicy = field(default_factory=replay_files.ReplayPolicy)
    validation_seeds: int = params.VALIDATION_SEEDS
    validation_genomes: int = params.VALIDATION_GENOMES
    seed_reference: bool = False  # put the Perseverance-like genome into generation 0 (off: a clean experiment)
    # The GA's methods and parameters (selection, crossover, mutation, elitism; Tabla 1 by default). Its population size
    # and seed are not used: they come from the preset and the run seed (``ga_config``).
    ga_params: ga.GAConfig = field(default_factory=ga.GAConfig)

    @property
    def seeds_per_generation(self) -> int:
        """Terrain seeds every rover of a generation is scored on (a property of the preset; ``--seeds-per-generation``)."""
        return self.preset.seeds_per_generation

    def ga_config(self) -> ga.GAConfig:
        """``ga_params`` with the preset's population and the run seed."""
        return dataclasses.replace(self.ga_params, population_size=self.preset.population, seed=self.seed)

    def schedule(self) -> SeedSchedule:
        """The run's terrain seeds: training seeds per generation and the validation seeds. A GA run and a baseline with
        the same run seed, generations, terrains per generation and validation seeds get the same ones."""
        return SeedSchedule.for_run(self.seed, self.preset.generations, self.validation_seeds, self.seeds_per_generation)


@dataclass(frozen=True)
class RunResult:
    history: ga.RunHistory
    schedule: SeedSchedule
    out_dir: Path
    manifest: dict


# -- one generation: seed, simulate, score ------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationBatch:
    """What one generation produced: its terrain seeds (the recorded one first), the runs by seed (``summaries[s][i]`` =
    rover ``i`` on seed ``s``) and one scored ``Evaluation`` per rover."""

    generation: int
    seeds: tuple[int, ...]
    summaries: list
    evaluations: list
    wall_s: float

    @property
    def terrain_seed(self) -> int:
        """The seed whose frames are recorded for replay."""
        return self.seeds[0]


class GenerationEvaluator:
    """The batch evaluator ``ga.run`` calls once per generation, in order: it knows which generation it is in
    (hence which terrain seeds) and keeps the last batch for the recorder. Holds no fitness from earlier generations."""

    def __init__(self, pool, schedule: SeedSchedule, config: EvolveConfig):
        self.pool, self.schedule, self.config = pool, schedule, config
        self.generation = 0
        self.last: GenerationBatch | None = None

    def _prefetch(self, seeds, rules, keep) -> None:
        for seed in seeds:
            self.pool.prefetch(seed, rules, keep=keep)

    def __call__(self, population: np.ndarray) -> list[ga.Evaluation]:
        cfg, g = self.config, self.generation
        seeds = self.schedule.seeds_for_generation(g)
        rules = cfg.eval_config
        # Tracks are built in the background while workers simulate: this generation's (in case nobody asked yet), then
        # the next generation's (every validation track after the last one). The disk room covers every track that has
        # to wait together, so asking for one never deletes another that is still needed: both generations' tracks, or
        # the last generation's plus every validation track. With one seed that is the default room of two.
        room = max(SHARED_TRACKS_KEPT, 2 * len(seeds))
        keep = room if room > SHARED_TRACKS_KEPT else None
        self._prefetch(seeds, rules, keep)
        if g + 1 < len(self.schedule.training):
            self._prefetch(self.schedule.seeds_for_generation(g + 1), rules, keep)
        else:
            self._prefetch(self.schedule.validation, rules, max(SHARED_TRACKS_KEPT, len(seeds) + len(self.schedule.validation)))
        hz = cfg.replay.hz if cfg.replay.records_generation(g, cfg.preset.generations) else None
        rates = [hz] + [None] * (len(seeds) - 1)  # frames of the first seed only; the others keep their summaries
        t0 = time.perf_counter()
        by_seed = self.pool.evaluate_seeds(population, seeds, rules, rates)  # ONE batch: population x seeds jobs
        evaluations = [fitness_from_summaries([runs[i] for runs in by_seed], cfg.fitness_params) for i in range(len(population))]
        self.last = GenerationBatch(g, seeds, by_seed, evaluations, time.perf_counter() - t0)
        self.generation += 1
        return evaluations


# -- the manifest -----------------------------------------------------------------------------------------------


def genome_spec_info() -> dict:
    genes = [{"name": g.name, "low": g.low, "high": g.high, "scale": g.scale} for g in genome_spec.GENES]
    digest = hashlib.sha1(json.dumps(genes, sort_keys=True).encode()).hexdigest()[:12]
    return {"length": genome_spec.GENOME_LENGTH, "hash": digest, "genes": genes}


def git_info() -> dict:
    """Commit of the code that ran (and whether it had uncommitted changes); nulls when git is not available."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=10, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def environment_info(workers: int | None, jobs: int) -> dict:
    import os

    import mujoco

    return {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(), "workers": resolve_workers(workers, jobs)}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def status_label(details: dict) -> str:
    """``finished`` or ``dead:<reason>``."""
    return "finished" if details["finished"] else f"dead:{details['reason']}"


def joined(values, spec: str = "", percent: bool = False) -> str:
    """Values of the seeds of a generation on one console line: ``a + b`` (``percent``: shares as ``25% + 8%``)."""
    if percent:
        return " + ".join(f"{100 * v:.0f}%" for v in values)
    return " + ".join(format(v, spec) for v in values)


# -- validation: the best designs and the reference on the unseen terrains ----------------------------------------------

REFERENCE_LABEL = "reference"
BEST_EVER_LABEL = "best_ever"


def reference_genome() -> np.ndarray:
    """The Perseverance-like reference design (Tabla 4 of the partial delivery): the baseline every run is validated with."""
    return reference_rovers.genome(reference_rovers.PERSEVERANCE_LIKE)


def best_designs(genomes: np.ndarray, fitness: np.ndarray, count: int, prefix: str) -> list[tuple[str, np.ndarray, float]]:
    """The ``count`` best distinct ``genomes`` by ``fitness`` (a non-finite fitness counts as the worst; ties keep their
    order), as (label, genome, fitness) with labels ``<prefix>_best``, ``<prefix>_2``, ``<prefix>_3`` ..."""
    fitness = np.asarray(fitness, dtype=float)
    order = np.argsort(-np.where(np.isfinite(fitness), fitness, -np.inf), kind="stable")
    chosen: list[tuple[str, np.ndarray, float]] = []
    for index in order:
        if len(chosen) == count:
            break
        genome = genomes[index]
        if not any(np.array_equal(genome, c[1]) for c in chosen):
            chosen.append((f"{prefix}_best" if not chosen else f"{prefix}_{len(chosen) + 1}", genome, float(fitness[index])))
    return chosen


def validation_entry(label: str, genome: np.ndarray, training_fitness: float | None, seeds, evals) -> dict:
    """What one design did on the validation seeds (``evals``: one scored run per seed, in order)."""
    dist = [e.details["distance_m"] for e in evals]
    fit = [e.fitness for e in evals]
    first = evals[0].details
    return {
        "label": label, "genome": np.asarray(genome).tolist(), "decoded": genome_spec.decode(genome), "training_fitness": training_fitness,
        "seeds": list(seeds), "distances_m": dist, "fitness": fit, "statuses": [status_label(e.details) for e in evals],
        "finished_by_seed": [bool(e.details["finished"]) for e in evals],
        "sections": [e.details["section_reached"] for e in evals], "finished": sum(e.details["finished"] for e in evals),
        "mean_distance_m": float(np.mean(dist)), "min_distance_m": min(dist), "max_distance_m": max(dist),
        "mean_fitness": float(np.mean(fit)), "total_violation": first["total_violation"], "legal": first["legal"],
        "violation_codes": first["violation_codes"]}


def validate_designs(pool, config: EvolveConfig, seeds, candidates: list[tuple[str, np.ndarray, float]]) -> dict:
    """``candidates`` (label, genome, training fitness) and the reference design on the validation ``seeds``, as ONE batch
    for the pool (a task per seed and design, so the reference costs no extra batch). Returns ``seeds``, ``candidates``
    (one ``validation_entry`` each, in order) and ``reference`` (``None`` without validation seeds)."""
    seeds = list(seeds)
    designs = [*candidates, (REFERENCE_LABEL, reference_genome(), None)]
    results: dict[str, list] = {label: [] for label, _, _ in designs}
    if seeds:
        per_seed = pool.evaluate_seeds(np.array([g for _, g, _ in designs]), seeds, config.eval_config)
        for summaries in per_seed:
            for (label, _, _), s in zip(designs, summaries):
                results[label].append(fitness_from_summary(s, config.fitness_params))
    entries = {label: validation_entry(label, genome, fitness, seeds, results[label])
               for label, genome, fitness in designs if results[label]}
    return {"seeds": seeds, "candidates": [entries[label] for label, _, _ in candidates if label in entries],
            "reference": entries.get(REFERENCE_LABEL)}


# -- the success criterion (partial delivery, section 6.3) ---------------------------------------------------------------

SUCCESS_CRITERION = (
    f"partial delivery, section 6.3: the best design is legal (feasible) and finishes on at least "
    f"{params.SUCCESS_MIN_FINISHES} of {params.SUCCESS_OF_SEEDS} validation terrains (with another number of validation "
    f"terrains, the same share rounded up). The best design of a GA run is the final population's best (final_best); of a "
    f"random search, its best design by training fitness (random_best)")


def required_finishes(seeds: int) -> int:
    """Finishes the success criterion asks for on ``seeds`` validation terrains: 3 of 5, the same share rounded up otherwise."""
    return -(-params.SUCCESS_MIN_FINISHES * seeds // params.SUCCESS_OF_SEEDS)


def meets_success_criterion(legal: bool, finished: int, seeds: int) -> bool:
    """A legal design that finished on at least ``required_finishes(seeds)`` of its ``seeds`` validation terrains."""
    return bool(legal) and seeds > 0 and finished >= required_finishes(seeds)


def success_flag(entry: dict) -> dict:
    seeds = len(entry["seeds"])
    return {"label": entry["label"], "legal": entry["legal"], "finished": entry["finished"], "seeds": seeds,
            "success": meets_success_criterion(entry["legal"], entry["finished"], seeds)}


def success_report(validation: dict) -> dict | None:
    """The success flag of the run's best design (its first validated candidate), the finishes of its top designs
    (``top3``: every candidate but the best-ever one) and the reference's flag; ``None`` without validation seeds."""
    seeds = len(validation["seeds"])
    if not seeds:
        return None
    candidates, reference = validation["candidates"], validation.get("reference")
    return {"criterion": SUCCESS_CRITERION, "required_finishes": required_finishes(seeds), "validation_seeds": seeds,
            "best": success_flag(candidates[0]) if candidates else None,
            "top3": [success_flag(c) for c in candidates if c["label"] != BEST_EVER_LABEL],
            "reference": success_flag(reference) if reference else None}


def fmt_distance(entry: dict) -> str:
    return f"{entry['mean_distance_m']:.1f} m (min {entry['min_distance_m']:.1f}, max {entry['max_distance_m']:.1f})"


def log_validation(validation: dict, log) -> None:
    """The validation on the console: one line per design, the reference last with its outcome on every terrain."""
    reference = validation.get("reference")
    rows = [(c["label"], c) for c in validation["candidates"]] + ([("referencia", reference)] if reference else [])
    for label, c in rows:
        log(f"  {label:<10}: distancia media {fmt_distance(c)} | llegaron {c['finished']}/{len(c['seeds'])} | "
            f"{'legal' if c['legal'] else 'ilegal, violacion ' + format(c['total_violation'], '.3f')} | "
            f"estados {dict(Counter(c['statuses']))}")
    if reference:
        log("    referencia por terreno: " + " | ".join(
            f"{seed} {d:.1f} m {status}" for seed, d, status in zip(reference["seeds"], reference["distances_m"], reference["statuses"])))


def log_success(report: dict | None, log) -> None:
    if report is None:
        return
    n = report["validation_seeds"]

    def verdict(flag: dict) -> str:
        return f"{'SI' if flag['success'] else 'NO'} (llego en {flag['finished']}/{n}, {'legal' if flag['legal'] else 'ilegal'})"

    log(f"Criterio de exito (diseno legal que llega al final en >= {report['required_finishes']} de {n} terrenos de validacion):")
    if report["best"]:
        log(f"  mejor diseno ({report['best']['label']}): {verdict(report['best'])}")
    if report["top3"]:
        log("  top-3, llegaron: " + ", ".join(f"{t['label']} {t['finished']}/{n}" for t in report["top3"]))
    if report["reference"]:
        log(f"  referencia (tipo Perseverance): {verdict(report['reference'])}")


# -- a run ------------------------------------------------------------------------------------------------------


class _Run:
    """State of one GA run in progress. Its ``log`` is what ``ga.run`` calls after every generation.

    ``execute`` drives a run through its steps (``announce``, ``search``, ``validate``, ``finish``); a baseline
    (``baselines.RandomSearchRun``) is a subclass that replaces the search and keeps the seeds, the scoring, the files,
    the validation and the success flag."""

    kind = "ga"  # ``kind`` in run.json
    unit = "gen"  # first word of a generation's console line
    records_replay = True
    chart_title = ""  # suffix of the charts' titles

    def __init__(self, config: EvolveConfig, out_dir: Path, pool, log):
        self.config, self.pool, self.say = config, pool, log
        self.schedule = config.schedule()
        self.logger = ga.RunLogger(out_dir)  # claims the directory: refuses a non-empty one
        self.out_dir = self.logger.out_dir
        self.batches = GenerationEvaluator(pool, self.schedule, config)
        self.t0 = time.perf_counter()
        self.manifest = {
            "status": "running", "kind": self.kind, "run_seed": config.seed, "started_at": _now(), "finished_at": None,
            "preset": {"name": config.preset.name, "population": config.preset.population,
                       "generations": config.preset.generations, "seeds_per_generation": config.preset.seeds_per_generation,
                       "description": config.preset.description},
            **self.method_manifest(),
            "seeds": self.schedule.to_dict(),
            "fitness": {
                "rule": "Deb (2000) feasibility rules: a legal design beats an illegal one; legal designs by larger distance; "
                        "illegal designs by smaller total violation. Legality is the design's: the same on every terrain seed",
                "formula": "legal: fitness = mean over the generation's terrain seeds of max_distance_m (metres); "
                           "illegal: fitness = -(infeasible_offset + total_violation), dimensionless, "
                           "total_violation = sum(violation.magnitude)",
                "seeds_per_generation": config.seeds_per_generation,
                "infeasible_offset": config.fitness_params.infeasible_offset},
            "eval_rules": dataclasses.asdict(config.eval_config),
            "validation_rules": {"seeds": config.validation_seeds, "genomes": config.validation_genomes},
            "genome_spec": genome_spec_info(),
            "git": git_info(),
            "environment": environment_info(config.workers, config.preset.population * config.seeds_per_generation),  # jobs per batch
            "generations": [], "validation": None, "success": None, "final": None, "timings": None, "sizes": None,
        }
        self.write_manifest()

    def method_manifest(self) -> dict:
        """The part of run.json that describes the search method: the GA's methods and parameters."""
        cfg = self.config
        return {"seed_reference": cfg.seed_reference, "ga": dataclasses.asdict(cfg.ga_config()),
                "ga_operators": ga.describe_operators(cfg.ga_config()), "replay": dataclasses.asdict(cfg.replay)}

    # -- files
    def write_manifest(self) -> dict:
        self.manifest = ga.jsonable(self.manifest)
        (self.out_dir / "run.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return self.manifest

    def replay_sizes(self) -> list[int]:
        folder = self.out_dir / "replay"
        return [p.stat().st_size for p in sorted(folder.iterdir())] if folder.exists() else []

    def write_replay(self, record: ga.GenerationRecord, batch: GenerationBatch) -> int:
        """The generation's replay files; returns their size in bytes."""
        cfg = self.config
        npz = replay_files.write_generation(self.out_dir, record.generation, batch.terrain_seed, record.population, batch.summaries[0],
                                            batch.evaluations, cfg.replay, cfg.preset.generations,
                                            other_seeds=list(zip(batch.seeds[1:], batch.summaries[1:])))
        return sum(p.stat().st_size for p in (npz, npz.with_suffix(".json")))

    # -- the steps ``execute`` drives
    def announce(self) -> None:
        cfg, p = self.config, self.config.preset
        per_generation = f" x {p.seeds_per_generation} semillas por generacion" if p.seeds_per_generation > 1 else ""
        self.say(f"Evolucion: preset {p.name} ({p.population} rovers x {p.generations} generaciones{per_generation} = {p.evaluations} simulaciones), "
                 f"semilla {cfg.seed}, {self.manifest['environment']['workers']} procesos")
        self.say(f"AG (DEAP): {ga_summary(cfg.ga_config())}")
        self.say(f"Carpeta: {self.out_dir}")

    def search(self) -> ga.RunHistory:
        cfg = self.config
        seeds = [reference_genome()] if cfg.seed_reference else None
        return ga.run(self.batches, cfg.ga_config(), cfg.preset.generations, logger=self, seeds=seeds)

    # -- called by ga.run after every generation
    def log(self, record: ga.GenerationRecord) -> None:
        cfg, batch = self.config, self.batches.last
        assert batch is not None and batch.generation == record.generation
        self.logger.log(record)
        replay_bytes = self.write_replay(record, batch) if self.records_replay else 0
        details = [e.details for e in batch.evaluations]
        finished, legal = sum(d["finished"] for d in details), sum(d["legal"] for d in details)
        best = record.best_details
        # Blind-generation indicator: the share of the population within 1 m of the most common stop point, on each seed and
        # on the averaged distance that is scored. Near 1 on a seed = one obstacle stopped almost everybody there.
        clusters = [stop_cluster_fraction([s.max_distance_m for s in runs]) for runs in batch.summaries]
        scored_cluster = stop_cluster_fraction([d["distance_m"] for d in details])
        finished_by_seed = [sum(s.status == "finished" for s in runs) for runs in batch.summaries]
        self.manifest["generations"].append({
            "generation": record.generation, "terrain_seed": batch.terrain_seed, "terrain_seeds": list(batch.seeds),
            "best": record.best_fitness,
            "best_distance_m": best["distance_m"], "best_distances_m": best["distances_m"],
            "best_total_violation": best["total_violation"], "best_legal": best["legal"],
            "best_status": status_label(best), "best_statuses": [status_label(p) for p in best["per_seed"]],
            "best_section": best["section_reached"], "best_sections": [p["section_reached"] for p in best["per_seed"]],
            "mean": record.mean_fitness, "median": record.median_fitness,
            "finished": finished, "finished_by_seed": finished_by_seed, "legal": legal,
            "stop_cluster_by_seed": clusters, "stop_cluster_scored": scored_cluster,
            "diversity": record.diversity, "sigma": record.sigma,
            "eval_s": batch.wall_s, "elapsed_s": record.elapsed, "replay_bytes": replay_bytes})
        self.write_manifest()
        n, width, many = len(details), len(str(cfg.preset.generations)), len(batch.seeds) > 1
        sigma = f" | sigma {record.sigma:.2f}" if np.isfinite(record.sigma) else ""  # a baseline has no mutation step
        self.say(
            f"{self.unit} {record.generation + 1:>{width}}/{cfg.preset.generations} | {'semillas' if many else 'semilla'} {joined(batch.seeds)} | "
            f"mejor {record.best_fitness:7.1f} ({'legal' if best['legal'] else 'ilegal'}, dist {joined(best['distances_m'], '.1f')} m, "
            f"violacion {best['total_violation']:.3f}; estado {joined(status_label(p) for p in best['per_seed'])}, "
            f"tramo {joined(p['section_reached'] for p in best['per_seed'])}) | "
            f"media {record.mean_fitness:7.1f} | "
            f"mediana {record.median_fitness:7.1f} | llegaron {finished}/{n}{f' ({joined(finished_by_seed)} por semilla)' if many else ''} | "
            f"legales {legal}/{n} | mismo punto (+-1 m) {joined(clusters, percent=True)}"
            f"{f' (promedio {100 * scored_cluster:.0f}%)' if many else ''} | "
            f"diversidad {record.diversity:.3f}{sigma} | {batch.wall_s:.1f} s")

    # -- end of run
    def candidates(self, history: ga.RunHistory) -> list[tuple[str, np.ndarray, float]]:
        """The designs to validate: the best genomes of the final population (``final_best`` first: the run's best design
        for the success criterion) and the best-ever one."""
        cfg, last = self.config, history.records[-1]
        chosen = best_designs(last.population, last.fitness, cfg.validation_genomes, "final")
        ever = history.best_record
        if cfg.validation_genomes > 0 and not any(np.array_equal(ever.best_genome, c[1]) for c in chosen):
            chosen.append((BEST_EVER_LABEL, ever.best_genome, ever.best_fitness))
        return chosen

    def validate(self, history: ga.RunHistory) -> dict:
        """The candidates and the reference design on the unseen validation seeds."""
        return validate_designs(self.pool, self.config, self.schedule.validation, self.candidates(history))

    def final_summary(self, history: ga.RunHistory) -> dict:
        last, ever = history.records[-1], history.best_record
        return {"generation": last.generation, "best_fitness": last.best_fitness, "best_genome": last.best_genome.tolist(),
                "best_details": last.best_details, "best_ever_generation": ever.generation, "best_ever_fitness": ever.best_fitness}

    def finish(self, history: ga.RunHistory, validation_s: float, validation: dict) -> dict:
        m = self.manifest
        sizes = [g["replay_bytes"] for g in m["generations"]]
        m["validation"] = validation
        m["success"] = success_report(validation)
        m["final"] = self.final_summary(history)
        batch_times = [g["eval_s"] for g in m["generations"]]
        m["timings"] = {"total_s": time.perf_counter() - self.t0, "evaluation_s": sum(batch_times), "per_generation_s": batch_times,
                        "validation_s": validation_s}
        m["sizes"] = {"replay_bytes_total": sum(self.replay_sizes()), "replay_bytes_per_generation_median": float(np.median(sizes)),
                      "replay_bytes_per_generation_max": max(sizes)}
        m["status"], m["finished_at"] = "finished", _now()
        return self.write_manifest()

    def abort(self, status: str, error: str | None = None) -> None:
        self.manifest["status"], self.manifest["finished_at"] = status, _now()
        if error:
            self.manifest["error"] = error
        self.manifest["timings"] = {"total_s": time.perf_counter() - self.t0}
        self.write_manifest()


def say(message: str = "") -> None:
    """Console output, flushed at once (a long run is usually watched through a pipe or a log file)."""
    print(message, flush=True)


def ga_summary(cfg: ga.GAConfig) -> str:
    """The GA's methods and parameters on one console line."""
    select = f"torneo k={cfg.tournament_size}" if cfg.selection == "tournament" else "ruleta (aptitud desplazada)"
    cross = f"BLX-alfa {cfg.blx_alpha:g}" if cfg.crossover == "blx" else f"binomial (por gen {cfg.binomial_indpb:g})"
    sigma = (f"sigma {cfg.mutation_sigma:g}-{cfg.sigma_max:g} adaptativa" if cfg.adaptive_mutation
             else f"sigma fija {cfg.mutation_sigma:g}")
    return (f"seleccion {select} | cruce {cross}, p {cfg.crossover_rate:g} | mutacion gaussiana por gen "
            f"{cfg.mutation_gene_rate:g}, {sigma} | elite {cfg.elite_count}")


def write_charts(out_dir: Path, log=say, title_suffix: str = "") -> list[Path]:
    """The charts of a finished run (``plot_run``: fitness, and diversity with sigma) next to its ``generations.csv``.
    A plotting failure is only a warning: the run is already finished and saved, and the charts can be drawn again
    from the CSV alone (``plot_run.py``)."""
    try:
        import plot_run  # matplotlib is imported here only: a missing or broken plotting stack never stops a run

        charts = plot_run.plot_run(Path(out_dir) / "generations.csv", title_suffix=title_suffix)
    except Exception as error:  # noqa: BLE001 - whatever went wrong, the finished run stays as it is
        log(f"Aviso: no se pudieron generar los graficos ({type(error).__name__}: {error}); "
            f"se regeneran con: uv run python plot_run.py {out_dir}")
        return []
    log(f"Graficos: {charts.fitness_png.name}, {charts.diversity_png.name}")
    return [charts.fitness_png, charts.diversity_png]


def execute(run_type: type[_Run], config: EvolveConfig, out_dir: str | Path, evaluator=None, log=say) -> RunResult:
    """One run of ``run_type`` (the GA's ``_Run`` or a baseline's subclass) into ``out_dir`` (new or empty): the search,
    the validation with the reference design, the success flag, the manifest and the charts. ``evaluator`` is a worker
    pool (``evaluate.Evaluator``) to reuse; by default one is made for this run and closed at its end."""
    own_pool = evaluator is None
    pool = Evaluator(workers=config.workers) if own_pool else evaluator
    run = None
    try:  # everything after the pool is taken: whatever fails, the pool is closed or its room given back (``finally``)
        run = run_type(config, Path(out_dir), pool, log)
        run.announce()
        history = run.search()
        t_val = time.perf_counter()
        log(f"Validacion en {len(run.schedule.validation)} semillas no vistas {list(run.schedule.validation)}:")
        validation = run.validate(history)
        log_validation(validation, log)
        log_success(success_report(validation), log)
        manifest = run.finish(history, time.perf_counter() - t_val, validation)
    except KeyboardInterrupt:
        if run is not None:
            run.abort("interrupted")
            log("Interrumpido: la corrida queda guardada hasta la ultima generacion completa.")
        raise
    except BaseException as error:
        if run is not None:
            run.abort("failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        if own_pool:
            pool.close()
        else:  # a pool shared with the next run: give back the track room this run widened (the last generation and the validation)
            pool.release_room()
    write_charts(run.out_dir, log, run.chart_title)  # after the run is finished and saved: a plotting failure cannot mark it failed
    replay = f"; replay {manifest['sizes']['replay_bytes_total'] / 1e6:.1f} MB" if run.records_replay else ""
    log(f"Tiempo total {manifest['timings']['total_s']:.0f} s{replay} en {run.out_dir}")
    return RunResult(history, run.schedule, run.out_dir, manifest)


def run_evolution(config: EvolveConfig, out_dir: str | Path, evaluator=None, log=say) -> RunResult:
    """One GA run into ``out_dir`` (new or empty). ``evaluator`` is a worker pool (``evaluate.Evaluator``) to reuse; by
    default one is made for this run and closed at its end."""
    return execute(_Run, config, out_dir, evaluator, log)


# -- several independent runs ---------------------------------------------------------------------------------------


KIND_NAMES = {"ga": "AG", "random_search": "aleatoria"}  # the ``tipo`` column of the summary table


def summary_row(manifest: dict) -> dict:
    """The numbers of one finished run that the TP compares across runs. A run saved before the success flag existed
    has no ``kind`` (it is a GA run), ``success`` or validated reference: those cells are ``None``."""
    validation = manifest.get("validation") or {}
    candidates = validation.get("candidates") or []  # empty without validation seeds
    champion = candidates[0] if candidates else {}
    final = manifest.get("final") or {}
    success = manifest.get("success") or {}
    best, reference = success.get("best") or {}, success.get("reference") or {}
    return {
        "kind": manifest.get("kind", "ga"),
        "run_seed": manifest["run_seed"], "preset": manifest["preset"]["name"], "status": manifest["status"],
        "generations": len(manifest["generations"]), "final_best_fitness": final.get("best_fitness"),
        "best_ever_fitness": final.get("best_ever_fitness"),
        "first_generation_best": manifest["generations"][0]["best"] if manifest["generations"] else None,
        "champion_mean_distance_m": champion.get("mean_distance_m"), "champion_min_distance_m": champion.get("min_distance_m"),
        "champion_max_distance_m": champion.get("max_distance_m"), "champion_finished": champion.get("finished"),
        "champion_seeds": len(champion.get("seeds", [])), "champion_statuses": champion.get("statuses"),
        "champion_legal": champion.get("legal"), "total_s": (manifest.get("timings") or {}).get("total_s"),
        "success": best.get("success"), "required_finishes": success.get("required_finishes"),
        "top3_finished": [t["finished"] for t in success.get("top3") or []],
        "reference_finished": reference.get("finished"), "reference_success": reference.get("success"),
        "reference_mean_distance_m": (validation.get("reference") or {}).get("mean_distance_m"),
    }


def _stats(values) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"mean": None, "std": None, "min": None, "max": None}
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


def aggregate(rows: list[dict]) -> dict:
    """Across runs: fitness and validation statistics, and how many runs met the success criterion (``successes`` of
    the ``judged`` ones, which had validation seeds), for their best design and for the reference."""
    flags, reference_flags = [r.get("success") for r in rows], [r.get("reference_success") for r in rows]
    return {"runs": len(rows), "final_best_fitness": _stats(r["final_best_fitness"] for r in rows),
            "champion_mean_distance_m": _stats(r["champion_mean_distance_m"] for r in rows),
            "champion_finished": sum(r["champion_finished"] or 0 for r in rows),
            "champion_evaluations": sum(r["champion_seeds"] or 0 for r in rows),
            "successes": sum(f is True for f in flags), "judged": sum(f is not None for f in flags),
            "reference_successes": sum(f is True for f in reference_flags),
            "reference_judged": sum(f is not None for f in reference_flags)}


def format_summary(rows: list[dict]) -> str:
    """A table of runs, one line each (GA runs and baselines alike), plus the means across the runs of each kind."""
    def num(x, spec="7.1f"):
        return format(x, spec) if x is not None else "    n/d"

    def flag(x):
        return "n/d" if x is None else ("si" if x else "no")

    head = ("tipo      | semilla | preset   | mejor final (m) | validacion media / min / max (m) | llegaron | exito | "
            "top-3 llegaron | referencia | estado      | tiempo (s)")
    lines = [head, "-" * len(head)]
    for r in rows:
        kind = r.get("kind", "ga")
        top3 = ", ".join(str(n) for n in r.get("top3_finished") or []) or "-"
        reference = "n/d" if r.get("reference_finished") is None else f"{r['reference_finished']}/{r['champion_seeds']} {flag(r.get('reference_success'))}"
        lines.append(
            f"{KIND_NAMES.get(kind, kind):<9} | {r['run_seed']:>7} | {r['preset']:<8} | {num(r['final_best_fitness']):>15} | "
            f"{num(r['champion_mean_distance_m'])} / {num(r['champion_min_distance_m'])} / {num(r['champion_max_distance_m'])} | "
            f"{(r['champion_finished'] if r['champion_finished'] is not None else '-')!s:>3}/{r['champion_seeds']:<4} | "
            f"{flag(r.get('success')):<5} | {top3:<14} | {reference:<10} | "
            f"{r['status']:<11} | {num(r['total_s'], '9.0f')}")
    for kind in dict.fromkeys(r.get("kind", "ga") for r in rows):  # the kinds in the order they come
        a = aggregate([r for r in rows if r.get("kind", "ga") == kind])
        mean, std = a["champion_mean_distance_m"]["mean"], a["champion_mean_distance_m"]["std"]
        if mean is None:
            continue
        text = (f"Entre {a['runs']} corridas{'' if kind == 'ga' else ' de busqueda aleatoria'}: distancia de validacion media "
                f"{mean:.1f} m (desvio {std:.1f}); llegaron {a['champion_finished']}/{a['champion_evaluations']} validaciones")
        if a["judged"]:
            text += f"; exito {a['successes']}/{a['judged']} corridas"
        if a["reference_judged"]:
            text += f" (referencia {a['reference_successes']}/{a['reference_judged']})"
        lines.append(text)
    return "\n".join(lines)


def load_summaries(path: str | Path) -> list[dict]:
    """Summary rows of the run in ``path`` (a run folder) or of every run folder inside it, by run seed."""
    path = Path(path)
    manifests = [path / "run.json"] if (path / "run.json").exists() else sorted(path.glob("*/run.json"))
    rows = [summary_row(json.loads(m.read_text(encoding="utf-8"))) for m in manifests]
    return sorted(rows, key=lambda r: r["run_seed"])


def run_many(config: EvolveConfig, runs: int, base_dir: str | Path, evaluator=None, log=say, run=run_evolution) -> list[RunResult]:
    """``runs`` independent runs with seeds ``config.seed``, ``config.seed + 1`` ... one after the other, each in
    ``base_dir/seed-N``, sharing one worker pool; ``base_dir/summary.json`` is refreshed after every run. ``run`` is the
    function of one run: the GA's ``run_evolution`` or a baseline's (``baselines.run_random_search``)."""
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    base = Path(base_dir)
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f"{base} already exists and is not empty")
    base.mkdir(parents=True, exist_ok=True)
    own_pool = evaluator is None
    pool = Evaluator(workers=config.workers) if own_pool else evaluator
    results: list[RunResult] = []
    try:
        for i in range(runs):
            cfg = dataclasses.replace(config, seed=config.seed + i)
            log(f"=== corrida {i + 1}/{runs} (semilla {cfg.seed}) ===")
            results.append(run(cfg, base / f"seed-{cfg.seed}", evaluator=pool, log=log))
            rows = [summary_row(r.manifest) for r in results]
            (base / "summary.json").write_text(
                json.dumps(ga.jsonable({"runs": rows, "aggregate": aggregate(rows)}), indent=2), encoding="utf-8")
    finally:
        if own_pool:
            pool.close()
    log("")
    log(format_summary([summary_row(r.manifest) for r in results]))
    return results


# -- command line -----------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evoluciona rovers sin graficos, graba la corrida para verla despues y la valida en terrenos no vistos.",
        epilog="Al terminar, cada corrida guarda junto a su generations.csv los graficos generations_fitness.png (aptitud "
               "maxima y promedio por generacion) y generations_diversity.png (diversidad y sigma de mutacion). Se "
               "regeneran solo desde el CSV, sin simulador: uv run python plot_run.py CARPETA (una corrida o una carpeta "
               "con varias). Cada corrida valida tambien el diseno de referencia (tipo Perseverance) en sus terrenos de "
               "validacion y marca si cumple el criterio de exito (el mejor diseno es legal y llega al final en >= 3 de 5 "
               "terrenos de validacion; run.json: reference y success).")
    names = [*presets.PRESETS, presets.SMOKE.name]
    p.add_argument("--preset", default="rapido", choices=names, help="tamano de la corrida (default: rapido)")
    p.add_argument("--seed", type=int, default=1, help="semilla de la corrida (con --runs N se usan seed, seed+1, ...)")
    p.add_argument("--runs", type=int, default=1, help="corridas independientes, una tras otra (default: 1)")
    p.add_argument("--workers", type=int, default=None, help="procesos de simulacion (default: uno por nucleo)")
    p.add_argument("--out", default=None, help="carpeta de la corrida (con --runs N: carpeta base); default runs/<preset>-...")
    p.add_argument("--seed-reference", action="store_true", help="incluye el rover tipo Perseverance en la generacion 0 (default: no)")
    p.add_argument("--population", type=int, default=None, help="cambia la poblacion del preset")
    p.add_argument("--generations", type=int, default=None, help="cambia las generaciones del preset")
    p.add_argument("--seeds-per-generation", type=int, default=None,
                   help="terrenos (semillas de rocas) en los que se puntua cada rover de una generacion; la distancia se promedia "
                        "(default: el del preset, 2; con 1 cada generacion tiene un solo terreno)")
    p.add_argument("--replay", choices=["all", "best", "none"], default="all", help="que rovers se graban para el replay (default: all)")
    p.add_argument("--replay-best-k", type=int, default=params.REPLAY_BEST_K, help="con --replay best: cuantos por generacion")
    p.add_argument("--replay-every", type=int, default=1, help="graba las trayectorias cada N generaciones (y la ultima)")
    p.add_argument("--validation-seeds", type=int, default=params.VALIDATION_SEEDS, help="semillas no vistas para validar al final (default: 5)")
    p.add_argument("--max-sim-time", type=float, default=None, help="tope de tiempo simulado por rover, en segundos (default: derivado de la pista)")
    p.add_argument("--summarize", metavar="CARPETA", nargs="+", default=None,
                   help="solo compara las corridas ya terminadas de esas carpetas (AG y busqueda aleatoria; cada una: una "
                        "corrida o una carpeta con varias), con su criterio de exito y el de la referencia")
    p.add_argument("--random-search", action="store_true",
                   help="linea de base, en vez del AG: busqueda aleatoria con el mismo presupuesto que el AG de este mismo "
                        "comando (poblacion x generaciones evaluaciones de disenos con genes uniformes en [0, 1]; el lote g se "
                        "puntua con la misma aptitud en los terrenos de la generacion g del AG de la misma semilla) y validacion "
                        "de sus mejores disenos en las mismas semillas de validacion. No graba replay; no acepta las opciones "
                        "del AG, --seed-reference ni --replay all|best")
    d = ga.GAConfig()  # the defaults: Tabla 1 of the partial delivery
    g = p.add_argument_group("algoritmo genetico (DEAP; los valores por defecto son los de la Tabla 1 de la entrega parcial)")
    g.add_argument("--selection", choices=ga.SELECTION_METHODS, default=d.selection,
                   help="seleccion: tournament = tools.selTournament, roulette = tools.selRoulette sobre la aptitud "
                        f"desplazada f - min(f) + eps (default: {d.selection})")
    g.add_argument("--tournament-size", type=int, default=d.tournament_size, metavar="K",
                   help=f"participantes de cada torneo, con --selection tournament (default: {d.tournament_size})")
    g.add_argument("--crossover", choices=ga.CROSSOVER_METHODS, default=d.crossover,
                   help=f"cruce: blx = BLX-alfa (tools.cxBlend), binomial = tools.cxUniform (default: {d.crossover})")
    g.add_argument("--blx-alpha", type=float, default=d.blx_alpha, help=f"alfa del cruce BLX (default: {d.blx_alpha})")
    g.add_argument("--binomial-indpb", type=float, default=d.binomial_indpb,
                   help=f"probabilidad de intercambiar cada gen en el cruce binomial (default: {d.binomial_indpb})")
    g.add_argument("--crossover-rate", type=float, default=d.crossover_rate,
                   help=f"probabilidad de cruzar cada pareja seleccionada (default: {d.crossover_rate})")
    g.add_argument("--mutation-rate", type=float, default=d.mutation_gene_rate,
                   help=f"probabilidad de mutar cada gen, constante (default: {d.mutation_gene_rate})")
    g.add_argument("--sigma-min", type=float, default=d.mutation_sigma,
                   help=f"sigma de la mutacion gaussiana con la poblacion diversa; sin adaptacion, el sigma fijo (default: {d.mutation_sigma})")
    g.add_argument("--sigma-max", type=float, default=d.sigma_max,
                   help=f"sigma con la poblacion convergida, con mutacion adaptativa (default: {d.sigma_max})")
    g.add_argument("--adaptive-mutation", action=argparse.BooleanOptionalAction, default=d.adaptive_mutation,
                   help="sigma segun la convergencia: sigma_min + (sigma_max - sigma_min) * (1 - D/D0), D = diversidad "
                        "(desvio medio por gen), D0 = la de la generacion 0; --no-adaptive-mutation = sigma fijo en sigma_min "
                        "(default: adaptativa)")
    g.add_argument("--elite-count", type=int, default=d.elite_count,
                   help=f"mejores individuos que pasan sin cambios a la generacion siguiente (default: {d.elite_count})")
    return p


# The options of the GA (and of its recording) that a baseline has no use for: ``--random-search`` refuses them.
GA_ONLY_OPTIONS = ("seed_reference", "replay", "replay_best_k", "replay_every", "selection", "tournament_size", "crossover",
                   "blx_alpha", "binomial_indpb", "crossover_rate", "mutation_rate", "sigma_min", "sigma_max",
                   "adaptive_mutation", "elite_count")


def config_from_args(args: argparse.Namespace) -> EvolveConfig:
    """The run the arguments describe; ``ValueError`` for an invalid preset size or GA parameter."""
    preset = presets.resolve(args.preset)
    if args.population is not None or args.generations is not None:
        preset = dataclasses.replace(preset, population=args.population or preset.population, generations=args.generations or preset.generations)
    if args.seeds_per_generation is not None:
        preset = dataclasses.replace(preset, seeds_per_generation=args.seeds_per_generation)  # validates: at least 1
    rules = EvalConfig() if args.max_sim_time is None else EvalConfig(max_sim_time_s=args.max_sim_time)
    policy = replay_files.ReplayPolicy(mode=args.replay, every=args.replay_every, best_k=args.replay_best_k)
    ga_params = ga.GAConfig(
        population_size=preset.population, seed=args.seed, elite_count=args.elite_count, selection=args.selection,
        tournament_size=args.tournament_size, crossover=args.crossover, crossover_rate=args.crossover_rate,
        blx_alpha=args.blx_alpha, binomial_indpb=args.binomial_indpb, mutation_gene_rate=args.mutation_rate,
        mutation_sigma=args.sigma_min, sigma_max=args.sigma_max, adaptive_mutation=args.adaptive_mutation)
    return EvolveConfig(preset=preset, seed=args.seed, workers=args.workers, eval_config=rules, replay=policy,
                        validation_seeds=args.validation_seeds, seed_reference=args.seed_reference, ga_params=ga_params)


def default_out_dir(config: EvolveConfig, runs: int, prefix: str = "") -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{prefix}{config.preset.name}-seed{config.seed}" + (f"-x{runs}" if runs > 1 else "") + f"-{stamp}"
    path, n = Path("runs") / name, 2
    while path.exists():
        path = Path("runs") / f"{name}-{n}"
        n += 1
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summarize:
        rows = [row for path in args.summarize for row in load_summaries(path)]
        print(format_summary(rows) if rows else f"No hay corridas en {', '.join(args.summarize)}")
        return 0 if rows else 1
    if args.random_search:  # a baseline: an option it would not use is a usage error, never silently ignored
        given = [f"--{name.replace('_', '-')}" for name in GA_ONLY_OPTIONS if getattr(args, name) != parser.get_default(name)
                 and not (name == "replay" and args.replay == "none")]  # no replay is what a baseline does anyway
        if given:
            parser.error(f"--random-search no usa {', '.join(given)} (son opciones del AG)")
    try:
        config = config_from_args(args)
    except ValueError as error:  # an invalid GA parameter or preset size: a usage error (exit 2), not a traceback
        parser.error(str(error))
    run = run_evolution
    if args.random_search:
        import baselines  # it builds on this module, so it is imported here, not at the top

        run = baselines.run_random_search
    out = Path(args.out) if args.out else default_out_dir(config, args.runs, "random-" if args.random_search else "")
    try:
        if args.runs == 1:
            run(config, out)
        else:
            run_many(config, args.runs, out, run=run)
    except KeyboardInterrupt:
        return 130
    except FileExistsError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

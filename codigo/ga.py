"""
ga.py: el algoritmo genético, escrito con DEAP.

Para qué existe
    Es el motor de búsqueda del TP. No sabe nada de rovers: recibe una función que puntúa una
    generación entera y devuelve la población siguiente. Así el AG se puede probar sin el simulador
    y el simulador se puede cambiar sin tocar el AG.

Qué contiene
    - GAConfig: los métodos y parámetros del AG (por defecto, los de la Tabla 1 de la entrega parcial).
    - make_toolbox: registra en DEAP la selección, el cruce y la mutación.
    - next_generation: arma la generación siguiente (elitismo, selección, cruce, mutación y reflexión).
    - run: repite ese ciclo durante todas las generaciones, de forma reproducible para cada semilla.
    - RunLogger: guarda generations.csv, best_genomes.json y populations.npz de una corrida.

Cómo funciona
    Selección por torneo (k = 3) o por ruleta; cruce BLX-α (α = 0.3) o binomial; mutación gaussiana
    con un paso σ adaptativo, que crece de 0.1 a 0.3 a medida que la población pierde diversidad.
    Después de cruzar y de mutar, cada gen se refleja dentro de [0, 1] en lugar de recortarse.
    Los 2 mejores individuos pasan sin cambios a la generación siguiente.

Dónde encaja
    Etapa: operadores y ciclo de generaciones. Usa genome_spec para el largo del genoma.
    Lo usan evolve, baselines, fitness y replay. El notebook lo compara con su propio AG para
    comprobar que ambos producen los mismos resultados.
"""

from __future__ import annotations

import csv
import json
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Callable

import deap
import numpy as np
from deap import base, creator, tools

import genome_spec


SELECTION_METHODS = ("tournament", "roulette")
CROSSOVER_METHODS = ("blx", "binomial")
ROULETTE_EPSILON = 1e-6


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 30
    elite_count: int = 2
    selection: str = "tournament"
    tournament_size: int = 3
    crossover: str = "blx"
    crossover_rate: float = 0.9
    blx_alpha: float = 0.3
    binomial_indpb: float = 0.5
    mutation_gene_rate: float = 0.1
    mutation_sigma: float = 0.1
    sigma_max: float = 0.3
    adaptive_mutation: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0 <= self.elite_count <= self.population_size:
            raise ValueError("elite_count must be between 0 and population_size")
        if self.selection not in SELECTION_METHODS:
            raise ValueError(f"selection must be one of {', '.join(SELECTION_METHODS)}, got {self.selection!r}")
        if self.crossover not in CROSSOVER_METHODS:
            raise ValueError(f"crossover must be one of {', '.join(CROSSOVER_METHODS)}, got {self.crossover!r}")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        for name in ("crossover_rate", "mutation_gene_rate", "binomial_indpb"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.blx_alpha < 0:
            raise ValueError("blx_alpha must be non-negative")
        if not self.mutation_sigma > 0.0:
            raise ValueError("mutation_sigma (sigma min) must be positive")
        if self.adaptive_mutation and not self.mutation_sigma <= self.sigma_max:
            raise ValueError("adaptive mutation needs mutation_sigma (sigma min) <= sigma_max")


@dataclass(frozen=True)
class Evaluation:
    fitness: float
    details: dict = field(default_factory=dict)


Evaluator = Callable[[np.ndarray], "list[Evaluation]"]


@dataclass(frozen=True)
class GAState:
    sigma: float
    initial_diversity: float | None = None

    @classmethod
    def initial(cls, config: GAConfig) -> GAState:
        return cls(sigma=config.mutation_sigma)


if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def _sane(fitness) -> np.ndarray:
    f = np.asarray(fitness, dtype=float)
    return np.where(np.isfinite(f), f, -np.inf)


def _fitness_array(evaluations) -> np.ndarray:
    return _sane([e.fitness for e in evaluations])


def to_individuals(population: np.ndarray, fitness=None) -> list:
    individuals = [creator.Individual(float(x) for x in row) for row in np.asarray(population, dtype=float)]
    if fitness is not None:
        for ind, f in zip(individuals, _sane(fitness)):
            ind.fitness.values = (float(f),)
    return individuals


class _Slot:
    __slots__ = ("individual", "fitness")

    def __init__(self, individual, weight: float):
        self.individual = individual
        self.fitness = creator.FitnessMax((weight,))


def select_roulette(individuals: list, k: int) -> list:
    values = np.array([ind.fitness.values[0] for ind in individuals], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return tools.selRandom(individuals, k)
    weights = np.where(finite, values - values[finite].min() + ROULETTE_EPSILON, 0.0)
    wheel = [_Slot(ind, float(w)) for ind, w in zip(individuals, weights)]
    chosen = tools.selRoulette(wheel, k)
    while len(chosen) < k:
        chosen += tools.selRoulette(wheel, k - len(chosen))
    return [slot.individual for slot in chosen]


def make_toolbox(config: GAConfig) -> base.Toolbox:
    toolbox = base.Toolbox()
    if config.selection == "tournament":
        toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    else:
        toolbox.register("select", select_roulette)
    if config.crossover == "blx":
        toolbox.register("mate", tools.cxBlend, alpha=config.blx_alpha)
    else:
        toolbox.register("mate", tools.cxUniform, indpb=config.binomial_indpb)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=config.mutation_sigma, indpb=config.mutation_gene_rate)
    return toolbox


def _deap_version() -> str:
    try:
        return metadata.version("deap")
    except metadata.PackageNotFoundError:
        return deap.__version__


def describe_operators(config: GAConfig) -> dict:
    if config.selection == "tournament":
        select = f"tools.selTournament(tournsize={config.tournament_size})"
    else:
        select = f"tools.selRoulette on the shifted fitness f - min(f) + {ROULETTE_EPSILON:g} (ga.select_roulette)"
    kind = (f"tools.cxBlend(alpha={config.blx_alpha:g}) (BLX-alpha)" if config.crossover == "blx"
            else f"tools.cxUniform(indpb={config.binomial_indpb:g}) (binomial)")
    if config.adaptive_mutation:
        sigma = (f"adaptive: sigma_g = {config.mutation_sigma:g} + ({config.sigma_max:g} - {config.mutation_sigma:g}) * "
                 "clip(1 - D_g / D_0, 0, 1), D = mean per-gene std, D_0 = generation 0")
    else:
        sigma = f"fixed: {config.mutation_sigma:g}"
    return {
        "library": f"DEAP {_deap_version()}",
        "select": select,
        "mate": f"{kind}, a pair is mated with p={config.crossover_rate:g}",
        "mutate": f"tools.mutGaussian(mu=0, sigma=sigma_g, indpb={config.mutation_gene_rate:g})",
        "sigma": sigma,
        "elitism": f"the {config.elite_count} best copied unchanged (re-evaluated with every new generation)",
        "bounds": "genes reflected into [0, 1] after crossover and after mutation",
    }


@contextmanager
def _python_random(seed: int):
    saved = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(saved)


def reflect(x: np.ndarray) -> np.ndarray:
    t = np.mod(np.asarray(x, dtype=float), 2.0)
    return np.where(t > 1.0, 2.0 - t, t)


def _reflect_individual(individual) -> None:
    individual[:] = reflect(np.asarray(individual, dtype=float)).tolist()


def initial_population(config: GAConfig, rng: np.random.Generator, seeds: list[np.ndarray] | None = None) -> np.ndarray:
    pop = rng.uniform(0.0, 1.0, size=(config.population_size, genome_spec.GENOME_LENGTH))
    seeds = list(seeds or [])
    if len(seeds) > config.population_size:
        raise ValueError("more seed genomes than the population can hold")
    for i, seed in enumerate(seeds):
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (genome_spec.GENOME_LENGTH,):
            raise ValueError(f"seed genome must have shape ({genome_spec.GENOME_LENGTH},), got {seed.shape}")
        pop[i] = np.clip(seed, 0.0, 1.0)
    return pop


def vary(parents: list, toolbox: base.Toolbox, config: GAConfig, sigma: float) -> list:
    offspring = [toolbox.clone(p) for p in parents]
    for a, b in zip(offspring[::2], offspring[1::2]):
        if random.random() < config.crossover_rate:
            toolbox.mate(a, b)
            _reflect_individual(a)
            _reflect_individual(b)
    for child in offspring:
        toolbox.mutate(child, sigma=sigma)
        _reflect_individual(child)
    return offspring


def adaptive_sigma(diversity: float, initial_diversity: float, config: GAConfig) -> float:
    if not config.adaptive_mutation:
        return config.mutation_sigma
    convergence = 1.0 if initial_diversity <= 0 else float(np.clip(1.0 - diversity / initial_diversity, 0.0, 1.0))
    return config.mutation_sigma + (config.sigma_max - config.mutation_sigma) * convergence


def update_state(state: GAState, population: np.ndarray, config: GAConfig) -> GAState:
    diversity = gene_std_diversity(population)
    initial = diversity if state.initial_diversity is None else state.initial_diversity
    return GAState(sigma=adaptive_sigma(diversity, initial, config), initial_diversity=initial)


def gene_std_diversity(population: np.ndarray) -> float:
    return float(np.std(population, axis=0).mean())


def mean_pairwise_distance(population: np.ndarray) -> float:
    n = len(population)
    if n < 2:
        return 0.0
    dist = np.linalg.norm(population[:, None, :] - population[None, :, :], axis=-1)
    return float(dist[np.triu_indices(n, k=1)].mean())


def next_generation(population: np.ndarray, evaluations, config: GAConfig, state: GAState,
                    rng: np.random.Generator, toolbox: base.Toolbox | None = None) -> tuple[np.ndarray, GAState]:
    if len(evaluations) != len(population):
        raise ValueError(f"expected {len(population)} evaluations, got {len(evaluations)}")
    toolbox = make_toolbox(config) if toolbox is None else toolbox
    new_state = update_state(state, population, config)
    individuals = to_individuals(population, _fitness_array(evaluations))
    elites = tools.selBest(individuals, config.elite_count)
    with _python_random(int(rng.integers(2**63))):
        parents = toolbox.select(individuals, config.population_size - config.elite_count)
        children = vary(parents, toolbox, config, new_state.sigma)
    return np.array(elites + children, dtype=float).reshape(config.population_size, population.shape[1]), new_state


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    population: np.ndarray
    fitness: np.ndarray
    evaluations: tuple
    best_index: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    diversity: float
    pairwise_distance: float
    sigma: float
    elapsed: float

    @property
    def best_genome(self) -> np.ndarray:
        return self.population[self.best_index]

    @property
    def best_details(self) -> dict:
        return self.evaluations[self.best_index].details


@dataclass(frozen=True)
class RunHistory:
    config: GAConfig
    records: tuple

    @property
    def best_record(self) -> GenerationRecord:
        return max(self.records, key=lambda r: r.best_fitness)

    @property
    def best_fitness(self) -> float:
        return self.best_record.best_fitness

    @property
    def best_genome(self) -> np.ndarray:
        return self.best_record.best_genome


def _make_record(generation, population, evaluations, state, elapsed) -> GenerationRecord:
    raw = np.array([e.fitness for e in evaluations], dtype=float)
    sane = _fitness_array(evaluations)
    finite = raw[np.isfinite(raw)]
    stats = (finite.mean(), np.median(finite), finite.std()) if len(finite) else (np.nan,) * 3
    best_index = int(np.argmax(sane))
    return GenerationRecord(
        generation=generation, population=population.copy(), fitness=raw, evaluations=tuple(evaluations),
        best_index=best_index, best_fitness=float(sane[best_index]), mean_fitness=float(stats[0]),
        median_fitness=float(stats[1]), std_fitness=float(stats[2]), diversity=gene_std_diversity(population),
        pairwise_distance=mean_pairwise_distance(population), sigma=state.sigma, elapsed=elapsed)


make_record = _make_record


def run(evaluate: Evaluator, config: GAConfig, generations: int, logger: RunLogger | None = None,
        seeds: list[np.ndarray] | None = None, toolbox: base.Toolbox | None = None) -> RunHistory:
    if generations < 1:
        raise ValueError(f"generations must be at least 1, got {generations}")
    toolbox = make_toolbox(config) if toolbox is None else toolbox
    rng = np.random.default_rng(config.seed)
    population = initial_population(config, rng, seeds)
    state = GAState.initial(config)
    start = time.perf_counter()
    records = []
    for generation in range(generations):
        evaluations = list(evaluate(population))
        if len(evaluations) != len(population):
            raise ValueError(f"evaluator returned {len(evaluations)} results for {len(population)} genomes")
        if generation < generations - 1:
            next_population, state = next_generation(population, evaluations, config, state, rng, toolbox)
        else:
            next_population, state = population, update_state(state, population, config)
        record = _make_record(generation, population, evaluations, state, time.perf_counter() - start)
        records.append(record)
        if logger is not None:
            logger.log(record)
        population = next_population
    return RunHistory(config=config, records=tuple(records))


CSV_COLUMNS = ["generation", "best", "mean", "median", "std", "diversity", "pairwise_distance", "mutation_sigma", "elapsed"]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    return str(obj)


jsonable = _jsonable


class RunLogger:
    def __init__(self, out_dir: str | Path | None = None):
        self.out_dir = self._claim_directory(Path(out_dir)) if out_dir is not None else self._new_default_directory()
        self._best: list[dict] = []
        self._populations: list[np.ndarray] = []
        self._fitness: list[np.ndarray] = []
        self._sigma: list[float] = []

    @staticmethod
    def _claim_directory(path: Path) -> Path:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"{path} already exists and is not an empty directory; a run never overwrites another")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _new_default_directory() -> Path:
        base = Path("runs")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base.mkdir(parents=True, exist_ok=True)
        for n in range(1, 10_000):
            path = base / (stamp if n == 1 else f"{stamp}-{n}")
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue
        raise FileExistsError(f"no free run directory under {base} for {stamp}")

    def log(self, record: GenerationRecord) -> None:
        if record.generation != len(self._best):
            raise ValueError(
                f"a RunLogger serves a single run: expected generation {len(self._best)}, got {record.generation} "
                "(make a new logger for a new run)")
        first = not self._best
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "generations.csv", "w" if first else "a", newline="") as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(CSV_COLUMNS)
            writer.writerow([record.generation, record.best_fitness, record.mean_fitness, record.median_fitness,
                             record.std_fitness, record.diversity, record.pairwise_distance, record.sigma,
                             round(record.elapsed, 6)])
        genome = record.best_genome
        self._best.append({
            "generation": record.generation, "fitness": record.best_fitness, "genome": genome.tolist(),
            "decoded": genome_spec.decode(genome), "details": record.best_details})
        self._populations.append(record.population)
        self._fitness.append(record.fitness)
        self._sigma.append(record.sigma)
        (self.out_dir / "best_genomes.json").write_text(json.dumps(_jsonable(self._best), indent=2), encoding="utf-8")
        np.savez(self.out_dir / "populations.npz", populations=np.stack(self._populations),
                 fitness=np.stack(self._fitness), sigma=np.array(self._sigma))

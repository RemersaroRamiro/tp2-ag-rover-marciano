"""A MuJoCo-free genetic algorithm for real-valued genomes in [0, 1] (`genome_spec.GENOME_LENGTH` genes), on DEAP.

The GA knows nothing about rovers: fitness comes from a pluggable batch evaluator, `evaluate(population) ->
list[Evaluation]`, so the whole generation is simulated at once (one batch for the worker pool, see `evolve.py`).

DEAP (`deap.base`, `creator`, `tools`), the library of the course's demos, provides the individuals and the operators.
Every operator is registered on a `base.Toolbox` (`make_toolbox`), so switching a method is one `toolbox.register(...)`
line, as in the course's demo notebook:

- select: `tools.selTournament(tournsize=k)` (default) or roulette, `tools.selRoulette` on a shifted fitness
  (`select_roulette`, because Deb's fitness is negative for illegal designs). Tournament pressure is one integer and
  does not depend on the fitness scale, which matters because rover fitness has no natural normalisation.
- mate: `tools.cxBlend(alpha=a)` = BLX-alpha (default; each child gene uniform in the parents' interval widened by
  `alpha * |difference|` on each side; its exploration shrinks by itself as the population converges) or
  `tools.cxUniform(indpb=p)` = binomial crossover (each gene swapped between the two children with probability p).
  A selected pair is mated with probability `crossover_rate`.
- mutate: `tools.mutGaussian(mu=0, sigma=s, indpb=mutation_gene_rate)`: each gene changes with a constant probability by
  N(0, s), s in [0, 1] units (a fraction of each gene's physical range); s is passed per generation (adaptive).

Explicit custom code, not DEAP's:

- Reflection: after crossover and after mutation every gene is reflected back into [0, 1] (`reflect`), not clipped, so
  genes do not pile up on 0 or 1.
- Elitism: the best `elite_count` individuals are copied unchanged into the next generation. They are re-evaluated with
  it, so an evaluator whose environment changes every generation (a new terrain seed) judges them afresh.
- Adaptive mutation by convergence (`adaptive_sigma`): s_g = sigma_min + (sigma_max - sigma_min) * clip(1 - D_g / D_0,
  0, 1), with D = `gene_std_diversity` (mean per-gene std) and D_0 the diversity of generation 0. A diverse population
  explores with small steps; as it converges, the steps grow. With `adaptive_mutation=False`, s = sigma_min always.
- Deb's feasibility fitness (`fitness.py`) is the scalar the GA maximises (`FitnessMax`, `weights=(1.0,)`).
- Randomness: DEAP draws from Python's global `random` module. Each generation's operators run on a private stream
  seeded from the run's numpy generator (`_python_random`), so a run is deterministic per seed and the caller's
  `random` state is left as it was.
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


# --- configuration and types -----------------------------------------------------------------------------------

SELECTION_METHODS = ("tournament", "roulette")
CROSSOVER_METHODS = ("blx", "binomial")
ROULETTE_EPSILON = 1e-6  # slice of the wheel left to the worst individual (see `select_roulette`)


@dataclass(frozen=True)
class GAConfig:
    """Every GA method and parameter. The defaults are Tabla 1 of the partial delivery (N=30, tournament k=3, BLX-alpha
    0.3 with p=0.9, gene rate 0.1, sigma 0.1-0.3 adaptive, elitism 2); generations and terrains belong to the run."""

    population_size: int = 30
    elite_count: int = 2
    selection: str = "tournament"  # "tournament" (tools.selTournament) | "roulette" (tools.selRoulette, shifted)
    tournament_size: int = 3
    crossover: str = "blx"  # "blx" (tools.cxBlend) | "binomial" (tools.cxUniform)
    crossover_rate: float = 0.9  # probability that a selected pair is mated
    blx_alpha: float = 0.3
    binomial_indpb: float = 0.5  # per-gene swap probability of the binomial crossover
    mutation_gene_rate: float = 0.1  # per-gene mutation probability (constant)
    mutation_sigma: float = 0.1  # sigma_min: the sigma of a diverse population, and the fixed sigma without adaptation
    sigma_max: float = 0.3  # the sigma of a fully converged population
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
    """Result of evaluating one genome. `details` is free-form (distance, total violation, violation reasons, ...)."""

    fitness: float
    details: dict = field(default_factory=dict)


Evaluator = Callable[[np.ndarray], "list[Evaluation]"]


@dataclass(frozen=True)
class GAState:
    """Adaptive-mutation state carried from generation to generation."""

    sigma: float
    initial_diversity: float | None = None  # D_0, fixed by the first generation (None before it)

    @classmethod
    def initial(cls, config: GAConfig) -> GAState:
        return cls(sigma=config.mutation_sigma)


# --- DEAP: fitness, individuals, toolbox -----------------------------------------------------------------------

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # one objective, maximised
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def _sane(fitness) -> np.ndarray:
    """Fitness as floats; NaN/inf (a failed evaluation) becomes -inf so it never beats a finite one."""
    f = np.asarray(fitness, dtype=float)
    return np.where(np.isfinite(f), f, -np.inf)


def _fitness_array(evaluations) -> np.ndarray:
    return _sane([e.fitness for e in evaluations])


def to_individuals(population: np.ndarray, fitness=None) -> list:
    """DEAP individuals (`creator.Individual`, a list of floats) from the rows of `population`. `fitness` (one number
    per row) becomes their fitness values; a non-finite (failed) value becomes -inf."""
    individuals = [creator.Individual(float(x) for x in row) for row in np.asarray(population, dtype=float)]
    if fitness is not None:
        for ind, f in zip(individuals, _sane(fitness)):
            ind.fitness.values = (float(f),)
    return individuals


class _Slot:
    """One individual on the roulette wheel, with its shifted fitness."""

    __slots__ = ("individual", "fitness")

    def __init__(self, individual, weight: float):
        self.individual = individual
        self.fitness = creator.FitnessMax((weight,))


def select_roulette(individuals: list, k: int) -> list:
    """Roulette-wheel selection (`tools.selRoulette`) on a SHIFTED fitness: `k` references to `individuals`.

    `tools.selRoulette` needs positive fitness values, but Deb's feasibility fitness is negative for every illegal
    design (`fitness.py`). So the wheel uses w_i = f_i - min_j f_j + ROULETTE_EPSILON, the minimum taken over the
    finite values: the order is unchanged, the worst individual keeps a tiny slice, and equal fitness gives a uniform
    wheel. A failed evaluation (non-finite fitness) gets no slice; if every evaluation failed the pick is uniform.
    The shift changes the selection probabilities (they depend on the distance to the worst individual), which is
    the usual windowing of roulette selection. The individuals' own fitness is not modified."""
    values = np.array([ind.fitness.values[0] for ind in individuals], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return tools.selRandom(individuals, k)
    weights = np.where(finite, values - values[finite].min() + ROULETTE_EPSILON, 0.0)
    wheel = [_Slot(ind, float(w)) for ind, w in zip(individuals, weights)]
    chosen = tools.selRoulette(wheel, k)
    while len(chosen) < k:  # a float rounding of the running sum can leave a spin unassigned: spin again
        chosen += tools.selRoulette(wheel, k - len(chosen))
    return [slot.individual for slot in chosen]


def make_toolbox(config: GAConfig) -> base.Toolbox:
    """A DEAP toolbox with the configured operators: `select`, `mate`, `mutate` (call `mutate` with the generation's
    `sigma=`). Switching a method is one `register` line, e.g. `toolbox.register("select", tools.selTournament,
    tournsize=5)`; `run` and `next_generation` accept a toolbox changed that way."""
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
    """The installed DEAP release (``deap.__version__`` only says the minor version)."""
    try:
        return metadata.version("deap")
    except metadata.PackageNotFoundError:
        return deap.__version__


def describe_operators(config: GAConfig) -> dict:
    """The GA's methods in words, for a run's manifest (every run documents itself)."""
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
    """Run DEAP's operators (they draw from the global `random` module) on a private stream seeded with `seed`, and
    give the caller's `random` state back afterwards."""
    saved = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(saved)


# --- helpers ---------------------------------------------------------------------------------------------------

def reflect(x: np.ndarray) -> np.ndarray:
    """Fold values back into [0, 1] by reflecting at both borders (a triangle wave)."""
    t = np.mod(np.asarray(x, dtype=float), 2.0)
    return np.where(t > 1.0, 2.0 - t, t)


def _reflect_individual(individual) -> None:
    individual[:] = reflect(np.asarray(individual, dtype=float)).tolist()


# --- population ------------------------------------------------------------------------------------------------

def initial_population(config: GAConfig, rng: np.random.Generator, seeds: list[np.ndarray] | None = None) -> np.ndarray:
    """Uniform random genomes; `seeds` (e.g. a Perseverance-like design) take the first rows, clipped to [0, 1]."""
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


# --- variation -------------------------------------------------------------------------------------------------

def vary(parents: list, toolbox: base.Toolbox, config: GAConfig, sigma: float) -> list:
    """Offspring of the selected `parents` (DEAP individuals): clones mated in consecutive pairs with probability
    `crossover_rate` (an odd last one is not mated), then every clone mutated (`toolbox.mutate`, per-gene probability
    from the toolbox, std `sigma`); genes are reflected into [0, 1] after each step. Draws from Python's `random`, as
    DEAP does; the parents are not modified."""
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


# --- adaptive mutation and diversity ---------------------------------------------------------------------------

def adaptive_sigma(diversity: float, initial_diversity: float, config: GAConfig) -> float:
    """sigma_min + (sigma_max - sigma_min) * clip(1 - D / D_0, 0, 1): sigma_min while the population is as diverse as
    generation 0 (or more), sigma_max once it has collapsed to clones (also when D_0 is already 0). Without adaptive
    mutation, sigma_min."""
    if not config.adaptive_mutation:
        return config.mutation_sigma
    convergence = 1.0 if initial_diversity <= 0 else float(np.clip(1.0 - diversity / initial_diversity, 0.0, 1.0))
    return config.mutation_sigma + (config.sigma_max - config.mutation_sigma) * convergence


def update_state(state: GAState, population: np.ndarray, config: GAConfig) -> GAState:
    """The sigma that breeds the next generation from `population` (the generation just evaluated). The first call
    fixes D_0 = the diversity of that population."""
    diversity = gene_std_diversity(population)
    initial = diversity if state.initial_diversity is None else state.initial_diversity
    return GAState(sigma=adaptive_sigma(diversity, initial, config), initial_diversity=initial)


def gene_std_diversity(population: np.ndarray) -> float:
    """Mean over genes of the population standard deviation (0 for clones)."""
    return float(np.std(population, axis=0).mean())


def mean_pairwise_distance(population: np.ndarray) -> float:
    """Mean Euclidean distance over all distinct genome pairs (0 for fewer than two genomes)."""
    n = len(population)
    if n < 2:
        return 0.0
    dist = np.linalg.norm(population[:, None, :] - population[None, :, :], axis=-1)
    return float(dist[np.triu_indices(n, k=1)].mean())


# --- one generation --------------------------------------------------------------------------------------------

def next_generation(population: np.ndarray, evaluations, config: GAConfig, state: GAState,
                    rng: np.random.Generator, toolbox: base.Toolbox | None = None) -> tuple[np.ndarray, GAState]:
    """Elitism + `toolbox.select` + `vary` (crossover, Gaussian mutation, reflection). Returns (new population, new
    state). `toolbox` defaults to `make_toolbox(config)`; DEAP's draws come from a stream seeded by `rng`."""
    if len(evaluations) != len(population):
        raise ValueError(f"expected {len(population)} evaluations, got {len(evaluations)}")
    toolbox = make_toolbox(config) if toolbox is None else toolbox
    new_state = update_state(state, population, config)
    individuals = to_individuals(population, _fitness_array(evaluations))
    elites = tools.selBest(individuals, config.elite_count)  # copied unchanged; sorted() is stable: first wins ties
    with _python_random(int(rng.integers(2**63))):
        parents = toolbox.select(individuals, config.population_size - config.elite_count)
        children = vary(parents, toolbox, config, new_state.sigma)
    return np.array(elites + children, dtype=float).reshape(config.population_size, population.shape[1]), new_state


# --- the loop --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    population: np.ndarray  # (N, L) genomes that were evaluated
    fitness: np.ndarray  # (N,) raw fitness (NaN/inf kept as reported)
    evaluations: tuple  # the N Evaluation objects
    best_index: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float  # statistics over finite fitness values only
    diversity: float  # mean per-gene std
    pairwise_distance: float
    sigma: float  # mutation sigma used to breed the next generation
    elapsed: float  # seconds since the run started

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
        return max(self.records, key=lambda r: r.best_fitness)  # first record wins ties

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


make_record = _make_record  # public name: a baseline (``baselines.py``) logs its batches with the GA's statistics


def run(evaluate: Evaluator, config: GAConfig, generations: int, logger: RunLogger | None = None,
        seeds: list[np.ndarray] | None = None, toolbox: base.Toolbox | None = None) -> RunHistory:
    """Evolve for `generations` generations (each evaluated exactly once, as one batch); deterministic per seed.

    `toolbox` defaults to `make_toolbox(config)`. Nothing is cached between generations: elites are part of the next
    population and are evaluated again with it, so an evaluator whose environment changes from generation to
    generation (a new terrain seed) judges them afresh."""
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


# --- logging ---------------------------------------------------------------------------------------------------

CSV_COLUMNS = ["generation", "best", "mean", "median", "std", "diversity", "pairwise_distance", "mutation_sigma", "elapsed"]


def _jsonable(obj):
    """Recursively convert numpy/odd values to plain JSON types; non-finite floats become null."""
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


jsonable = _jsonable  # public name for the modules that write run files


class RunLogger:
    """Persists ONE run under `out_dir`; nothing is written outside it.

    The directory is created (or an existing *empty* one accepted) when the logger is built: a non-empty directory
    raises `FileExistsError`, so a run can never overwrite or mix with another one. The default is
    `runs/<timestamp>` (`-2`, `-3`, ... when that name is taken). A logger serves a single run: its records must be
    generations 0, 1, 2, ... in order, and feeding it a second run raises `ValueError`.

    Files, refreshed after every generation so an interrupted run stays analysable:
    - `generations.csv`: one row per generation (see `CSV_COLUMNS`)
    - `best_genomes.json`: per generation the best genome, its decoded physical values and its evaluation details
    - `populations.npz`: `populations` (G, N, L), `fitness` (G, N) and `sigma` (G,) for replay and analysis
    """

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
                path.mkdir()  # atomic: the directory is ours or this raises
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

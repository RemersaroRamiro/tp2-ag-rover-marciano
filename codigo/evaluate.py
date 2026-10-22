"""
evaluate.py: simula muchos rovers en paralelo, sin gráficos, y resume qué hizo cada uno.

Para qué existe
    Es la puerta de entrada al simulador: recibe genomas y devuelve, para cada uno, la distancia
    recorrida, cómo terminó y qué restricciones viola. Usa varios procesos a la vez y el resultado no
    depende de la máquina ni de la cantidad de procesos.

Qué contiene
    - EvalConfig: las reglas de la evaluación (paso de simulación, criterio de rover atascado, tiempo
      máximo); dos resultados solo son comparables si se obtuvieron con la misma configuración.
    - RunSummary: lo que hizo un rover (distancia máxima, estado final, sección alcanzada, energía).
    - evaluate_one: decodifica, construye, simula y resume un genoma en el terreno de una semilla.
    - Evaluator: grupo de procesos que evalúa una generación tras otra; genera cada terreno una sola
      vez y lo comparte con los procesos mediante un archivo.

Cómo funciona
    Cada genoma se convierte en un rover, se revisan sus restricciones (las violaciones se anotan,
    pero el rover se simula igual) y se lo maneja con el controlador y el modelo de energía hasta que
    la corrida termina. Cada simulación depende solo del genoma, la semilla y la configuración.

Dónde encaja
    Etapa: simulación. Llama a rover, headless, rover_run, genome_spec y a los archivos de parámetros.
    Lo usa evolve.py; fitness.py convierte sus RunSummary en aptitud.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import pickle
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

import genome_spec
import headless
import mars_params as mp
import rover
import rover_params as rp
import rover_run

TRAJECTORY_COLUMNS = rover_run.TRAJECTORY_COLUMNS
PREFIX = "r0_"


DEFAULT_TIMESTEP_S = 0.0025
DEFAULT_STUCK_WINDOW_S = rp.STUCK_WINDOW_S
MIN_CRUISE_SPEED_MPS = min(g.low for g in genome_spec.GENES if g.name == "target_speed_mps")
TIME_MARGIN = 1.25
DEFAULT_ROCK_WINDOW_M = 40.0


def lane_length_m() -> float:
    return headless.lane_length_m()


@dataclass(frozen=True)
class EvalConfig:
    timestep_s: float = DEFAULT_TIMESTEP_S
    stuck_window_s: float = DEFAULT_STUCK_WINDOW_S
    stuck_progress_m: float = rp.STUCK_PROGRESS_M
    max_sim_time_s: float | None = None
    min_cruise_speed_mps: float = MIN_CRUISE_SPEED_MPS
    time_margin: float = TIME_MARGIN
    cell_m: float = mp.TRACK_CELL_M
    spawn_x_m: float = rp.SPAWN_X_M
    rock_window_m: float | None = DEFAULT_ROCK_WINDOW_M

    def __post_init__(self):
        ratio = rp.CONTROL_DT_S / self.timestep_s if self.timestep_s > 0 else 0.0
        if not (self.timestep_s > 0 and ratio >= 1.0 - 1e-9 and abs(ratio - round(ratio)) < 1e-6):
            raise ValueError(
                f"timestep_s={self.timestep_s} must divide the control period {rp.CONTROL_DT_S} s into whole steps "
                f"(e.g. 0.002, 0.004, 0.005, 0.01)"
            )
        for name in ("stuck_window_s", "stuck_progress_m", "min_cruise_speed_mps", "cell_m"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.rock_window_m is not None and not self.rock_window_m > 0:
            raise ValueError(f"rock_window_m must be positive or None, got {self.rock_window_m}")
        if not self.time_margin >= 1.0:
            raise ValueError(f"time_margin must be at least 1, got {self.time_margin}")
        if self.max_sim_time_s is None:
            object.__setattr__(self, "max_sim_time_s", self.time_margin * lane_length_m() / self.min_cruise_speed_mps)
        elif not self.max_sim_time_s > 0:
            raise ValueError(f"max_sim_time_s must be positive, got {self.max_sim_time_s}")

    @property
    def steps_per_control(self) -> int:
        return round(rp.CONTROL_DT_S / self.timestep_s)


@dataclass(frozen=True)
class RunSummary:
    seed: int
    status: str
    reason: str | None
    max_distance_m: float
    distance_m: float
    max_x_m: float
    alive_time_s: float
    section_reached: str | None
    drive_energy_wh: float
    net_drain_wh: float
    usable_wh: float
    soc: float
    max_abs_y_m: float
    torque_limit_nm: float
    total_mass_kg: float
    repairs: tuple[str, ...]
    violations: tuple[rover.Violation, ...]
    steps: int
    windows_used: int
    trajectory: np.ndarray = field(compare=False, repr=False)
    wall_s: float = field(default=0.0, compare=False)
    build_s: float = field(default=0.0, compare=False)
    sim_s: float = field(default=0.0, compare=False)


SHARED_TRACKS_KEPT = 2
_ANY_POPULATION = 1 << 30


@lru_cache(maxsize=SHARED_TRACKS_KEPT)
def _world(seed: int, cell_m: float):
    return headless.generate_headless_world(seed, cell_m=cell_m)


def save_shared_world(world, directory: str | Path, name: str | None = None) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or f"world_{world.seed}.pkl")
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as f:
        pickle.dump(world, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(partial, path)
    return path


def load_shared_world(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


_SHARED: OrderedDict[str, object] = OrderedDict()


def _shared_world(path: str):
    if path not in _SHARED:
        _SHARED[path] = load_shared_world(path)
        while len(_SHARED) > SHARED_TRACKS_KEPT:
            _SHARED.popitem(last=False)
    _SHARED.move_to_end(path)
    return _SHARED[path]


def _check_genome(genome) -> np.ndarray:
    g = np.asarray(genome, dtype=float)
    if g.shape != (genome_spec.GENOME_LENGTH,):
        raise ValueError(f"genome must have shape ({genome_spec.GENOME_LENGTH},), got {g.shape}")
    return g


def _is_unstable(data, steps: int, timestep_s: float) -> bool:
    return data.time < (steps - 0.5) * timestep_s or not np.isfinite(data.qpos).all()


def evaluate_one(genome, seed: int, config: EvalConfig | None = None, world=None) -> RunSummary:
    config = config or EvalConfig()
    genome = _check_genome(genome)
    t_start = time.perf_counter()
    design = rover.design_from_genome(genome)
    violations = tuple(rover.check_constraints(design))
    world = world if world is not None else _world(seed, config.cell_m)
    pos, yaw = rover.spawn_pose(world, design, x=config.spawn_x_m)
    windows = headless.RockWindows(config.rock_window_m) if config.rock_window_m else None
    zone = windows.index(pos[0]) if windows else 0
    model = headless.build_model(world, design, PREFIX, pos, yaw, config.timestep_s, windows.rock_range(zone) if windows else None)
    data = mujoco.MjData(model)
    limits = rover_run.RunLimits.on_track(
        timeout_s=config.max_sim_time_s, stuck_window_s=config.stuck_window_s, stuck_progress_m=config.stuck_progress_m
    )
    run = rover_run.RoverRun(model, data, design, PREFIX, world=world, limits=limits)
    t_built = time.perf_counter()

    step, update = mujoco.mj_step, run.update
    steps, unstable, models_built = 0, False, 1
    chassis, spc = run.controller.chassis_body, config.steps_per_control
    check_every = round(1.0 / config.timestep_s)
    max_steps = math.ceil(config.max_sim_time_s / config.timestep_s) + 2 * config.steps_per_control
    while run.running and steps < max_steps:
        step(model, data)
        update(data)
        steps += 1
        if windows is not None and steps % spc == 0 and windows.needs_switch(zone, data.xpos[chassis][0]):
            zone = windows.index(data.xpos[chassis][0])
            new_model = headless.build_model(world, design, PREFIX, pos, yaw, config.timestep_s, windows.rock_range(zone))
            data = headless.transfer_state(model, data, new_model)
            model = new_model
            run.controller.limit_torque(model)
            models_built += 1
        if steps % check_every == 0 and _is_unstable(data, steps, config.timestep_s):
            unstable = True
            break
    t_done = time.perf_counter()

    summary = run.summary()
    status, reason = summary["status"], summary["reason"]
    if unstable:
        status, reason = "dead", "unstable"
    return RunSummary(
        seed=seed, status=status, reason=reason,
        max_distance_m=summary["max_distance_m"], distance_m=summary["distance_m"], max_x_m=summary["max_x_m"],
        alive_time_s=summary["alive_time_s"], section_reached=summary["section_reached"],
        drive_energy_wh=summary["drive_energy_wh"], net_drain_wh=summary["net_drain_wh"], usable_wh=summary["usable_wh"],
        soc=summary["soc"], max_abs_y_m=summary["max_abs_y_m"], torque_limit_nm=summary["torque_limit_nm"],
        total_mass_kg=design.total_mass_kg, repairs=design.repairs, violations=violations, steps=steps, windows_used=models_built,
        trajectory=run.trajectory_array(),
        wall_s=t_done - t_start, build_s=t_built - t_start, sim_s=t_done - t_built,
    )


def resolve_workers(workers: int | None, population: int) -> int:
    if workers is None:
        workers = os.cpu_count() or 1
    elif workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    return max(1, min(workers, population))


def _check_population(genomes) -> np.ndarray:
    g = np.asarray(genomes, dtype=float)
    if g.ndim != 2 or g.shape[1] != genome_spec.GENOME_LENGTH:
        raise ValueError(f"genomes must have shape (n, {genome_spec.GENOME_LENGTH}), got {g.shape}")
    return g


def _evaluate_task(task: tuple[tuple[float, ...], int, EvalConfig, str | None]) -> RunSummary:
    genome, seed, config, world_path = task
    world = _shared_world(world_path) if world_path is not None else None
    return evaluate_one(np.array(genome), seed, config, world)


class Evaluator:
    def __init__(self, workers: int | None = None, warm: tuple[int, EvalConfig] | None = None):
        self._requested = workers
        self._warm = warm
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._builder: ThreadPoolExecutor | None = None
        self._tracks: OrderedDict[tuple, Future] = OrderedDict()
        self._room = SHARED_TRACKS_KEPT
        self._dir: Path | None = None

    def _ensure_pool(self, population: int) -> ProcessPoolExecutor | None:
        if resolve_workers(self._requested, population) == 1:
            return None
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._requested if self._requested is not None else (os.cpu_count() or 1),
                mp_context=multiprocessing.get_context("spawn"),
            )
            if self._warm is not None:
                self._track(*self._warm)
        return self._pool

    def _publish(self, seed: int, config: EvalConfig) -> str:
        world = _world(seed, config.cell_m)
        with self._lock:
            if self._dir is None:
                self._dir = Path(tempfile.mkdtemp(prefix="rover3d-tracks-"))
            directory = self._dir
        return str(save_shared_world(world, directory, f"world_{seed}_{config.cell_m:g}.pkl"))

    def _track(self, seed: int, config: EvalConfig) -> Future | None:
        key = (seed, config.cell_m)
        with self._lock:
            if key in self._tracks:
                self._tracks.move_to_end(key)
                return self._tracks[key]
            if self._builder is None:
                self._builder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="track-builder")
            future = self._builder.submit(self._publish, seed, config)
            self._tracks[key] = future
            self._evict_locked()
            return future

    def _evict_locked(self) -> None:
        while len(self._tracks) > max(self._room, 1):
            _, old = self._tracks.popitem(last=False)
            if old.done() and old.exception() is None:
                Path(old.result()).unlink(missing_ok=True)

    def prefetch(self, seed: int, config: EvalConfig | None = None, keep: int | None = None) -> None:
        config = config or EvalConfig()
        if resolve_workers(self._requested, _ANY_POPULATION) > 1:
            with self._lock:
                if keep is not None:
                    self._room = max(self._room, keep)
            self._track(seed, config)

    def evaluate(self, genomes, seed: int, config: EvalConfig | None = None) -> list[RunSummary]:
        config = config or EvalConfig()
        genomes = _check_population(genomes)
        if len(genomes) == 0:
            return []
        pool = self._ensure_pool(len(genomes))
        if pool is None:
            return [evaluate_one(g, seed, config) for g in genomes]
        track = self._track(seed, config)
        path = track.result() if track is not None else None
        tasks = [(tuple(float(x) for x in g), seed, config, path) for g in genomes]
        return list(pool.map(_evaluate_task, tasks, chunksize=1))

    def evaluate_seeds(self, genomes, seeds, config: EvalConfig | None = None) -> list[list[RunSummary]]:
        config = config or EvalConfig()
        genomes, seeds = _check_population(genomes), list(seeds)
        if not seeds or len(genomes) == 0:
            return [[] for _ in seeds]
        pool = self._ensure_pool(len(genomes) * len(seeds))
        if pool is None:
            return [[evaluate_one(g, seed, config) for g in genomes] for seed in seeds]
        with self._lock:
            previous = self._room
            self._room = max(previous, len(seeds))
        try:
            tracks = [self._track(seed, config) for seed in seeds]
            paths = [t.result() if t is not None else None for t in tracks]
            tasks = [(tuple(float(x) for x in g), seed, config, path) for seed, path in zip(seeds, paths) for g in genomes]
            flat = list(pool.map(_evaluate_task, tasks, chunksize=1))
        finally:
            with self._lock:
                self._room = previous
                self._evict_locked()
        n = len(genomes)
        return [flat[i * n:(i + 1) * n] for i in range(len(seeds))]

    def release_room(self) -> None:
        with self._lock:
            self._room = SHARED_TRACKS_KEPT
            self._evict_locked()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self._builder is not None:
            self._builder.shutdown(wait=True)
            self._builder = None
        self._tracks.clear()
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def evaluate_population(genomes, seed: int, config: EvalConfig | None = None, workers: int | None = None) -> list[RunSummary]:
    config = config or EvalConfig()
    genomes = _check_population(genomes)
    with Evaluator(workers, warm=(seed, config)) as evaluator:
        return evaluator.evaluate(genomes, seed, config)

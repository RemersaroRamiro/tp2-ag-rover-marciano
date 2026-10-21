"""Headless evaluation of rover genomes on the Jezero track: one run per genome, in parallel, reproducibly.

``evaluate_population(genomes, seed, config, workers)`` turns every genome into a rover (``rover.design_from_genome``),
checks the mission constraints (violations are *recorded*; the rover is simulated anyway, ranking them is the
fitness function's job), drives it with the T16 controller / energy model / run tracker on the headless model
(``headless``), and returns one ``RunSummary`` per genome, in input order.

The evaluation rules are the fields of ``EvalConfig``; they are what the report has to state. Nothing else is
configurable, and nothing in a result depends on the machine, the worker count or the other genomes of the batch:
a run is a pure function of ``(genome, seed, config)`` (its own model and ``MjData``, no shared RNG), the track of
a seed is deterministic, and the pool returns results by position. ``RunSummary.same_result`` compares two
results bit for bit (wall-clock fields excluded).

Parallelism: a ``spawn`` process pool (same behaviour on Windows, macOS and Linux), ``os.cpu_count()`` workers by
default (started on demand, so never more than genomes). Each worker builds the track of a seed once and keeps it
(``_world``); results are handed back in genome order. No GPU, no graphics.

Shared tracks: generating the track of a seed takes ~1 s of CPU (terrain, rocks, collision tiles). A pool whose
workers each did that for every new seed (the evolution loop uses a new seed every generation) would spend the first
seconds of every generation rebuilding the same track N times, all at once. ``Evaluator`` builds it once, in the
parent, writes it to a file in its own temporary directory, and the workers read that file (milliseconds) instead.
The track a worker reads is bit-for-bit the one it would have generated, so nothing in a result changes.

Replay: ``evaluate_one(..., replay_hz=10.0)`` also records the pose and every joint of the rover (``replay.COLUMNS``)
at that rate, for the replay viewer. Recording reads the state; it changes nothing the rover does.
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
import replay as replay_format
import rover_run
import scene

TRAJECTORY_COLUMNS = rover_run.TRAJECTORY_COLUMNS
PREFIX = "r0_"

# -- the evaluation rules ---------------------------------------------------------------------------------------

DEFAULT_TIMESTEP_S = 0.0025  # physics step; must divide the 20 ms control period (see ``EvalConfig``). User decision
# (T17a): 2.5 ms differs from 2 ms no more than a 1 um spawn shift does (43/44 statuses, 40/44 distances in
# tools/validate_eval.py), i.e. within the chaos of wheels over rocks, at ~25 % less CPU; 4 ms lost finishers.
DEFAULT_STUCK_WINDOW_S = rp.STUCK_WINDOW_S  # no progress for this long ends a run as dead(stuck)
# The slowest an evolved rover can drive is the lower bound of the ``target_speed_mps`` gene.
MIN_CRUISE_SPEED_MPS = min(g.low for g in genome_spec.GENES if g.name == "target_speed_mps")
TIME_MARGIN = 1.25  # the time cap allows this many times the time the slowest rover needs for the whole lane
DEFAULT_ROCK_WINDOW_M = 40.0  # length of a rock window (see ``headless.RockWindows``)


def lane_length_m() -> float:
    """Start line to finish line along +x, in meters (373.45 m)."""
    return headless.lane_length_m()


@dataclass(frozen=True)
class EvalConfig:
    """Everything that defines how a rover is judged. Two results are comparable only for equal configs.

    * ``timestep_s``: physics step. The controller runs every ``rover_params.CONTROL_DT_S`` = 20 ms, so the step
      must divide it exactly (2, 2.5, 4, 5, 10 ms ...); a step that does not would silently change the control rate.
    * ``stuck_window_s`` / ``stuck_progress_m``: a rover whose furthest x grew by less than ``stuck_progress_m``
      during the last ``stuck_window_s`` seconds is dead(stuck) and its run ends (early termination).
    * ``max_sim_time_s``: simulated-time cap, the normal end of a rover that keeps driving (dead(timeout)). By default
      ``time_margin`` times the time the *slowest possible* rover (``min_cruise_speed_mps``, the lower bound of the
      speed gene) needs to drive the whole lane, so the cap never cuts off a rover that is making its designed
      progress. An explicit value wins. (``dataclasses.replace`` keeps a derived value: pass ``max_sim_time_s=None``
      to derive it again.)
    * ``cell_m``: terrain heightfield cell (the track's real resolution by default; coarser only for tests).
    * ``spawn_x_m``: where the rover appears (inside the first section).
    * ``rock_window_m``: performance only. The headless model holds just the rocks near the rover, in zones this
      long (``headless.RockWindows``); the model is swapped, state unchanged, as the rover moves on. It changes
      nothing the rover can feel: ``tools/validate_eval.py`` found 44 of 44 runs identical to a single model with
      every rock (finish times included). ``None`` = one model with every rock of the lane (about ten times more geoms to
      update on every step).
    * ``headless``: ``True`` = the stripped physics model (``headless``); ``False`` = the full scene model of
      ``scene.py`` (slow, for validating the stripped model).
    """

    timestep_s: float = DEFAULT_TIMESTEP_S
    stuck_window_s: float = DEFAULT_STUCK_WINDOW_S
    stuck_progress_m: float = rp.STUCK_PROGRESS_M
    max_sim_time_s: float | None = None
    min_cruise_speed_mps: float = MIN_CRUISE_SPEED_MPS
    time_margin: float = TIME_MARGIN
    cell_m: float = mp.TRACK_CELL_M
    spawn_x_m: float = rp.SPAWN_X_M
    rock_window_m: float | None = DEFAULT_ROCK_WINDOW_M
    headless: bool = True

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


# -- the result of one run --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """What one rover did. Everything but the ``compare=False`` fields is a deterministic function of the inputs."""

    seed: int
    status: str  # "finished" | "dead"
    reason: str | None  # dead: "energy" | "out_of_lane" | "tipped" | "stuck" | "timeout" | "unstable"
    max_distance_m: float  # furthest x reached minus the spawn x: the fitness basis
    distance_m: float  # x at the end of the run minus the spawn x
    max_x_m: float
    alive_time_s: float  # simulated time at which the run ended
    section_reached: str | None
    drive_energy_wh: float
    net_drain_wh: float
    usable_wh: float
    soc: float
    max_abs_y_m: float
    torque_limit_nm: float
    total_mass_kg: float
    repairs: tuple[str, ...]  # genes the decoder had to change to keep the geometry buildable
    violations: tuple[rover.Violation, ...]  # broken mission constraints (empty = a legal rover)
    steps: int  # physics steps simulated
    windows_used: int  # rock-window models built for this run (1 = a single model)
    trajectory: np.ndarray = field(compare=False, repr=False)  # rows of TRAJECTORY_COLUMNS, 1 Hz + the last state
    replay: np.ndarray | None = field(default=None, compare=False, repr=False)  # rows of replay.COLUMNS (float32), if asked for
    wall_s: float = field(default=0.0, compare=False)  # total wall time of this evaluation (not deterministic)
    build_s: float = field(default=0.0, compare=False)  # ... of which decoding + building the model
    sim_s: float = field(default=0.0, compare=False)  # ... of which stepping it

    def same_result(self, other: "RunSummary") -> bool:
        """Bit-for-bit equality of everything deterministic, trajectory and replay included."""
        if (self.replay is None) != (other.replay is None):
            return False
        same_replay = self.replay is None or np.array_equal(self.replay, other.replay)
        return self == other and np.array_equal(self.trajectory, other.trajectory) and same_replay


# -- one evaluation ---------------------------------------------------------------------------------------------


SHARED_TRACKS_KEPT = 2  # tracks kept at a time, in every cache: the generation being evaluated and the next one
_ANY_POPULATION = 1 << 30  # "a population big enough to need the pool": ``resolve_workers`` only asks whether it is > 1 worker


@lru_cache(maxsize=SHARED_TRACKS_KEPT)
def _world(seed: int, cell_m: float, is_headless: bool):
    """The track of ``seed`` for this process, generated once (the last ``SHARED_TRACKS_KEPT`` are kept)."""
    if is_headless:
        return headless.generate_headless_world(seed, cell_m=cell_m)
    return scene.generate_track_world(seed, cell_m=cell_m)


def save_shared_world(world, directory: str | Path, name: str | None = None) -> Path:
    """Write ``world`` (a ``headless.HeadlessWorld``) to ``directory`` for the workers; returns the file. The file
    appears atomically (written under another name, then renamed), so a reader never sees half of it."""
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


_SHARED: OrderedDict[str, object] = OrderedDict()  # per process: the shared tracks it has read (the last SHARED_TRACKS_KEPT)


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
    """A blown-up simulation: non-finite state, or MuJoCo reset the data after a bad acceleration. A reset puts the
    clock back to zero, so the time is compared with the one ``steps`` steps must have reached (not with the last
    check, which would miss a reset within the first checked second)."""
    return data.time < (steps - 0.5) * timestep_s or not np.isfinite(data.qpos).all()


def _replay_interval(replay_hz: float, timestep_s: float) -> int:
    every = round(1.0 / (replay_hz * timestep_s)) if replay_hz > 0 else 0
    if every < 1:
        raise ValueError(f"replay_hz={replay_hz} must be positive and no faster than the physics step ({1.0 / timestep_s:g} Hz)")
    return every


def evaluate_one(genome, seed: int, config: EvalConfig | None = None, replay_hz: float | None = None, world=None) -> RunSummary:
    """Decode, build, simulate and summarise one genome on the track of ``seed``.

    ``replay_hz`` also records the rover's pose and joints at that rate (``RunSummary.replay``). ``world`` is the
    already-generated track of ``seed`` (``headless.generate_headless_world`` for the default headless config), for
    callers that share it; by default it is generated (and cached for this process)."""
    config = config or EvalConfig()
    genome = _check_genome(genome)
    replay_every = None if replay_hz is None else _replay_interval(replay_hz, config.timestep_s)
    t_start = time.perf_counter()
    design = rover.design_from_genome(genome)
    violations = tuple(rover.check_constraints(design))
    world = world if world is not None else _world(seed, config.cell_m, config.headless)
    pos, yaw = rover.spawn_pose(world, design, x=config.spawn_x_m)
    windows = headless.RockWindows(config.rock_window_m) if config.headless and config.rock_window_m else None
    zone = windows.index(pos[0]) if windows else 0
    if config.headless:
        model = headless.build_model(world, design, PREFIX, pos, yaw, config.timestep_s, windows.rock_range(zone) if windows else None)
    else:
        model = rover.build_model_with_rovers(world, [rover.RoverPlacement(design, PREFIX, pos, yaw)])
        model.opt.timestep = config.timestep_s
    data = mujoco.MjData(model)
    limits = rover_run.RunLimits.on_track(
        timeout_s=config.max_sim_time_s, stuck_window_s=config.stuck_window_s, stuck_progress_m=config.stuck_progress_m
    )
    run = rover_run.RoverRun(model, data, design, PREFIX, world=world, limits=limits)
    t_built = time.perf_counter()

    step, update = mujoco.mj_step, run.update
    steps, unstable, models_built = 0, False, 1
    chassis, spc = run.controller.chassis_body, config.steps_per_control
    check_every = round(1.0 / config.timestep_s)  # sanity check once per simulated second
    frames: list[np.ndarray] = []
    last_recorded_step = -1
    if replay_every is not None:
        qpos_index = replay_format.qpos_addresses(model, PREFIX)  # the same in every rock-window model of this rover
        n_free = replay_format.N_FREE

        def record() -> None:
            nonlocal last_recorded_step
            last_recorded_step = steps
            q = data.qpos[qpos_index]  # ``data`` is the current one: a rock-window swap rebinds it
            if np.isfinite(q).all():
                frames.append(np.concatenate(([data.time], q[:n_free], replay_format.wrap_angles(q[n_free:]))).astype(np.float32))

        record()
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
        if replay_every is not None and steps % replay_every == 0:
            record()
        if steps % check_every == 0 and _is_unstable(data, steps, config.timestep_s):
            unstable = True
            break
    if replay_every is not None and last_recorded_step != steps:
        record()  # the last state, whatever the rate (never twice: a step is recorded once, however float32 rounds its time)
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
        replay=np.stack(frames) if frames else None,
        wall_s=t_done - t_start, build_s=t_built - t_start, sim_s=t_done - t_built,
    )


# -- a population, in parallel --------------------------------------------------------------------------------


def resolve_workers(workers: int | None, population: int) -> int:
    """Worker processes to use: ``workers`` or, by default, the machine's CPU count; never more than the population."""
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


def _evaluate_task(task: tuple[tuple[float, ...], int, EvalConfig, float | None, str | None]) -> RunSummary:
    genome, seed, config, replay_hz, world_path = task
    world = _shared_world(world_path) if world_path is not None else None
    return evaluate_one(np.array(genome), seed, config, replay_hz, world)


class Evaluator:
    """A pool of worker processes that can evaluate several generations.

    The track of each seed is built once, here in the parent, and shared with the workers through a file (see the
    module docstring); ``prefetch`` builds the next generation's track while the pool is still busy with this one.
    Use it as a context manager (closing it stops the pool and removes the shared files). ``workers=1`` evaluates
    in this process, without a pool and without sharing. ``warm=(seed, config)`` shares that track up front."""

    def __init__(self, workers: int | None = None, warm: tuple[int, EvalConfig] | None = None):
        self._requested = workers
        self._warm = warm
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._builder: ThreadPoolExecutor | None = None  # one thread that builds and writes shared tracks
        self._tracks: OrderedDict[tuple, Future] = OrderedDict()  # (seed, cell_m) -> Future of the file path
        self._room = SHARED_TRACKS_KEPT  # how many tracks may stay on disk; ``prefetch`` / ``evaluate_seeds`` widen it for a while
        self._dir: Path | None = None

    @property
    def shared_dir(self) -> Path | None:
        """Directory holding the shared tracks (created on first use; None before, and after ``close``)."""
        return self._dir

    def _ensure_pool(self, population: int) -> ProcessPoolExecutor | None:
        if resolve_workers(self._requested, population) == 1:
            return None
        if self._pool is None:
            # The pool is sized for the machine, not for the first population: Python starts worker processes on demand,
            # so a small first batch does not start idle ones, and a later bigger one can use every core.
            self._pool = ProcessPoolExecutor(
                max_workers=self._requested if self._requested is not None else (os.cpu_count() or 1),
                mp_context=multiprocessing.get_context("spawn"),
            )
            if self._warm is not None:
                self._track(*self._warm)
        return self._pool

    def _publish(self, seed: int, config: EvalConfig) -> str:
        world = _world(seed, config.cell_m, config.headless)
        with self._lock:
            if self._dir is None:
                self._dir = Path(tempfile.mkdtemp(prefix="rover3d-tracks-"))
            directory = self._dir
        return str(save_shared_world(world, directory, f"world_{seed}_{config.cell_m:g}.pkl"))

    def _track(self, seed: int, config: EvalConfig) -> Future | None:
        """The (pending) shared file of a track, building it in the background thread if it is new. At most ``_room``
        tracks stay on disk: the least recently asked for are deleted."""
        if not config.headless:
            return None  # only the headless world is small and simple enough to share; workers build the full one
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
        """Start building the track of ``seed`` in the background so the next ``evaluate`` does not wait for it.
        ``keep`` widens how many tracks may stay on disk (at least ``SHARED_TRACKS_KEPT``) for the tracks that have to
        wait together, e.g. the current one and every validation track: asking for a track never deletes one that is
        still inside that room, and neither ``evaluate`` nor ``evaluate_seeds`` narrows it: ``release_room`` gives it back.
        A no-op without a worker pool (in-process evaluation has nothing to share)."""
        config = config or EvalConfig()
        if resolve_workers(self._requested, _ANY_POPULATION) > 1:
            with self._lock:
                if keep is not None:
                    self._room = max(self._room, keep)
            self._track(seed, config)

    def evaluate(self, genomes, seed: int, config: EvalConfig | None = None, replay_hz: float | None = None) -> list[RunSummary]:
        """One ``RunSummary`` per genome, in the order of ``genomes``; with ``replay_hz`` each carries its replay."""
        config = config or EvalConfig()
        genomes = _check_population(genomes)
        if len(genomes) == 0:
            return []
        if replay_hz is not None:
            _replay_interval(replay_hz, config.timestep_s)  # fail here, not once per worker
        pool = self._ensure_pool(len(genomes))
        if pool is None:
            return [evaluate_one(g, seed, config, replay_hz) for g in genomes]
        track = self._track(seed, config)
        path = track.result() if track is not None else None
        tasks = [(tuple(float(x) for x in g), seed, config, replay_hz, path) for g in genomes]
        return list(pool.map(_evaluate_task, tasks, chunksize=1))

    def evaluate_seeds(self, genomes, seeds, config: EvalConfig | None = None, replay_hz=None) -> list[list[RunSummary]]:
        """``evaluate`` on several tracks at once: ONE batch of ``len(seeds) * len(genomes)`` tasks (seed by seed, the genomes
        in order), so the pool stays busy even with few genomes. It scores a generation on all its terrain seeds and
        validates on the unseen ones. Returns, per seed in order, one ``RunSummary`` per genome: the same values
        ``evaluate`` gives seed by seed, whatever the number of workers.

        ``replay_hz``: one rate for every seed, or one per seed (``None`` = no replay on that seed), e.g. ``[10.0, None]``
        to record the first seed only.

        Every track of ``seeds`` must be on disk together, so the room is widened to ``len(seeds)`` for the call and put
        back to what it was before, whether the call succeeds or fails. It never narrows a room the caller widened with
        ``prefetch(keep=)`` (that room holds tracks the caller still needs): ``release_room`` gives it back."""
        config = config or EvalConfig()
        genomes, seeds = _check_population(genomes), list(seeds)
        rates = list(replay_hz) if isinstance(replay_hz, (list, tuple)) else [replay_hz] * len(seeds)
        if len(rates) != len(seeds):
            raise ValueError(f"replay_hz needs one rate per seed ({len(seeds)} seeds), got {len(rates)}")
        for rate in rates:
            if rate is not None:
                _replay_interval(rate, config.timestep_s)  # fail here, not once per worker
        if not seeds or len(genomes) == 0:
            return [[] for _ in seeds]
        pool = self._ensure_pool(len(genomes) * len(seeds))
        if pool is None:
            return [[evaluate_one(g, seed, config, rate) for g in genomes] for seed, rate in zip(seeds, rates)]
        with self._lock:
            previous = self._room
            self._room = max(previous, len(seeds))
        try:
            tracks = [self._track(seed, config) for seed in seeds]
            paths = [t.result() if t is not None else None for t in tracks]
            tasks = [(tuple(float(x) for x in g), seed, config, rate, path) for seed, rate, path in zip(seeds, rates, paths) for g in genomes]
            flat = list(pool.map(_evaluate_task, tasks, chunksize=1))
        finally:
            with self._lock:  # the results are in (or the call failed): put the room back (the workers already hold what they read)
                self._room = previous
                self._evict_locked()
        n = len(genomes)
        return [flat[i * n:(i + 1) * n] for i in range(len(seeds))]

    def release_room(self) -> None:
        """Give back the room that ``prefetch(keep=)`` widened: only the last ``SHARED_TRACKS_KEPT`` tracks stay on disk."""
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


def evaluate_population(genomes, seed: int, config: EvalConfig | None = None, workers: int | None = None,
                        replay_hz: float | None = None) -> list[RunSummary]:
    """Evaluate every genome (rows of ``genomes``) on the track of ``seed``; results are in row order and
    identical for any ``workers`` (None = one per CPU core, at most one per genome)."""
    config = config or EvalConfig()
    genomes = _check_population(genomes)
    with Evaluator(workers, warm=(seed, config)) as evaluator:
        return evaluator.evaluate(genomes, seed, config, replay_hz)

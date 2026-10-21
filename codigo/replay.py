"""Replay data of an evolution run: what is recorded per rover, how it is stored, and how to read it back.

One rover's replay is a float32 array with a row per frame and the ``COLUMNS`` below: the simulated time, the chassis
free joint (position ``x y z`` and unit quaternion ``qw qx qy qz``, world frame) and every other joint of the rover (the
two rockers, the two bogies, the six wheel spin angles, in radians). That is enough to redraw the WHOLE rover later:
set ``data.qpos[qpos_addresses(model, prefix)] = frame[1:]`` on a model built from the rover's design, then
``mj_forward`` (the wheel angles are wrapped to [-pi, pi), which a rotating wheel cannot tell apart from the unwrapped
angle). ``rocker_r`` and ``rocker_l`` are always opposite (the differential bar); both are kept so a viewer need not know.

The recording itself happens inside ``evaluate.evaluate_one`` (``replay_hz``); this module owns the layout and the
files of one generation, ``<run dir>/replay/gen_0003.npz`` + ``gen_0003.json``:

* ``.npz`` (``np.savez_compressed``): ``frames_q`` (F, 18) integers, the frames of every recorded rover one after the
  other, and ``offsets`` (N+1,) so that rover ``i`` owns rows ``offsets[i]:offsets[i+1]``; ``frame_step`` (N,) is
  the thinning factor of rover ``i`` (1 = the full ``hz`` rate); ``recorded`` (N,) bool; ``genomes`` (N, L) and
  ``fitness`` (N,) in population order; ``terrain_seed``, ``terrain_seeds``, ``generation``, ``hz``, ``columns`` and ``scales``.
  ``load_generation`` decodes ``frames_q`` back to float32 ``frames`` (see "Storage" below), so readers never see it.
* ``.json``: per rover its index, rank (0 = best), genome, decoded gene values, decoded design (geometry, masses,
  derived metrics: enough to build the rover again), ``RunSummary`` fields (no arrays) and fitness with its details.

Several terrain seeds per generation. A generation is scored on ``N`` terrains (``terrain_seeds``, the first one is
``terrain_seed``) but only the FIRST is recorded as frames: the frames are what the replay viewer draws, and they
are the bulk of the storage, so the file of a generation stays the size it had with one seed (a generation's rovers
are replayed on the first terrain). The other seeds are kept as summaries only, in each rover's ``other_seeds``
(``seed`` and the ``RunSummary`` fields without arrays: distance, status, section, time, energy ...), and the score
(``fitness`` / ``rank``) is the one over all seeds, so a viewer can say how far each rover got on each terrain. Frames
on the other seeds would multiply the size by ``N``; ``evaluate.Evaluator.evaluate_seeds`` can record them
(``replay_hz``) if a later task needs them.

``ReplayPolicy`` decides what is recorded; ``load_generation`` reads a generation back for the replay viewer.

Storage. Float32 frames barely compress (1.06x measured: the low mantissa bits are noise), and 50 rovers x 6000 frames
(``REPLAY_MAX_FRAMES``, the most a rover keeps) would be 21.6 MB per generation. So each column is rounded to a fixed step (``SCALES``: 1 ms, 1 mm, 0.1 mrad of
quaternion, 1 mrad of joint angle: far below anything visible) and stored as the difference from the previous frame
of the same rover, which are small integers that compress 4x better (int16 when they fit, else int32). The error of
a decoded value is at most half a step. Decoding is a cumulative sum per rover.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import evolution_params as params
import ga
import genome_spec
import rover

JOINTS = ("rocker_l", "rocker_r", "bogie_l", "bogie_r", *(f"wheel_{w}" for w in rover.WHEEL_IDS))
COLUMNS = ("t", "x", "y", "z", "qw", "qx", "qy", "qz", *JOINTS)
N_FREE = 7  # qpos entries of a free joint: position + quaternion
SCALES = np.array([1e-3, 1e-3, 1e-3, 1e-3, *([1e-4] * 4), *([1e-3] * len(JOINTS))])  # rounding step of every column


def qpos_addresses(model, prefix: str) -> np.ndarray:
    """``model.qpos`` indices of a rover's free joint (7 consecutive entries) followed by ``JOINTS``, in ``COLUMNS[1:]`` order."""
    import mujoco

    def address(name: str) -> int:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}")
        if joint < 0:
            raise ValueError(f"no joint {prefix}{name} in the model")
        return int(model.jnt_qposadr[joint])

    free = address("free")
    return np.array([*range(free, free + N_FREE), *(address(j) for j in JOINTS)], dtype=int)


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Angles folded into [-pi, pi)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


# -- what to record ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayPolicy:
    """What the evolution loop keeps for replay.

    * ``mode``: ``"all"`` (every rover), ``"best"`` (only the ``best_k`` fittest of each generation) or ``"none"``.
    * ``every``: only generations 0, ``every``, 2 * ``every`` ... and the last one keep their frames; the other
      generations still write their (small) descriptions.
    * ``hz`` / ``max_frames``: frames per simulated second, and the most frames one rover may keep (longer runs are
      thinned by a whole factor, see ``thin``): this bounds the size of a generation.
    """

    mode: str = "all"
    every: int = 1
    best_k: int = params.REPLAY_BEST_K
    hz: float = params.REPLAY_HZ
    max_frames: int = params.REPLAY_MAX_FRAMES

    def __post_init__(self):
        if self.mode not in ("all", "best", "none"):
            raise ValueError(f"mode must be 'all', 'best' or 'none', got {self.mode!r}")
        if self.every < 1 or self.best_k < 1:
            raise ValueError("every and best_k must be at least 1")
        if not self.hz > 0:
            raise ValueError(f"hz must be positive, got {self.hz}")
        if self.max_frames < 2:
            raise ValueError("max_frames must be at least 2")

    @property
    def wants_frames(self) -> bool:
        return self.mode != "none"

    def records_generation(self, generation: int, generations: int) -> bool:
        return self.wants_frames and (generation % self.every == 0 or generation == generations - 1)

    def selected(self, fitness: np.ndarray) -> np.ndarray:
        """Indices of the rovers whose frames are kept, fittest first (a failed evaluation counts as the worst)."""
        if self.mode == "none":
            return np.zeros(0, dtype=int)
        order = _ranking(fitness)
        return order if self.mode == "all" else order[: self.best_k]


def _ranking(fitness: np.ndarray) -> np.ndarray:
    """Indices from the fittest to the least fit; ties keep population order, failed evaluations come last."""
    return np.argsort(-np.where(np.isfinite(fitness), fitness, -np.inf), kind="stable")


def thin(frames: np.ndarray, max_frames: int) -> tuple[np.ndarray, int]:
    """``frames`` keeping every ``step``-th row, the smallest whole ``step`` that leaves at most ``max_frames`` rows;
    the last row (the final state) is always kept. Returns ``(rows, step)``."""
    step = max(1, -(-len(frames) // max_frames))
    if step == 1:
        return frames, 1
    index = np.arange(0, len(frames), step)
    if index[-1] != len(frames) - 1:
        if len(index) >= max_frames:
            index[-1] = len(frames) - 1
        else:
            index = np.append(index, len(frames) - 1)
    return frames[index], step


# -- writing and reading one generation ----------------------------------------------------------------------------


def encode_frames(frames: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Quantised, delta-coded form of ``frames`` (rows of ``COLUMNS``; rover ``i`` = rows ``offsets[i]:offsets[i+1]``):
    the first row of each rover is absolute, the others are differences from the row before."""
    q = np.round(frames.astype(np.float64) / SCALES).astype(np.int64)
    out = np.empty_like(q)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi > lo:
            out[lo] = q[lo]
            out[lo + 1: hi] = np.diff(q[lo:hi], axis=0)
    small = np.abs(out).max(initial=0) <= np.iinfo(np.int16).max
    return out.astype(np.int16 if small else np.int32)


def decode_frames(coded: np.ndarray, offsets: np.ndarray, scales: np.ndarray = SCALES) -> np.ndarray:
    """Inverse of ``encode_frames``: float32 frames, each value within half a step of the recorded one."""
    out = np.empty(coded.shape, dtype=np.float32)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi > lo:
            out[lo:hi] = (np.cumsum(coded[lo:hi].astype(np.int64), axis=0) * scales).astype(np.float32)
    return out


def _design_dict(genome: np.ndarray) -> dict:
    design = dataclasses.asdict(rover.design_from_genome(genome))
    design.pop("bodies")  # per-body mass properties: derivable, and bulky
    return design


def _summary_dict(summary) -> dict:
    out = {f.name: getattr(summary, f.name) for f in dataclasses.fields(summary) if f.name not in ("trajectory", "replay")}
    out["violations"] = [dataclasses.asdict(v) for v in summary.violations]
    return out


def generation_paths(directory: str | Path, generation: int) -> tuple[Path, Path]:
    folder = Path(directory)
    folder = folder if folder.name == "replay" else folder / "replay"
    return folder / f"gen_{generation:04d}.npz", folder / f"gen_{generation:04d}.json"


def write_generation(directory: str | Path, generation: int, terrain_seed: int, genomes: np.ndarray, summaries,
                     evaluations, policy: ReplayPolicy, generations: int | None = None, other_seeds=()) -> Path:
    """Write the files of one generation under ``<directory>/replay/``; returns the ``.npz`` path.

    ``summaries`` are the runs on ``terrain_seed`` (the only ones whose frames are stored); ``other_seeds`` is a list of
    ``(seed, summaries)`` for the generation's other terrain seeds, one summary per rover, described in the JSON only.
    ``generations`` (the run's total) only matters for ``policy.every``: the last generation is always recorded."""
    genomes = np.asarray(genomes, dtype=float)
    other_seeds = [(int(seed), list(runs)) for seed, runs in other_seeds]
    for seed, runs in other_seeds:
        if len(runs) != len(genomes):
            raise ValueError(f"other seed {seed} has {len(runs)} summaries for {len(genomes)} rovers")
    fitness = np.array([e.fitness for e in evaluations], dtype=float)
    total = generations if generations is not None else generation + 1
    wanted = policy.selected(fitness) if policy.records_generation(generation, total) else np.zeros(0, dtype=int)
    recorded = np.zeros(len(genomes), dtype=bool)
    recorded[[i for i in wanted if summaries[i].replay is not None]] = True
    chunks, steps = [], np.ones(len(genomes), dtype=np.int32)
    for i in range(len(genomes)):
        rows = np.zeros((0, len(COLUMNS)), dtype=np.float32)
        if recorded[i]:
            rows, steps[i] = thin(summaries[i].replay, policy.max_frames)
        chunks.append(rows.astype(np.float32, copy=False))
    offsets = np.concatenate(([0], np.cumsum([len(c) for c in chunks]))).astype(np.int64)
    npz_path, json_path = generation_paths(directory, generation)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path, generation=generation, terrain_seed=terrain_seed, terrain_seeds=np.array([terrain_seed, *(s for s, _ in other_seeds)]),
        hz=policy.hz, columns=np.array(COLUMNS), scales=SCALES,
        genomes=genomes, fitness=fitness, recorded=recorded, frames_q=encode_frames(np.concatenate(chunks), offsets),
        offsets=offsets, frame_step=steps)
    rank = np.empty(len(genomes), dtype=int)
    rank[_ranking(fitness)] = np.arange(len(genomes))
    rovers = [
        {"index": i, "rank": int(rank[i]), "recorded": bool(recorded[i]), "genome": genomes[i].tolist(),
         "decoded": genome_spec.decode(genomes[i]), "design": _design_dict(genomes[i]), "summary": _summary_dict(summaries[i]),
         "other_seeds": [{"seed": seed, "summary": _summary_dict(runs[i])} for seed, runs in other_seeds],
         "evaluation": {"fitness": evaluations[i].fitness, "details": evaluations[i].details}}
        for i in range(len(genomes))
    ]
    document = {"generation": generation, "terrain_seed": terrain_seed, "terrain_seeds": [terrain_seed, *(s for s, _ in other_seeds)],
                "rovers": rovers}
    json_path.write_text(json.dumps(ga.jsonable(document), indent=1), encoding="utf-8")
    return npz_path


@dataclass(frozen=True)
class GenerationReplay:
    generation: int
    terrain_seed: int  # the terrain of the frames (the first of ``terrain_seeds``)
    terrain_seeds: tuple[int, ...]  # every terrain the generation was scored on
    hz: float
    genomes: np.ndarray
    fitness: np.ndarray
    recorded: np.ndarray
    frames: np.ndarray
    offsets: np.ndarray
    frame_step: np.ndarray
    meta: dict

    @property
    def n_rovers(self) -> int:
        return len(self.genomes)

    def frames_of(self, i: int) -> np.ndarray:
        """The (F, 18) frames of rover ``i`` (``COLUMNS``); no rows when it was not recorded."""
        return self.frames[self.offsets[i]: self.offsets[i + 1]]


def load_generation(directory: str | Path, generation: int) -> GenerationReplay:
    """Read generation ``generation`` from a run directory (or its ``replay`` folder)."""
    npz_path, json_path = generation_paths(directory, generation)
    if not npz_path.exists():
        raise FileNotFoundError(f"no replay for generation {generation} in {npz_path.parent}")
    with np.load(npz_path) as data:
        arrays = {k: data[k] for k in data.files}
    if tuple(arrays["columns"]) != COLUMNS:
        raise ValueError(f"{npz_path} was written with other columns: {list(arrays['columns'])}")
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    return GenerationReplay(
        generation=int(arrays["generation"]), terrain_seed=int(arrays["terrain_seed"]),
        terrain_seeds=tuple(int(s) for s in arrays.get("terrain_seeds", [arrays["terrain_seed"]])), hz=float(arrays["hz"]),
        genomes=arrays["genomes"], fitness=arrays["fitness"], recorded=arrays["recorded"],
        frames=decode_frames(arrays["frames_q"], arrays["offsets"], arrays["scales"]),
        offsets=arrays["offsets"], frame_step=arrays["frame_step"], meta=meta)

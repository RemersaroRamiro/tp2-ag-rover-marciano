"""
replay.py: el formato y los archivos de las grabaciones de cada generación.

Para qué existe
    Permite volver a ver cómo se movió cada rover después de una corrida, sin simular de nuevo.
    Este archivo define qué se graba, cómo se comprime y cómo se vuelve a leer.

Qué contiene
    - COLUMNS: lo que se guarda en cada cuadro (tiempo, posición y orientación del chasis y el ángulo
      de cada articulación).
    - ReplayPolicy: decide qué rovers se graban (todos, los mejores o ninguno) y en qué generaciones.
    - write_generation: guarda replay/gen_NNNN.npz (los cuadros) y gen_NNNN.json (genoma, diseño y
      resultados de cada rover).
    - load_generation: vuelve a leer una generación grabada.

Cómo funciona
    Solo se graban cuadros en el primer terreno de cada generación; de los demás se guarda el resumen.
    Cada valor se redondea a un paso fijo (1 ms, 1 mm, 1 mrad en las articulaciones) y se guarda como
    diferencia con el cuadro anterior, lo que comprime unas 4 veces mejor sin cambios visibles.

Dónde encaja
    Etapa: resultados. Usa ga, genome_spec, rover y evolution_params. Lo usan evolve.py (que graba
    cada generación), evaluate.py y baselines.py.
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
N_FREE = 7
SCALES = np.array([1e-3, 1e-3, 1e-3, 1e-3, *([1e-4] * 4), *([1e-3] * len(JOINTS))])


def qpos_addresses(model, prefix: str) -> np.ndarray:
    import mujoco

    def address(name: str) -> int:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}")
        if joint < 0:
            raise ValueError(f"no joint {prefix}{name} in the model")
        return int(model.jnt_qposadr[joint])

    free = address("free")
    return np.array([*range(free, free + N_FREE), *(address(j) for j in JOINTS)], dtype=int)


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class ReplayPolicy:
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
        if self.mode == "none":
            return np.zeros(0, dtype=int)
        order = _ranking(fitness)
        return order if self.mode == "all" else order[: self.best_k]


def _ranking(fitness: np.ndarray) -> np.ndarray:
    return np.argsort(-np.where(np.isfinite(fitness), fitness, -np.inf), kind="stable")


def thin(frames: np.ndarray, max_frames: int) -> tuple[np.ndarray, int]:
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


def encode_frames(frames: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    q = np.round(frames.astype(np.float64) / SCALES).astype(np.int64)
    out = np.empty_like(q)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi > lo:
            out[lo] = q[lo]
            out[lo + 1: hi] = np.diff(q[lo:hi], axis=0)
    small = np.abs(out).max(initial=0) <= np.iinfo(np.int16).max
    return out.astype(np.int16 if small else np.int32)


def decode_frames(coded: np.ndarray, offsets: np.ndarray, scales: np.ndarray = SCALES) -> np.ndarray:
    out = np.empty(coded.shape, dtype=np.float32)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi > lo:
            out[lo:hi] = (np.cumsum(coded[lo:hi].astype(np.int64), axis=0) * scales).astype(np.float32)
    return out


def _design_dict(genome: np.ndarray) -> dict:
    design = dataclasses.asdict(rover.design_from_genome(genome))
    design.pop("bodies")
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
    terrain_seed: int
    terrain_seeds: tuple[int, ...]
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
        return self.frames[self.offsets[i]: self.offsets[i + 1]]


def load_generation(directory: str | Path, generation: int) -> GenerationReplay:
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

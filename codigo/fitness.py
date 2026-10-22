"""
fitness.py: calcula la aptitud de cada rover.

Para qué existe
    El AG necesita un único número por individuo para poder compararlos.
    Este archivo convierte el resultado de la simulación (distancia recorrida
    y restricciones violadas) en ese número.

Qué contiene
    - fitness_from_summary: aptitud de un rover en un solo terreno.
    - fitness_from_summaries: aptitud de un rover en varios terrenos.
    - stop_cluster_fraction: mide si muchos rovers se frenaron en el mismo lugar.

Cómo funciona
    Se aplican las reglas de Deb:
    - Diseño válido: aptitud = distancia media recorrida, en metros (≥ 0).
    - Diseño inválido: aptitud = -(1 + violación total), siempre negativa.
    Así un diseño válido siempre le gana a uno inválido, sin pesos que ajustar.

Dónde encaja
    Etapa: función de aptitud. Lo usa evolve.py después de simular cada generación.

Fuentes
    - Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import evaluate
import evolution_params as params
import ga


@dataclass(frozen=True)
class FitnessParams:
    infeasible_offset: float = params.INFEASIBLE_OFFSET

    def __post_init__(self):
        if not (math.isfinite(self.infeasible_offset) and self.infeasible_offset > 0):
            raise ValueError(f"infeasible_offset must be positive and finite, got {self.infeasible_offset}")


def violation_code(v) -> str:
    return f"{v.code}:{v.detail}" if v.detail else v.code


def _violation_fields(summary: evaluate.RunSummary) -> list[dict]:
    return [{"code": v.code, "detail": v.detail, "value": v.value, "limit": v.limit, "magnitude": v.magnitude}
            for v in summary.violations]


def _same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _same_violations(a: list[dict], b: list[dict]) -> bool:
    return len(a) == len(b) and all(x.keys() == y.keys() and all(_same_value(x[k], y[k]) for k in x) for x, y in zip(a, b))


def _run_details(summary: evaluate.RunSummary) -> dict:
    return {
        "seed": summary.seed,
        "distance_m": float(summary.max_distance_m),
        "status": summary.status,
        "reason": summary.reason,
        "finished": summary.status == "finished",
        "section_reached": summary.section_reached,
        "alive_time_s": summary.alive_time_s,
        "drive_energy_wh": summary.drive_energy_wh,
        "net_drain_wh": summary.net_drain_wh,
    }


def _design_details(summary: evaluate.RunSummary) -> dict:
    total_violation = float(sum(v.magnitude for v in summary.violations))
    return {
        "total_violation": total_violation,
        "legal": not summary.violations,
        "violations": _violation_fields(summary),
        "violation_codes": [violation_code(v) for v in summary.violations],
        "total_mass_kg": summary.total_mass_kg,
        "repairs": list(summary.repairs),
    }


def fitness_from_summary(summary: evaluate.RunSummary, fitness_params: FitnessParams | None = None) -> ga.Evaluation:
    fitness_params = fitness_params or FitnessParams()
    distance = float(summary.max_distance_m)
    design = _design_details(summary)
    if not (math.isfinite(distance) and math.isfinite(design["total_violation"])):
        value = -math.inf
    elif design["legal"]:
        value = max(distance, 0.0)
    else:
        value = -(fitness_params.infeasible_offset + design["total_violation"])
    run = _run_details(summary)
    details = {
        "seed": run["seed"], "distance_m": run["distance_m"], "total_violation": design["total_violation"], "legal": design["legal"],
        "violations": design["violations"], "violation_codes": design["violation_codes"], "status": run["status"],
        "reason": run["reason"], "finished": run["finished"], "section_reached": run["section_reached"],
        "alive_time_s": run["alive_time_s"], "drive_energy_wh": run["drive_energy_wh"], "net_drain_wh": run["net_drain_wh"],
        "total_mass_kg": design["total_mass_kg"], "repairs": design["repairs"],
    }
    return ga.Evaluation(fitness=value, details=details)


def fitness_from_summaries(summaries, fitness_params: FitnessParams | None = None) -> ga.Evaluation:
    summaries = list(summaries)
    if not summaries:
        raise ValueError("a design needs at least one run to be scored")
    fitness_params = fitness_params or FitnessParams()
    first = fitness_from_summary(summaries[0], fitness_params)
    if not all(_same_violations(_violation_fields(s), _violation_fields(summaries[0])) for s in summaries[1:]):
        raise ValueError("the constraint violations of a design must be the same on every terrain seed "
                         f"(got {[s.violations for s in summaries]}): legality depends on the genome only")
    runs = [_run_details(s) for s in summaries]
    distances = [r["distance_m"] for r in runs]
    if not all(math.isfinite(d) for d in distances):
        value = -math.inf
    elif first.details["legal"]:
        value = sum(max(d, 0.0) for d in distances) / len(distances)
    else:
        value = first.fitness
    details = {**first.details, "distance_m": sum(distances) / len(distances), "finished": all(r["finished"] for r in runs),
               "seeds": [r["seed"] for r in runs], "distances_m": distances,
               "min_distance_m": min(distances), "max_distance_m": max(distances),
               "finished_seeds": sum(r["finished"] for r in runs),
               "per_seed": [{k: r[k] for k in ("seed", "distance_m", "status", "reason", "finished", "section_reached", "alive_time_s")}
                            for r in runs]}
    return ga.Evaluation(fitness=value, details=details)


def stop_cluster_fraction(distances_m, window_m: float = 1.0) -> float:
    d = np.asarray(list(distances_m), dtype=float)
    if not len(d):
        return 0.0
    finite = d[np.isfinite(d)]
    if not len(finite):
        return 0.0
    within = np.abs(finite[:, None] - finite[None, :]) <= window_m
    return float(within.sum(axis=1).max()) / len(d)

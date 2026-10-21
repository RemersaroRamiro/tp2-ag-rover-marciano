"""The fitness of a rover, with Deb's (2000) feasibility rules for the mission constraints.

Deb, K. (2000), "An efficient constraint handling method for genetic algorithms", Comput. Methods Appl. Mech. Engrg.
186:311-338, compares two designs like this:

    (a) a feasible (legal) design always beats an infeasible (illegal) one;
    (b) two feasible designs: the one that drove further wins;
    (c) two infeasible designs: the one with the smaller total constraint violation wins.

The GA (DEAP tournament or roulette selection, elitism; see ``ga``) only compares one number per design, so the three
rules are folded into a scalar that orders designs exactly as they do, with no weight to tune. (Roulette needs positive
values, so ``ga.select_roulette`` spins on the shifted fitness ``f - min(f) + eps``.) The two branches are NOT in the
same unit:

    legal:     fitness = mean over the terrain seeds of max_distance_m        metres, >= 0
    illegal:   fitness = -(INFEASIBLE_OFFSET + total_violation)               dimensionless, < -INFEASIBLE_OFFSET < 0

so a fitness is a distance in metres only when it is >= 0 (a legal design); a negative one is an index of how far the
design is from the legal region, and is not metres.

* ``max_distance_m`` is the furthest x the rover reached minus the spawn x, whatever happened afterwards (it may have
  tipped over or got stuck later; what it drove still counts). There are no checkpoints and no other reward. It is
  never negative, so every legal design scores at least 0.
* ``total_violation`` is the sum of ``violation.magnitude`` over the broken mission constraints
  (``rover.check_constraints``: total mass, static tip-over, stowed envelope, ground clearance); a magnitude is the
  RELATIVE overshoot of one limit, 0.1 = 10 % past it. It is >= 0 and dimensionless.
* ``INFEASIBLE_OFFSET`` (> 0) is what makes rule (a) hold: every illegal score is below -offset, below every legal one.
  Its value only shifts the illegal scores; it changes no ordering.

Several terrain seeds per generation (``fitness_from_summaries``). One rock layout can be a wall: a single boulder in
the lane stopped the whole population at the same x in 5 of 20 generations, every rover got the same score and the
selection of that generation was blind. So every rover is scored on ``N`` terrains and Deb's rules apply to the design,
not to one run of it:

* Legality is a property of the DESIGN (``rover.check_constraints`` reads the genome, never the terrain), so a design
  is legal or illegal on every seed at once; the violation is the same number on each (a mismatch is a bug and raises).
* A legal design scores the MEAN of its ``max_distance_m`` over the seeds, not the minimum. The mean is the quantity the
  final validation reports (mean distance on unseen terrains), so selection and validation measure the same thing; it is
  the estimate of the expected distance on a random terrain. And it keeps a gradient where the minimum would not: if one
  seed holds a boulder that stops everybody, the minimum is pinned by it for all designs (the blind generation again,
  now on that seed), while the mean still ranks designs by what they do on the free seed. The price is that a design
  that is very good on one terrain and poor on the other can outscore a steadier one; the per-seed distances are kept in
  ``details`` so that can be looked at.
* An illegal design scores ``-(offset + total_violation)`` exactly as with one seed; the distances it drove stay in
  ``details`` for the analysis.

Illegal rovers are still built and simulated, so their distance, status and reasons are kept in ``Evaluation.details``
for the analysis (and for the replay), but only the violation enters their score.

``fitness_from_summary`` (one run) and ``fitness_from_summaries`` (one design, several runs) are pure functions of
``evaluate.RunSummary`` objects and the parameters. A failed evaluation (a non-finite distance or violation on any seed)
scores ``-inf``: never the best.
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
    """``code`` or ``code:detail`` (e.g. ``envelope_exceeds:width``): one label per kind of violation."""
    return f"{v.code}:{v.detail}" if v.detail else v.code


def _violation_fields(summary: evaluate.RunSummary) -> list[dict]:
    return [{"code": v.code, "detail": v.detail, "value": v.value, "limit": v.limit, "magnitude": v.magnitude}
            for v in summary.violations]


def _same_value(a, b) -> bool:
    """``a == b``, except that two NaN floats are the same value (a non-finite violation repeats on every seed)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _same_violations(a: list[dict], b: list[dict]) -> bool:
    """NaN-safe equality of two ``_violation_fields`` lists."""
    return len(a) == len(b) and all(x.keys() == y.keys() and all(_same_value(x[k], y[k]) for k in x) for x, y in zip(a, b))


def _run_details(summary: evaluate.RunSummary) -> dict:
    """What one run of the rover did (the part of ``details`` that belongs to a terrain seed)."""
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
    """What belongs to the design, the same on every seed."""
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
    """Fitness and a JSON-friendly ``details`` dict of one rover's run. The fitness is a distance in metres for a legal
    design and the dimensionless ``-(offset + total violation)`` for an illegal one (see the module docstring)."""
    fitness_params = fitness_params or FitnessParams()
    distance = float(summary.max_distance_m)
    design = _design_details(summary)
    if not (math.isfinite(distance) and math.isfinite(design["total_violation"])):
        value = -math.inf
    elif design["legal"]:
        value = max(distance, 0.0)  # rule (b): further is better
    else:
        value = -(fitness_params.infeasible_offset + design["total_violation"])  # rules (a) and (c)
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
    """Fitness of ONE design from its runs on several terrain seeds (``summaries``, the recorded seed first).

    Legal: the mean of ``max(max_distance_m, 0)`` over the seeds. Illegal: ``-(offset + total violation)``, the violation
    being the design's (the same on every seed). Any failed seed gives ``-inf``. ``details`` holds what ``fitness_from_summary``
    gives for the FIRST seed (``seed``, ``status``, ``reason``, ``section_reached``, ``alive_time_s``, energy) except
    ``distance_m`` (the mean over the seeds, the basis of the score) and ``finished`` (True only when every seed finished),
    plus ``seeds``, ``distances_m``, ``min_distance_m`` / ``max_distance_m``, ``finished_seeds`` and ``per_seed``: one
    dict per seed with ``seed``, ``distance_m``, ``status``, ``reason``, ``finished``, ``section_reached`` and ``alive_time_s``.
    With one summary every score and every old detail equals ``fitness_from_summary``'s."""
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
        value = first.fitness  # -(offset + violation), already -inf for a non-finite violation
    details = {**first.details, "distance_m": sum(distances) / len(distances), "finished": all(r["finished"] for r in runs),
               "seeds": [r["seed"] for r in runs], "distances_m": distances,
               "min_distance_m": min(distances), "max_distance_m": max(distances),
               "finished_seeds": sum(r["finished"] for r in runs),
               "per_seed": [{k: r[k] for k in ("seed", "distance_m", "status", "reason", "finished", "section_reached", "alive_time_s")}
                            for r in runs]}
    return ga.Evaluation(fitness=value, details=details)


def stop_cluster_fraction(distances_m, window_m: float = 1.0) -> float:
    """Share of a population that stopped at about the same place: the largest number of rovers whose distance is within
    ``window_m`` of one rover's, over the population size. 1.0 = every rover stopped at the same x (a blind generation:
    whatever the designs, one obstacle decided the score); near 1/N = nobody shares a stop point. Failed (non-finite)
    runs count in the population but never in a cluster; an empty population gives 0."""
    d = np.asarray(list(distances_m), dtype=float)
    if not len(d):
        return 0.0
    finite = d[np.isfinite(d)]
    if not len(finite):
        return 0.0
    within = np.abs(finite[:, None] - finite[None, :]) <= window_m
    return float(within.sum(axis=1).max()) / len(d)

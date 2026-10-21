"""The rover genome contract: 15 genes, each a number in [0, 1] that maps onto a physical range.

The GA only ever sees [0, 1] vectors (`GENOME_LENGTH` long, in `GENES` order); the rover decoder reads the
physical values from `decode`. Fixed topology: six wheels on a symmetric rocker-bogie, so one gene describes
both sides. Ranges are sourced from the T13 constructibility research where noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gene:
    name: str
    low: float
    high: float
    scale: str = "linear"  # "linear" or "log" (log for quantities spanning an order of magnitude)
    description: str = ""


GENES: tuple[Gene, ...] = (
    # chassis
    Gene("chassis_length_m", 1.5, 3.5, description="chassis box length; Perseverance ~3 m overall"),
    Gene("track_width_m", 1.5, 3.0, description="lateral distance between left and right wheel centres; Perseverance ~2.7 m wide"),
    Gene("chassis_height_m", 0.3, 0.8, description="chassis box height"),
    # rocker-bogie suspension
    Gene("rocker_pivot_height_m", 0.4, 1.2, description="height of the rocker pivot above the wheel axles"),
    Gene("rocker_pivot_offset", -0.3, 0.3, description="rocker pivot position along x, as a fraction of chassis length"),
    Gene("rocker_front_arm_m", 0.5, 1.5, description="rocker arm from pivot to the front wheel"),
    Gene("rocker_rear_arm_m", 0.3, 1.2, description="rocker arm from pivot to the bogie pivot"),
    Gene("bogie_length_m", 0.4, 1.4, description="distance between the middle and rear wheel axles"),
    # wheels
    Gene("wheel_diameter_m", 0.3, 0.8, description="all six wheels; Perseverance 0.525 m"),
    Gene("wheel_width_m", 0.1, 0.5, description="all six wheels; Perseverance ~0.4 m"),
    # drive and energy
    Gene("motor_peak_torque_nm", 15.0, 150.0, "log", "per-wheel peak torque; rolling needs ~14-42 N*m, a 20 deg slope ~+47, traction caps near ~85 N*m for a ~900 kg rover in Mars gravity (T13 research)"),
    Gene("battery_energy_wh", 500.0, 4000.0, "log", "battery capacity; Perseverance 2408 Wh (2 strings x 43 Ah x 28 V)"),
    # driving behaviour (controller)
    Gene("target_speed_mps", 0.3, 1.0, description="cruise speed; real rovers drive ~0.04 m/s, sped up for the GA (declared simplification)"),
    Gene("torque_limit_fraction", 0.3, 1.0, description="fraction of the peak torque the controller may use"),
    Gene("slope_brake", 0.0, 1.0, description="how strongly the controller slows down with pitch (0 = never)"),
)

GENOME_LENGTH = len(GENES)


def _to_value(gene: Gene, u: float) -> float:
    if gene.scale == "log":
        return math.exp(math.log(gene.low) + u * (math.log(gene.high) - math.log(gene.low)))
    return gene.low + u * (gene.high - gene.low)


def _to_unit(gene: Gene, value: float) -> float:
    if gene.scale == "log":
        return (math.log(value) - math.log(gene.low)) / (math.log(gene.high) - math.log(gene.low))
    return (value - gene.low) / (gene.high - gene.low)


def decode(genome: np.ndarray) -> dict[str, float]:
    """Physical value of every gene, by name. Values outside [0, 1] are clipped."""
    u = np.asarray(genome, dtype=float)
    if u.shape != (GENOME_LENGTH,):
        raise ValueError(f"genome must have shape ({GENOME_LENGTH},), got {u.shape}")
    u = np.clip(u, 0.0, 1.0)
    return {gene.name: _to_value(gene, float(x)) for gene, x in zip(GENES, u)}


def encode(values: dict[str, float]) -> np.ndarray:
    """Inverse of `decode`: the [0, 1] genome for the given physical values."""
    return np.array([_to_unit(gene, values[gene.name]) for gene in GENES])

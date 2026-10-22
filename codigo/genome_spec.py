"""
genome_spec.py: la tabla de genes del rover.

Para qué existe
    Define el cromosoma: qué significa cada uno de los 15 genes y entre qué valores físicos se mueve.
    El AG solo ve números entre 0 y 1; este archivo los traduce a medidas reales (m, N·m, Wh, m/s)
    y es el único lugar donde están esos rangos.

Qué contiene
    - Gene: nombre, mínimo, máximo, escala (lineal o logarítmica) y descripción de un gen.
    - GENES: los 15 genes (chasis, suspensión rocker-bogie, ruedas, motor y batería, controlador).
    - GENOME_LENGTH: la cantidad de genes (15).
    - decode: pasa un genoma en [0, 1] a valores físicos (lo que quede fuera de [0, 1] se recorta).
    - encode: hace lo inverso, de valores físicos a [0, 1].

Cómo funciona
    Cada gen se convierte linealmente entre su mínimo y su máximo; el torque del motor y la energía
    de la batería usan escala logarítmica porque abarcan un orden de magnitud. La topología es fija:
    seis ruedas en una suspensión rocker-bogie simétrica, así que un gen describe los dos lados.

Dónde encaja
    Etapa: cromosoma. No importa otros archivos del proyecto.
    Lo usan ga, rover, evaluate, evolve, replay, reference_rovers y track_rocks. El notebook lo
    importa para mostrar la tabla de genes.

Fuentes
    - Investigación de constructibilidad T13 (por ejemplo, el rango del torque del motor) y medidas de
      Perseverance (ruedas de 0.525 m, batería de 2408 Wh).
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
    scale: str = "linear"
    description: str = ""


GENES: tuple[Gene, ...] = (
    Gene("chassis_length_m", 1.5, 3.5, description="chassis box length; Perseverance ~3 m overall"),
    Gene("track_width_m", 1.5, 3.0, description="lateral distance between left and right wheel centres; Perseverance ~2.7 m wide"),
    Gene("chassis_height_m", 0.3, 0.8, description="chassis box height"),
    Gene("rocker_pivot_height_m", 0.4, 1.2, description="height of the rocker pivot above the wheel axles"),
    Gene("rocker_pivot_offset", -0.3, 0.3, description="rocker pivot position along x, as a fraction of chassis length"),
    Gene("rocker_front_arm_m", 0.5, 1.5, description="rocker arm from pivot to the front wheel"),
    Gene("rocker_rear_arm_m", 0.3, 1.2, description="rocker arm from pivot to the bogie pivot"),
    Gene("bogie_length_m", 0.4, 1.4, description="distance between the middle and rear wheel axles"),
    Gene("wheel_diameter_m", 0.3, 0.8, description="all six wheels; Perseverance 0.525 m"),
    Gene("wheel_width_m", 0.1, 0.5, description="all six wheels; Perseverance ~0.4 m"),
    Gene("motor_peak_torque_nm", 15.0, 150.0, "log", "per-wheel peak torque; rolling needs ~14-42 N*m, a 20 deg slope ~+47, traction caps near ~85 N*m for a ~900 kg rover in Mars gravity (T13 research)"),
    Gene("battery_energy_wh", 500.0, 4000.0, "log", "battery capacity; Perseverance 2408 Wh (2 strings x 43 Ah x 28 V)"),
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
    u = np.asarray(genome, dtype=float)
    if u.shape != (GENOME_LENGTH,):
        raise ValueError(f"genome must have shape ({GENOME_LENGTH},), got {u.shape}")
    u = np.clip(u, 0.0, 1.0)
    return {gene.name: _to_value(gene, float(x)) for gene, x in zip(GENES, u)}


def encode(values: dict[str, float]) -> np.ndarray:
    return np.array([_to_unit(gene, values[gene.name]) for gene in GENES])

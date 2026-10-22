"""
reference_rovers.py: rovers de referencia con valores físicos conocidos.

Para qué existe
    Para saber si un diseño evolucionado es bueno hace falta compararlo con uno conocido. Este archivo
    guarda esos diseños de referencia, escritos como el valor físico de cada gen.

Qué contiene
    - PERSEVERANCE_LIKE: un rover del tamaño y la masa de Perseverance (~1 t, ruedas de 0.525 m). En el
      terreno de semilla 0 queda atascado detrás de una roca de ~0.5 m, cerca de x = 38 m.
    - FINISHER: el mismo rover con tres genes cambiados (ruedas de 0.7 m, 1 m/s y sin freno en
      pendientes); recorre toda la pista en unos 500 s simulados.
    - genome: convierte esos valores físicos en un genoma en [0, 1].

Dónde encaja
    Etapa: resultados. Usa genome_spec. evolve.py valida PERSEVERANCE_LIKE en cada corrida como línea
    de base (Tabla 4 de la entrega parcial). El notebook también lo importa.
"""

from __future__ import annotations

import numpy as np

import genome_spec

PERSEVERANCE_LIKE: dict[str, float] = {
    "chassis_length_m": 2.0, "track_width_m": 2.3, "chassis_height_m": 0.4,
    "rocker_pivot_height_m": 0.55, "rocker_pivot_offset": 0.0,
    "rocker_front_arm_m": 1.2, "rocker_rear_arm_m": 0.8, "bogie_length_m": 1.0,
    "wheel_diameter_m": 0.525, "wheel_width_m": 0.4,
    "motor_peak_torque_nm": 120.0, "battery_energy_wh": 2000.0,
    "target_speed_mps": 0.5, "torque_limit_fraction": 0.8, "slope_brake": 0.5,
}

FINISHER: dict[str, float] = {**PERSEVERANCE_LIKE, "wheel_diameter_m": 0.7, "target_speed_mps": 1.0, "slope_brake": 0.0}


def genome(values: dict[str, float]) -> np.ndarray:
    return genome_spec.encode(values)

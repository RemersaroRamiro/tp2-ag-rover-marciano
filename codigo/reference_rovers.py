"""
reference_rovers.py: el rover de referencia, con valores físicos conocidos.

Para qué existe
    Para saber si un diseño evolucionado es bueno hace falta compararlo con uno conocido. Este archivo
    guarda ese diseño de referencia, escrito como el valor físico de cada gen.

Qué contiene
    - PERSEVERANCE_LIKE: un rover del tamaño y la masa de Perseverance (~1 t, ruedas de 0.525 m). En el
      terreno de semilla 0 queda atascado detrás de una roca de ~0.5 m, cerca de x = 38 m.
    - genome: convierte esos valores físicos en un genoma en [0, 1].

Dónde encaja
    Etapa: resultados. Usa genome_spec. evolve.py valida PERSEVERANCE_LIKE en cada corrida como línea
    de base. El notebook también lo importa.
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

def genome(values: dict[str, float]) -> np.ndarray:
    return genome_spec.encode(values)

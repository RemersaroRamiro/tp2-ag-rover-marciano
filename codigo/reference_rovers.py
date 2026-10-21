"""Named reference rovers (physical gene values) shared by the tools and the tests.

``PERSEVERANCE_LIKE`` is a rover of the size and mass of Perseverance (~1 t, 0.525 m wheels); on the seed-0 track it
gets stuck behind a ~0.5 m rock in the rock field (x ~ 38 m). ``FINISHER`` only changes three genes (0.7 m wheels,
1 m/s, no slope brake) and drives the whole track in ~500 s simulated.
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
    """The [0, 1] genome of physical gene ``values``."""
    return genome_spec.encode(values)

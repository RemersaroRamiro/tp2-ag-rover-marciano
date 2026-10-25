"""
energy.py: el modelo de energía del rover (batería, RTG y consumo).

Para qué existe
    Un rover que gasta toda la energía disponible tiene que detenerse. Este archivo lleva la cuenta
    de la batería de cada rover durante la simulación, sin depender de MuJoCo.

Qué contiene
    - positive_mechanical_power_w: potencia mecánica de las ruedas (solo la que impulsa, sin regeneración).
    - Battery: estado de carga de la batería; avanza paso a paso e indica cuándo se agotó.

Cómo funciona
    Potencia neta = potencia de los motores / eficiencia + consumo de la aviónica - potencia del RTG.
    Si es positiva, descarga la batería; si es negativa, la recarga hasta llenarla. Solo se puede usar
    el 60 % de la capacidad: al llegar a ese límite el rover queda detenido para siempre.
    Simplificaciones: empujar contra un obstáculo sin moverse no consume energía, y el RTG y la
    aviónica tienen una potencia constante.

Dónde encaja
    Etapa: simulación. Usa rover_params. Lo usa rover_run.py.
"""

from __future__ import annotations

import numpy as np

import rover_params as rp


def positive_mechanical_power_w(torque_nm, wheel_speed_rad_s) -> float:
    return float(np.maximum(np.asarray(torque_nm, dtype=float) * np.asarray(wheel_speed_rad_s, dtype=float), 0.0).sum())


class Battery:
    def __init__(self, capacity_wh: float):
        self.capacity_wh = float(capacity_wh)
        self.usable_wh = self.capacity_wh * rp.BATTERY_MAX_DEPTH_OF_DISCHARGE
        self.energy_wh = self.capacity_wh
        self.drive_energy_wh = 0.0
        self.dead = False

    def step(self, mechanical_power_w: float, dt_s: float) -> None:
        if self.dead:
            return
        electrical_w = max(mechanical_power_w, 0.0) / rp.DRIVE_EFFICIENCY
        net_w = electrical_w + rp.HOTEL_LOAD_W - rp.RTG_POWER_W
        self.drive_energy_wh += electrical_w * dt_s / 3600.0
        self.energy_wh = min(self.energy_wh - net_w * dt_s / 3600.0, self.capacity_wh)
        if self.energy_wh <= self.capacity_wh - self.usable_wh:
            self.energy_wh = self.capacity_wh - self.usable_wh
            self.dead = True

    @property
    def net_drain_wh(self) -> float:
        return self.capacity_wh - self.energy_wh

    @property
    def usable_remaining_wh(self) -> float:
        return max(self.usable_wh - self.net_drain_wh, 0.0)

    @property
    def soc(self) -> float:
        return self.energy_wh / self.capacity_wh

"""Energy model: a battery topped up by the RTG, drained by the drive motors and the avionics.

Per step (no MuJoCo needed, the caller supplies the mechanical power):

    electrical drive power = sum over wheels of max(0, torque * wheel speed) / DRIVE_EFFICIENCY
    net power drawn from the battery = drive electrical + HOTEL_LOAD_W - RTG_POWER_W

A positive net power discharges the battery; a negative one (RTG surplus, mostly when idle) recharges it up to
full. Regenerative braking is off: a wheel braking against its motion (torque and speed of opposite sign) draws
no power and returns none. Only ``BATTERY_MAX_DEPTH_OF_DISCHARGE`` of the capacity is usable: when that much has
been drawn the rover is dead (``Battery.dead``) and stays dead.

Known simplifications: the motor's resistive loss at stall (torque without motion) is not modelled, so pushing
against an obstacle costs no power; the RTG output is constant; the hotel load is constant while the rover is awake.
"""

from __future__ import annotations

import numpy as np

import rover_params as rp


def positive_mechanical_power_w(torque_nm, wheel_speed_rad_s) -> float:
    """Total mechanical drive power of the wheels: the sum of max(0, torque * speed), in watts (no regeneration)."""
    return float(np.maximum(np.asarray(torque_nm, dtype=float) * np.asarray(wheel_speed_rad_s, dtype=float), 0.0).sum())


class Battery:
    """State of charge of the battery of one rover. Starts full."""

    def __init__(self, capacity_wh: float):
        self.capacity_wh = float(capacity_wh)
        self.usable_wh = self.capacity_wh * rp.BATTERY_MAX_DEPTH_OF_DISCHARGE
        self.energy_wh = self.capacity_wh  # stored energy
        self.drive_energy_wh = 0.0  # electrical energy the drive motors have drawn so far
        self.dead = False

    def step(self, mechanical_power_w: float, dt_s: float) -> None:
        """Advance ``dt_s`` seconds with ``mechanical_power_w`` (>= 0) of drive power at the wheels."""
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
        """Energy missing from the battery relative to full (RTG recharge already subtracted)."""
        return self.capacity_wh - self.energy_wh

    @property
    def usable_remaining_wh(self) -> float:
        return max(self.usable_wh - self.net_drain_wh, 0.0)

    @property
    def soc(self) -> float:
        """State of charge as a fraction of the full capacity (the rover dies at ``1 - depth of discharge``)."""
        return self.energy_wh / self.capacity_wh

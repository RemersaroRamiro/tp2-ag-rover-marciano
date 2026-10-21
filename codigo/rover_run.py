"""Run-state tracking for one rover: progress, status, energy, section reached and a compact trajectory log.

A run ends the first time one of these happens (checked in this order on every control tick):

* ``finished``: the rover crossed the finish line (only when a finish x is known, i.e. on the track);
* ``dead(energy)``: the battery's usable energy is exhausted;
* ``dead(out_of_lane)``: ``|y|`` beyond ``RunLimits.lane_half_width_m``. On the track this is the 20 m playable lane
  (``mars_params.LANE_WIDTH_M``), not the wider strip: outside the lane the rocks thin out and the cross slope fades
  (``track_rocks``, ``track_terrain``), so driving along the flank would be an easier, unintended route for a
  fitness that only counts distance along +x. The controller steers back to the centre, so a healthy rover never
  gets there. Off (``None``) on flat ground;
* ``dead(tipped)``: ``|roll|`` or ``|pitch|`` beyond ``TIP_LIMIT_DEG`` for more than ``TIP_TIME_S``;
* ``dead(stuck)``: the furthest x reached grew by less than ``STUCK_PROGRESS_M`` in the last ``STUCK_WINDOW_S``;
* ``dead(timeout)``: simulated time reached ``RunLimits.timeout_s`` (the normal end of a rover that keeps driving).

The first cause sticks: later samples change nothing. The fitness basis is ``max_distance_m`` (furthest x reached
minus the spawn x) and ``alive_time_s`` is the time at which the status left ``running``.

``RunTracker`` is pure (it consumes ``RunSample`` values, no MuJoCo); ``RoverRun`` wires it to a rover in a model:
controller, battery and tracker. Call ``RoverRun.update(data)`` after every ``mj_step`` (``simulate`` does).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

import controller as ctl
import energy
import mars_params as mp
import rover_params as rp

TRAJECTORY_COLUMNS = ("t", "x", "y", "z", "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg", "speed_mps", "soc")


@dataclass(frozen=True)
class RunLimits:
    """When a run ends (defaults in ``rover_params``)."""

    timeout_s: float = 300.0
    tip_limit_deg: float = rp.TIP_LIMIT_DEG
    tip_time_s: float = rp.TIP_TIME_S
    stuck_progress_m: float = rp.STUCK_PROGRESS_M
    stuck_window_s: float = rp.STUCK_WINDOW_S
    lane_half_width_m: float | None = None  # None: no lane (flat ground)

    @classmethod
    def on_track(cls, timeout_s: float = 300.0, **kw) -> "RunLimits":
        """Limits for the Jezero track: the lane is ``mars_params.LANE_WIDTH_M`` wide."""
        return cls(timeout_s=timeout_s, lane_half_width_m=mp.LANE_WIDTH_M / 2.0, **kw)


@dataclass(frozen=True)
class RunSample:
    """One observation of a rover (chassis pose in the world frame)."""

    t: float
    x: float
    y: float
    z: float
    quat: tuple[float, float, float, float]  # w, x, y, z
    roll_deg: float
    pitch_deg: float
    yaw_rad: float
    speed_mps: float
    soc: float
    energy_dead: bool = False

    def row(self) -> tuple[float, ...]:
        return (self.t, self.x, self.y, self.z, *self.quat, self.roll_deg, self.pitch_deg, math.degrees(self.yaw_rad), self.speed_mps, self.soc)


class RunTracker:
    """Status, progress and trajectory of one run, from a stream of ``RunSample``."""

    def __init__(self, x0: float, limits: RunLimits | None = None, section_at: Callable[[float], object] | None = None,
                 finish_x: float | None = None):
        self.x0 = x0
        self.limits = limits or RunLimits()
        self.finish_x = finish_x
        self._section_at = section_at
        self.status = "running"
        self.reason: str | None = None
        self.max_x_m = x0
        self.x_m = x0
        self.alive_time_s = 0.0
        self.last: RunSample | None = None
        self._rows: list[tuple[float, ...]] = []
        self._next_log_t = 0.0
        self._tip_since: float | None = None
        self._progress: deque[tuple[float, float]] = deque()  # (t, max_x) about once a second, for the stuck rule

    # -- derived
    @property
    def distance_m(self) -> float:
        """Current x relative to the spawn."""
        return self.x_m - self.x0

    @property
    def max_distance_m(self) -> float:
        """Furthest x reached relative to the spawn: the fitness basis."""
        return self.max_x_m - self.x0

    @property
    def section_reached(self) -> str | None:
        """Name of the track section at the furthest x reached."""
        return None if self._section_at is None else self._section_at(self.max_x_m).name

    @property
    def section_now(self) -> str | None:
        return None if self._section_at is None else self._section_at(self.x_m).name

    # -- updating
    def update(self, s: RunSample) -> None:
        if self.status != "running":
            return
        self.last, self.x_m, self.alive_time_s = s, s.x, s.t
        self.max_x_m = max(self.max_x_m, s.x)
        if s.t + 1e-9 >= self._next_log_t:
            self._rows.append(s.row())
            # anchored to this sample, not accumulated: a gap in the samples must not make the log catch up row by row
            self._next_log_t = (math.floor((s.t + 1e-9) / rp.TRAJECTORY_LOG_DT_S) + 1) * rp.TRAJECTORY_LOG_DT_S
        outcome = self._outcome(s)
        if outcome is not None:
            self.status, self.reason = outcome
            if self._rows[-1][0] != s.t:
                self._rows.append(s.row())

    def _outcome(self, s: RunSample) -> tuple[str, str | None] | None:
        lim = self.limits
        if self.finish_x is not None and s.x >= self.finish_x:
            return "finished", None
        if s.energy_dead:
            return "dead", "energy"
        if lim.lane_half_width_m is not None and abs(s.y) > lim.lane_half_width_m:
            return "dead", "out_of_lane"
        if max(abs(s.roll_deg), abs(s.pitch_deg)) > lim.tip_limit_deg:
            if self._tip_since is None:
                self._tip_since = s.t
            elif s.t - self._tip_since > lim.tip_time_s:
                return "dead", "tipped"
        else:
            self._tip_since = None
        if self._stuck(s.t):
            return "dead", "stuck"
        if s.t + 1e-9 >= lim.timeout_s:
            return "dead", "timeout"
        return None

    def _stuck(self, t: float) -> bool:
        lim = self.limits
        if not self._progress or t - self._progress[-1][0] >= 1.0 - 1e-9:
            self._progress.append((t, self.max_x_m))
        while len(self._progress) > 1 and self._progress[1][0] <= t - lim.stuck_window_s + 1e-9:
            self._progress.popleft()
        t_ref, x_ref = self._progress[0]
        return t - t_ref >= lim.stuck_window_s - 1e-9 and self.max_x_m - x_ref < lim.stuck_progress_m

    # -- log
    def trajectory_array(self) -> np.ndarray:
        """Rows of ``TRAJECTORY_COLUMNS``: one per simulated second plus the latest state."""
        rows = list(self._rows)
        if self.last is not None and (not rows or rows[-1][0] != self.last.t):
            rows.append(self.last.row())
        return np.array(rows, dtype=float).reshape(-1, len(TRAJECTORY_COLUMNS))


class RoverRun:
    """One rover in a model: its controller, its battery and its run tracker.

    Build it after the model and ``MjData`` exist and the rover sits at its spawn pose; it edits the model (the
    torque limit, see ``controller``) and writes the first drive command. ``battery_wh`` overrides the battery
    capacity of the design (for experiments and tests)."""

    def __init__(self, model, data, design, prefix: str, world=None, limits: RunLimits | None = None,
                 genes: ctl.ControllerGenes | None = None, params: ctl.ControllerParams | None = None,
                 battery_wh: float | None = None):
        import mujoco

        import track_terrain as tt

        mujoco.mj_forward(model, data)
        self.controller = ctl.RoverController(model, design, prefix, genes, params)
        self.battery = energy.Battery(design.battery_energy_wh if battery_wh is None else battery_wh)
        self.tracker = RunTracker(
            x0=float(data.xpos[self.controller.chassis_body][0]), limits=limits,
            section_at=None if world is None else world.section_at,
            finish_x=None if world is None else tt.finish_x(world.terrain),
        )
        self._dt = float(model.opt.timestep)
        self._steps_per_control = max(1, round(rp.CONTROL_DT_S / self._dt))
        self._steps = 0
        self.sim_time_s = 0.0
        self.max_abs_y_m = 0.0
        self.last_speeds = self.controller.apply(data)

    # -- forwarded state
    status = property(lambda self: self.tracker.status)
    reason = property(lambda self: self.tracker.reason)
    distance_m = property(lambda self: self.tracker.distance_m)
    max_distance_m = property(lambda self: self.tracker.max_distance_m)
    max_x_m = property(lambda self: self.tracker.max_x_m)
    alive_time_s = property(lambda self: self.tracker.alive_time_s)
    section_reached = property(lambda self: self.tracker.section_reached)
    running = property(lambda self: self.tracker.status == "running")

    def trajectory_array(self) -> np.ndarray:
        return self.tracker.trajectory_array()

    def sample(self, data) -> RunSample:
        return self._observe(data)[0]

    def _observe(self, data) -> tuple[RunSample, ctl.RoverState]:
        """The sample of this instant and the controller state it was built from."""
        c = self.controller
        state = c.read_state(data)
        vx, vy = data.qvel[c.free_dof: c.free_dof + 2].tolist()
        return RunSample(
            t=float(data.time), x=state.x_m, y=state.y_m, z=float(data.xpos[c.chassis_body][2]),
            quat=tuple(data.xquat[c.chassis_body].tolist()),
            roll_deg=state.roll_deg, pitch_deg=state.pitch_deg, yaw_rad=state.yaw_rad,
            speed_mps=math.hypot(vx, vy), soc=self.battery.soc, energy_dead=self.battery.dead,
        ), state

    def update(self, data) -> None:
        """Account for one physics step that has just been taken; on a control tick, observe and command."""
        if not self.running:
            return
        c = self.controller
        self.battery.step(energy.positive_mechanical_power_w(data.actuator_force[c.actuator_ids], data.qvel[c.wheel_dofs]), self._dt)
        self._steps += 1
        if self._steps % self._steps_per_control and not self.battery.dead:
            return
        s, state = self._observe(data)
        self.sim_time_s = s.t
        self.max_abs_y_m = max(self.max_abs_y_m, abs(s.y))
        self.tracker.update(s)
        if self.running:
            self.last_speeds = c.apply(data, state)
        else:
            c.stop(data)

    def summary(self) -> dict:
        b, t = self.battery, self.tracker
        return {
            "status": t.status, "reason": t.reason,
            "distance_m": t.distance_m, "max_distance_m": t.max_distance_m, "max_x_m": t.max_x_m,
            "alive_time_s": t.alive_time_s, "section_reached": t.section_reached,
            "drive_energy_wh": b.drive_energy_wh, "net_drain_wh": b.net_drain_wh,
            "usable_wh": b.usable_wh, "usable_remaining_wh": b.usable_remaining_wh, "soc": b.soc,
            "max_abs_y_m": self.max_abs_y_m, "torque_limit_nm": self.controller.torque_limit_nm,
        }


def simulate(model, data, runs: list[RoverRun], seconds: float) -> None:
    """Step the model for up to ``seconds`` of simulated time, or until no run is still running."""
    import mujoco

    while data.time < seconds - 1e-9 and any(r.running for r in runs):
        mujoco.mj_step(model, data)
        for r in runs:
            r.update(data)

"""The rover's driving controller: a fixed, explainable heading hold with three evolved parameters.

The controller is not learned. Its law is the same for every rover; only three genes change how it behaves
(``target_speed_mps``, ``torque_limit_fraction``, ``slope_brake``). The rover has no steering joints, so it is
skid-steered: the left and the right wheels get different rim speeds.

The law (``command``), for a rover at lateral position ``y`` heading ``yaw`` (0 = along the +x lane axis):

1. Cruise speed  ``v = v0 * (1 - brake * clip(tilt / TILT_REF, 0, 1))``, floored at ``MIN_SPEED_FRACTION * v0`` so the
   rover never asks to stop. ``v0`` is the target speed gene, ``brake`` the slope-brake gene and ``tilt`` the
   combined pitch and roll of the chassis.
2. Desired heading  ``psi* = clip(-LANE_KP * y, +-LANE_MAX_HEADING)``: a gentle turn back towards the lane centre
   when the rover has drifted (the cross slope pushes it sideways).
3. Turn  ``dv = clip(KP * wrap(psi* - yaw) - KD * yaw_rate, +-MAX_TURN_DIFF)`` (a PD law on the heading error).
4. Wheels  ``left = v - dv``, ``right = v + dv`` (rim speeds, m/s), clipped to the actuator range.

``torque_limit_fraction`` is applied to the model, not in the law: ``RoverController`` narrows the force range of
the six drive actuators to ``fraction * motor peak torque`` when it is built. That makes it a physical limit
(the same on every call, impossible to exceed by a bug in the law) and it is what the energy model measures.

The pure core (``command`` and friends) needs no MuJoCo; ``RoverController`` is the thin adapter that reads a
rover's state from ``MjData`` and writes its ``ctrl``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import rover_params as rp


@dataclass(frozen=True)
class ControllerGenes:
    """The three evolved controller genes, in physical units."""

    target_speed_mps: float
    torque_limit_fraction: float
    slope_brake: float

    @classmethod
    def from_values(cls, values: dict[str, float]) -> "ControllerGenes":
        """From decoded gene values (``genome_spec.decode`` or ``RoverDesign.genes``)."""
        return cls(values["target_speed_mps"], values["torque_limit_fraction"], values["slope_brake"])


@dataclass(frozen=True)
class ControllerParams:
    """Fixed gains and limits of the law (defaults in ``rover_params``); tests and experiments may override them."""

    heading_kp: float = rp.HEADING_KP_MPS_PER_RAD
    heading_kd: float = rp.HEADING_KD_MPS_PER_RAD_S
    max_turn_diff_mps: float = rp.MAX_TURN_DIFF_MPS
    lane_kp: float = rp.LANE_KP_RAD_PER_M
    lane_max_heading_rad: float = rp.LANE_MAX_HEADING_RAD
    tilt_ref_deg: float = rp.TILT_REF_DEG
    min_speed_fraction: float = rp.MIN_SPEED_FRACTION
    max_wheel_speed_mps: float = rp.MAX_WHEEL_SPEED_MPS


@dataclass(frozen=True)
class RoverState:
    """What the law sees: position along the lane, heading, yaw rate and chassis tilt (degrees, + pitch = nose up)."""

    x_m: float
    y_m: float
    yaw_rad: float
    yaw_rate_rad_s: float
    pitch_deg: float
    roll_deg: float


class WheelSpeeds(NamedTuple):
    """Commanded rim speed of the left and of the right wheels, in m/s (positive = forward)."""

    left_mps: float
    right_mps: float


def wrap_angle(angle: float) -> float:
    """``angle`` wrapped to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_target_rad(y_m: float, params: ControllerParams) -> float:
    """Desired heading: along +x, bent towards the lane centre in proportion to the offset, within a bound."""
    limit = params.lane_max_heading_rad
    return min(max(-params.lane_kp * y_m, -limit), limit)  # == np.clip, without numpy's per-call overhead


def cruise_speed_mps(genes: ControllerGenes, pitch_deg: float, roll_deg: float, params: ControllerParams) -> float:
    """Target speed reduced by the slope brake with the combined tilt, never below the floor."""
    tilt = math.hypot(pitch_deg, roll_deg)
    v0 = genes.target_speed_mps
    v = v0 * (1.0 - genes.slope_brake * min(tilt / params.tilt_ref_deg, 1.0))
    return max(v, params.min_speed_fraction * v0)


def command(state: RoverState, genes: ControllerGenes, params: ControllerParams) -> WheelSpeeds:
    """Left / right wheel rim speeds for ``state`` (see the module docstring for the law)."""
    v = cruise_speed_mps(genes, state.pitch_deg, state.roll_deg, params)
    error = wrap_angle(heading_target_rad(state.y_m, params) - state.yaw_rad)
    bound = params.max_turn_diff_mps
    turn = min(max(params.heading_kp * error - params.heading_kd * state.yaw_rate_rad_s, -bound), bound)  # > 0 turns left
    limit = params.max_wheel_speed_mps
    return WheelSpeeds(min(max(v - turn, -limit), limit), min(max(v + turn, -limit), limit))


def torque_limit_nm(motor_peak_torque_nm: float, genes: ControllerGenes) -> float:
    """Per-wheel torque the controller may use: the gene fraction of the motor's peak torque."""
    return genes.torque_limit_fraction * motor_peak_torque_nm


class RoverController:
    """Adapter: reads one rover's state from ``MjData`` and writes its six drive commands.

    ``prefix`` is the rover's name prefix in the model (``rover.RoverPlacement.prefix``). Building it limits
    the force range of this rover's actuators to ``torque_limit_fraction * peak torque`` (it edits the model)."""

    def __init__(self, model, design, prefix: str, genes: ControllerGenes | None = None, params: ControllerParams | None = None):
        import mujoco

        import rover

        self.genes = genes or ControllerGenes.from_values(design.genes)
        self.params = params or ControllerParams()
        self.wheel_radius_m = design.wheel_radius_m
        self.torque_limit_nm = torque_limit_nm(design.motor_peak_torque_nm, self.genes)

        def actuator(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        names = rover.actuator_names(prefix)
        self.actuator_ids = np.array([actuator(n) for n in names])  # in ``rover.WHEEL_IDS`` order
        self.left_ids = np.array([actuator(n) for n, w in zip(names, rover.WHEEL_IDS) if w.endswith("l")])
        self.right_ids = np.array([actuator(n) for n, w in zip(names, rover.WHEEL_IDS) if w.endswith("r")])
        if (self.actuator_ids < 0).any():
            raise ValueError(f"no rover with prefix {prefix!r} in the model")
        self.wheel_dofs = np.array([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in rover.wheel_joint_names(prefix)
        ])
        self.chassis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}chassis")
        self.free_dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}free")])
        self.limit_torque(model)

    def limit_torque(self, model) -> None:
        """Narrow the force range of this rover's actuators to the torque limit. Done for the model given to the
        constructor; call it again on a model rebuilt for the same rover (same ids), e.g. a headless rock window."""
        model.actuator_forcelimited[self.actuator_ids] = 1
        model.actuator_forcerange[self.actuator_ids] = (-self.torque_limit_nm, self.torque_limit_nm)

    def read_state(self, data) -> RoverState:
        """Pose, yaw rate and tilt of the chassis (call after the kinematics are up to date)."""
        pos = data.xpos[self.chassis_body]
        rot = data.xmat[self.chassis_body].reshape(3, 3)
        omega_world = rot @ data.qvel[self.free_dof + 3: self.free_dof + 6]  # free-joint angular velocity is body-local
        return RoverState(
            x_m=float(pos[0]), y_m=float(pos[1]),
            yaw_rad=math.atan2(rot[1, 0], rot[0, 0]), yaw_rate_rad_s=float(omega_world[2]),
            pitch_deg=math.degrees(math.asin(float(np.clip(rot[2, 0], -1.0, 1.0)))),
            roll_deg=math.degrees(math.atan2(rot[2, 1], rot[2, 2])),
        )

    def apply(self, data, state: RoverState | None = None) -> WheelSpeeds:
        """Write the drive commands (rad/s at the wheel) for the current state; returns the rim speeds.
        ``state`` is the already-read ``read_state(data)`` of this same instant, when the caller has it."""
        speeds = command(self.read_state(data) if state is None else state, self.genes, self.params)
        data.ctrl[self.left_ids] = speeds.left_mps / self.wheel_radius_m
        data.ctrl[self.right_ids] = speeds.right_mps / self.wheel_radius_m
        return speeds

    def stop(self, data) -> None:
        """Zero every drive command of this rover (a dead rover rolls to a halt)."""
        data.ctrl[self.actuator_ids] = 0.0

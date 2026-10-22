"""
controller.py: el controlador de manejo del rover, una regla fija con tres parámetros evolucionados.

Para qué existe
    Cada rover necesita un conductor. El controlador no se aprende: es la misma regla, simple y
    explicable, para todos los rovers, y solo tres genes cambian su comportamiento: la velocidad
    objetivo, la fracción del torque que puede usar y el freno en pendientes.

Qué contiene
    - command: calcula la velocidad de las ruedas izquierdas y derechas para el estado actual.
    - cruise_speed_mps: la velocidad de crucero, reducida por el freno en pendientes.
    - heading_target_rad: el rumbo deseado, que corrige de a poco el desvío respecto del centro del carril.
    - RoverController: lee el estado del rover en MuJoCo, limita el torque de sus motores y les envía
      las órdenes.

Cómo funciona
    El rover no tiene ruedas direccionales: dobla dando velocidades distintas a cada lado. Un control
    PD sobre el error de rumbo decide esa diferencia. El límite de torque se aplica a los motores del
    modelo físico, no a la regla, así nunca se puede superar por un error de cálculo.

Dónde encaja
    Etapa: simulación. Usa rover_params (y rover, para encontrar los motores del rover en el modelo).
    Lo usa rover_run.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import rover_params as rp


@dataclass(frozen=True)
class ControllerGenes:
    target_speed_mps: float
    torque_limit_fraction: float
    slope_brake: float

    @classmethod
    def from_values(cls, values: dict[str, float]) -> "ControllerGenes":
        return cls(values["target_speed_mps"], values["torque_limit_fraction"], values["slope_brake"])


@dataclass(frozen=True)
class ControllerParams:
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
    x_m: float
    y_m: float
    yaw_rad: float
    yaw_rate_rad_s: float
    pitch_deg: float
    roll_deg: float


class WheelSpeeds(NamedTuple):
    left_mps: float
    right_mps: float


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_target_rad(y_m: float, params: ControllerParams) -> float:
    limit = params.lane_max_heading_rad
    return min(max(-params.lane_kp * y_m, -limit), limit)


def cruise_speed_mps(genes: ControllerGenes, pitch_deg: float, roll_deg: float, params: ControllerParams) -> float:
    tilt = math.hypot(pitch_deg, roll_deg)
    v0 = genes.target_speed_mps
    v = v0 * (1.0 - genes.slope_brake * min(tilt / params.tilt_ref_deg, 1.0))
    return max(v, params.min_speed_fraction * v0)


def command(state: RoverState, genes: ControllerGenes, params: ControllerParams) -> WheelSpeeds:
    v = cruise_speed_mps(genes, state.pitch_deg, state.roll_deg, params)
    error = wrap_angle(heading_target_rad(state.y_m, params) - state.yaw_rad)
    bound = params.max_turn_diff_mps
    turn = min(max(params.heading_kp * error - params.heading_kd * state.yaw_rate_rad_s, -bound), bound)
    limit = params.max_wheel_speed_mps
    return WheelSpeeds(min(max(v - turn, -limit), limit), min(max(v + turn, -limit), limit))


def torque_limit_nm(motor_peak_torque_nm: float, genes: ControllerGenes) -> float:
    return genes.torque_limit_fraction * motor_peak_torque_nm


class RoverController:
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
        self.actuator_ids = np.array([actuator(n) for n in names])
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
        model.actuator_forcelimited[self.actuator_ids] = 1
        model.actuator_forcerange[self.actuator_ids] = (-self.torque_limit_nm, self.torque_limit_nm)

    def read_state(self, data) -> RoverState:
        pos = data.xpos[self.chassis_body]
        rot = data.xmat[self.chassis_body].reshape(3, 3)
        omega_world = rot @ data.qvel[self.free_dof + 3: self.free_dof + 6]
        return RoverState(
            x_m=float(pos[0]), y_m=float(pos[1]),
            yaw_rad=math.atan2(rot[1, 0], rot[0, 0]), yaw_rate_rad_s=float(omega_world[2]),
            pitch_deg=math.degrees(math.asin(float(np.clip(rot[2, 0], -1.0, 1.0)))),
            roll_deg=math.degrees(math.atan2(rot[2, 1], rot[2, 2])),
        )

    def apply(self, data, state: RoverState | None = None) -> WheelSpeeds:
        speeds = command(self.read_state(data) if state is None else state, self.genes, self.params)
        data.ctrl[self.left_ids] = speeds.left_mps / self.wheel_radius_m
        data.ctrl[self.right_ids] = speeds.right_mps / self.wheel_radius_m
        return speeds

    def stop(self, data) -> None:
        data.ctrl[self.actuator_ids] = 0.0

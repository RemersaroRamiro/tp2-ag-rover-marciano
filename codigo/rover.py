"""
rover.py: convierte un genoma en un rover concreto: geometría, masas, restricciones y modelo MuJoCo.

Para qué existe
    El AG trabaja con 15 números, pero para simular hace falta un rover físico. Este archivo hace esa
    traducción y garantiza que cualquier genoma se pueda construir.

Qué contiene
    - design_from_genome: arma el diseño rocker-bogie de seis ruedas y lo repara si hace falta.
    - RoverDesign: el diseño resultante, con sus masas, su centro de masa y las reparaciones aplicadas.
    - check_constraints: revisa las restricciones de la misión (masa total, vuelco estático, volumen
      plegado y despeje del suelo) y mide cuánto se excede cada una, en forma relativa.
    - rover_mjcf: genera los cuerpos, los motores y las uniones del rover para MuJoCo.
    - spawn_pose: ubica el rover en la salida, apoyado sobre el terreno y las rocas.

Cómo funciona
    La geometría es válida por construcción: si dos ruedas se superponen o un brazo es demasiado corto,
    el decodificador alarga piezas en lugar de descartar el genoma. Las masas salen de los genes
    (ruedas, motores, batería, chasis y brazos) más una carga útil fija.

Dónde encaja
    Etapa: simulación. Usa genome_spec y rover_params.
    Lo usan evaluate, headless y controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

import genome_spec
import rover_params as rp

WHEEL_IDS = ("fl", "fr", "ml", "mr", "rl", "rr")


@dataclass(frozen=True)
class MassProps:
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]


def _pack(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    return (
        float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]),
        float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[1, 2]),
    )


def _part(mass: float, com, inertia: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    return mass, np.asarray(com, dtype=float), np.zeros((3, 3)) if inertia is None else inertia


def _box_part(mass: float, center, dims) -> tuple[float, np.ndarray, np.ndarray]:
    dx, dy, dz = dims
    return _part(mass, center, np.diag([dy * dy + dz * dz, dx * dx + dz * dz, dx * dx + dy * dy]) * mass / 12.0)


def _rod_part(mass: float, start, end, radius: float) -> tuple[float, np.ndarray, np.ndarray]:
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))
    u = (end - start) / length
    axial = mass * radius * radius / 2.0
    transverse = mass * (length * length / 12.0 + radius * radius / 4.0)
    return _part(mass, (start + end) / 2, transverse * (np.eye(3) - np.outer(u, u)) + axial * np.outer(u, u))


def _wheel_shell_part(mass: float, center, radius: float, width: float) -> tuple[float, np.ndarray, np.ndarray]:
    transverse = mass * (radius * radius / 2.0 + width * width / 12.0)
    return _part(mass, center, np.diag([transverse, mass * radius * radius, transverse]))


def _compose(parts: Sequence[tuple[float, np.ndarray, np.ndarray]]) -> MassProps:
    mass = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in parts:
        d = c - com
        inertia += i + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return MassProps(float(mass), tuple(float(v) for v in com), _pack(inertia))


@dataclass(frozen=True)
class RoverDesign:
    genes: dict[str, float]
    repairs: tuple[str, ...]
    wheel_radius_m: float
    wheel_width_m: float
    wheel_centers: dict[str, tuple[float, float, float]]
    chassis_length_m: float
    chassis_width_m: float
    chassis_height_m: float
    chassis_bottom_z_m: float
    track_width_m: float
    rocker_pivot_l: tuple[float, float, float]
    bogie_pivot_l: tuple[float, float, float]
    front_arm_m: float
    rear_arm_m: float
    bogie_length_m: float
    motor_peak_torque_nm: float
    battery_energy_wh: float
    wheel_mass_kg: float
    motor_mass_kg: float
    battery_mass_kg: float
    chassis_structure_mass_kg: float
    arms_mass_kg: float
    bodies: dict[str, MassProps]
    total_mass_kg: float
    com: tuple[float, float, float]
    ground_clearance_m: float
    wheelbase_m: float
    tipover_longitudinal_deg: float
    tipover_lateral_deg: float
    envelope_m: tuple[float, float, float]


def static_tipover_angles(com, x_front: float, x_rear: float, y_half: float) -> tuple[float, float]:
    x, y, h = com
    longitudinal = math.degrees(math.atan2(min(x_front - x, x - x_rear), h))
    lateral = math.degrees(math.atan2(y_half - abs(y), h))
    return longitudinal, lateral


def design_from_genome(genome) -> RoverDesign:
    v = genome_spec.decode(genome)
    radius, wheel_w = v["wheel_diameter_m"] / 2, v["wheel_width_m"]
    track, length, height = v["track_width_m"], v["chassis_length_m"], v["chassis_height_m"]
    hp = v["rocker_pivot_height_m"]
    hb = rp.BOGIE_PIVOT_HEIGHT_M
    repairs: list[str] = []

    bogie_len = v["bogie_length_m"]
    if bogie_len < 2 * radius + rp.MIN_WHEEL_GAP_M:
        bogie_len = 2 * radius + rp.MIN_WHEEL_GAP_M
        repairs.append("bogie_length_m")

    drop_rear = max(hp - hb, 0.0)
    rear_arm = v["rocker_rear_arm_m"]
    if rear_arm < math.hypot(drop_rear, rp.MIN_ARM_REACH_M):
        rear_arm = math.hypot(drop_rear, rp.MIN_ARM_REACH_M)
        repairs.append("rocker_rear_arm_m")
    reach_rear = math.sqrt(max(rear_arm**2 - drop_rear**2, 0.0))

    mid_ahead_of_pivot = bogie_len / 2 - reach_rear
    min_reach_front = max(rp.MIN_ARM_REACH_M, mid_ahead_of_pivot + 2 * radius + rp.MIN_WHEEL_GAP_M)
    front_arm = v["rocker_front_arm_m"]
    if front_arm < math.hypot(hp, min_reach_front):
        front_arm = math.hypot(hp, min_reach_front)
        repairs.append("rocker_front_arm_m")
    reach_front = math.sqrt(max(front_arm**2 - hp**2, 0.0))

    pivot_x = v["rocker_pivot_offset"] * length
    axle_z, pivot_z, bogie_z = radius, radius + hp, radius + hb
    half_track = track / 2
    bogie_x = pivot_x - reach_rear
    centers = {}
    for side, sign in (("l", 1.0), ("r", -1.0)):
        centers["f" + side] = (pivot_x + reach_front, sign * half_track, axle_z)
        centers["m" + side] = (bogie_x + bogie_len / 2, sign * half_track, axle_z)
        centers["r" + side] = (bogie_x - bogie_len / 2, sign * half_track, axle_z)

    chassis_width = track - wheel_w - 2 * rp.CHASSIS_SIDE_GAP_M
    bottom = max(pivot_z - height / 2, axle_z + rp.MIN_CHASSIS_BOTTOM_ABOVE_AXLE_M)

    wheel_mass = (
        (2 * math.pi * radius * wheel_w + 2 * math.pi * radius**2 * rp.WHEEL_SIDE_AREA_FRACTION)
        * rp.WHEEL_SHELL_THICKNESS_M * rp.ALUMINUM_DENSITY_KG_M3
    )
    motor_mass = v["motor_peak_torque_nm"] / rp.MOTOR_SPECIFIC_TORQUE_NM_PER_KG
    battery_mass = v["battery_energy_wh"] / rp.BATTERY_SPECIFIC_ENERGY_WH_PER_KG
    structure_mass = (
        2 * (length * chassis_width + length * height + chassis_width * height) * rp.CHASSIS_AREAL_DENSITY_KG_M2
    )
    tube = rp.ARM_TUBE_RADIUS_M
    bogie_arm = math.hypot(bogie_len / 2, hb)
    arms_mass = 2 * (front_arm + rear_arm + 2 * bogie_arm) * rp.ARM_LINEAR_DENSITY_KG_M

    lam = rp.ARM_LINEAR_DENSITY_KG_M
    bodies: dict[str, MassProps] = {}
    chassis_center = (0.0, 0.0, bottom + height / 2)
    loads = rp.PAYLOAD_MASS_KG + battery_mass
    load_x = float(np.clip(pivot_x * (structure_mass + loads) / loads, -rp.MAX_LOAD_OFFSET_FRACTION * length, rp.MAX_LOAD_OFFSET_FRACTION * length))
    bodies["chassis"] = _compose([
        _box_part(structure_mass, chassis_center, (length, chassis_width, height)),
        _box_part(
            rp.PAYLOAD_MASS_KG, (load_x, 0.0, bottom + rp.PAYLOAD_HEIGHT_FRACTION * height),
            (length * rp.PAYLOAD_SIZE_FRACTION, chassis_width * rp.PAYLOAD_SIZE_FRACTION, height * rp.PAYLOAD_SIZE_FRACTION),
        ),
        _box_part(battery_mass, (load_x, 0.0, bottom + rp.BATTERY_HEIGHT_FRACTION * height), (0.5 * length, 0.5 * chassis_width, 0.3 * height)),
    ])
    for side, sign in (("l", 1.0), ("r", -1.0)):
        pivot = np.array([pivot_x, sign * half_track, pivot_z])
        bogie_pivot = np.array([bogie_x, sign * half_track, bogie_z])
        front = np.array(centers["f" + side])
        mid, rear = np.array(centers["m" + side]), np.array(centers["r" + side])
        bodies["rocker_" + side] = _compose([
            _rod_part(lam * front_arm, pivot, front, tube), _rod_part(lam * rear_arm, pivot, bogie_pivot, tube),
        ])
        bodies["bogie_" + side] = _compose([
            _rod_part(lam * bogie_arm, bogie_pivot, mid, tube), _rod_part(lam * bogie_arm, bogie_pivot, rear, tube),
        ])
        for w in "fmr":
            center = centers[w + side]
            bodies[f"wheel_{w}{side}"] = _compose([
                _wheel_shell_part(wheel_mass, center, radius, wheel_w), _part(motor_mass, center),
            ])
    total = sum(b.mass for b in bodies.values())
    com = tuple(float(sum(b.mass * b.com[i] for b in bodies.values()) / total) for i in range(3))

    x_front, x_rear = centers["fl"][0], centers["rl"][0]
    tip_long, tip_lat = static_tipover_angles(com, x_front, x_rear, half_track + wheel_w / 2)
    x_min, x_max = min(x_rear - radius, -length / 2), max(x_front + radius, length / 2)
    envelope = (x_max - x_min, track + wheel_w, max(2 * radius, bottom + height))

    return RoverDesign(
        genes=v, repairs=tuple(repairs),
        wheel_radius_m=radius, wheel_width_m=wheel_w, wheel_centers=centers,
        chassis_length_m=length, chassis_width_m=chassis_width, chassis_height_m=height, chassis_bottom_z_m=bottom,
        track_width_m=track,
        rocker_pivot_l=(pivot_x, half_track, pivot_z), bogie_pivot_l=(bogie_x, half_track, bogie_z),
        front_arm_m=front_arm, rear_arm_m=rear_arm, bogie_length_m=bogie_len,
        motor_peak_torque_nm=v["motor_peak_torque_nm"], battery_energy_wh=v["battery_energy_wh"],
        wheel_mass_kg=wheel_mass, motor_mass_kg=motor_mass, battery_mass_kg=battery_mass,
        chassis_structure_mass_kg=structure_mass, arms_mass_kg=arms_mass,
        bodies=bodies, total_mass_kg=total, com=com,
        ground_clearance_m=bottom, wheelbase_m=x_front - x_rear,
        tipover_longitudinal_deg=tip_long, tipover_lateral_deg=tip_lat, envelope_m=envelope,
    )


@dataclass(frozen=True)
class Violation:
    code: str
    value: float
    limit: float
    magnitude: float
    detail: str = ""


def check_constraints(design: RoverDesign) -> list[Violation]:
    found: list[Violation] = []
    if design.total_mass_kg > rp.MAX_TOTAL_MASS_KG:
        found.append(Violation("mass_over_limit", design.total_mass_kg, rp.MAX_TOTAL_MASS_KG,
                               (design.total_mass_kg - rp.MAX_TOTAL_MASS_KG) / rp.MAX_TOTAL_MASS_KG))
    for axis, angle in (("longitudinal", design.tipover_longitudinal_deg), ("lateral", design.tipover_lateral_deg)):
        if angle < rp.MIN_TIPOVER_DEG:
            found.append(Violation("tipover_below_min", angle, rp.MIN_TIPOVER_DEG, (rp.MIN_TIPOVER_DEG - angle) / rp.MIN_TIPOVER_DEG, axis))
    for name, size, limit in zip(("length", "width", "height"), design.envelope_m, rp.MAX_ENVELOPE_M):
        if size > limit:
            found.append(Violation("envelope_exceeds", size, limit, (size - limit) / limit, name))
    if design.ground_clearance_m < rp.MIN_GROUND_CLEARANCE_M:
        found.append(Violation("clearance_below_min", design.ground_clearance_m, rp.MIN_GROUND_CLEARANCE_M,
                               (rp.MIN_GROUND_CLEARANCE_M - design.ground_clearance_m) / rp.MIN_GROUND_CLEARANCE_M))
    return found


class RoverXml(NamedTuple):
    body: str
    actuator: str
    equality: str


def wheel_joint_names(prefix: str) -> list[str]:
    return [f"{prefix}wheel_{w}" for w in WHEEL_IDS]


def actuator_names(prefix: str) -> list[str]:
    return [f"{prefix}drive_{w}" for w in WHEEL_IDS]


def _f(*values: float) -> str:
    return " ".join(f"{float(x):.6f}" for x in values)


def _inertial(props: MassProps, origin) -> str:
    com = np.asarray(props.com) - np.asarray(origin)
    return f'<inertial pos="{_f(*com)}" mass="{props.mass:.6f}" fullinertia="{_f(*props.inertia)}"/>'


_BODY = 'rgba="0.86 0.85 0.82 1"'
_TYRE = 'rgba="0.17 0.17 0.18 1"'
_COLLIDE = f'contype="{rp.ROVER_CONTYPE}" conaffinity="{rp.ROVER_CONAFFINITY}"'


def _wheel_body(p: str, wheel: str, offset, d: RoverDesign, indent: int) -> str:
    r, w = d.wheel_radius_m, d.wheel_width_m
    pad = " " * indent
    return (
        f'{pad}<body name="{p}wheel_{wheel}" pos="{_f(*offset)}">\n'
        f'{pad}  <joint name="{p}wheel_{wheel}" type="hinge" axis="0 1 0" damping="{rp.WHEEL_DAMPING_NMS}" '
        f'armature="{rp.WHEEL_ARMATURE_KG_M2}"/>\n'
        f"{pad}  {_inertial(d.bodies['wheel_' + wheel], d.wheel_centers[wheel])}\n"
        f'{pad}  <geom name="{p}wheel_{wheel}" type="cylinder" group="{rp.ROVER_GROUP}" {_COLLIDE} {_TYRE} '
        f'size="{_f(r, w / 2)}" euler="90 0 0" friction="{_f(*rp.WHEEL_FRICTION)}"/>\n'
        f"{pad}</body>"
    )


def rover_mjcf(design: RoverDesign, name_prefix: str, pos, yaw: float = 0.0) -> RoverXml:
    p, d = name_prefix, design
    r = d.wheel_radius_m
    half_track = d.track_width_m / 2
    quat = _f(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    origin = np.zeros(3)
    chassis_center = np.array([0.0, 0.0, d.chassis_bottom_z_m + d.chassis_height_m / 2])
    length = d.chassis_length_m
    lines = [
        f'<body name="{p}chassis" pos="{_f(*pos)}" quat="{quat}">',
        f'  <freejoint name="{p}free"/>',
        f"  {_inertial(d.bodies['chassis'], origin)}",
        f'  <geom name="{p}chassis" type="box" group="{rp.ROVER_GROUP}" {_COLLIDE} {_BODY} '
        f'size="{_f(length / 2, d.chassis_width_m / 2, d.chassis_height_m / 2)}" pos="{_f(*chassis_center)}"/>',
    ]
    for side, sign in (("l", 1.0), ("r", -1.0)):
        pivot = np.array([d.rocker_pivot_l[0], sign * half_track, d.rocker_pivot_l[2]])
        bogie = np.array([d.bogie_pivot_l[0], sign * half_track, d.bogie_pivot_l[2]])
        centers = {k: np.array(d.wheel_centers[k + side]) for k in "fmr"}
        lines += [
            f'  <body name="{p}rocker_{side}" pos="{_f(*pivot)}">',
            f'    <joint name="{p}rocker_{side}" type="hinge" axis="0 1 0" limited="true" '
            f'range="{_f(-rp.ROCKER_RANGE_DEG, rp.ROCKER_RANGE_DEG)}" damping="{rp.ROCKER_DAMPING_NMS}"/>',
            f"    {_inertial(d.bodies['rocker_' + side], pivot)}",
            _wheel_body(p, "f" + side, centers["f"] - pivot, d, indent=4),
            f'    <body name="{p}bogie_{side}" pos="{_f(*(bogie - pivot))}">',
            f'      <joint name="{p}bogie_{side}" type="hinge" axis="0 1 0" limited="true" '
            f'range="{_f(-rp.BOGIE_RANGE_DEG, rp.BOGIE_RANGE_DEG)}" damping="{rp.BOGIE_DAMPING_NMS}"/>',
            f"      {_inertial(d.bodies['bogie_' + side], bogie)}",
            _wheel_body(p, "m" + side, centers["m"] - bogie, d, indent=6),
            _wheel_body(p, "r" + side, centers["r"] - bogie, d, indent=6),
            "    </body>",
            "  </body>",
        ]
    lines.append("</body>")
    kv = d.motor_peak_torque_nm / rp.DRIVE_SATURATION_RAD_S
    omega_max = rp.MAX_WHEEL_SPEED_MPS / r
    actuators = [
        f'<velocity name="{p}drive_{wheel}" joint="{p}wheel_{wheel}" kv="{kv:.6f}" '
        f'ctrlrange="{_f(-omega_max, omega_max)}" forcerange="{_f(-d.motor_peak_torque_nm, d.motor_peak_torque_nm)}"/>'
        for wheel in WHEEL_IDS
    ]
    equality = f'<joint name="{p}differential" joint1="{p}rocker_l" joint2="{p}rocker_r" polycoef="0 -1 0 0 0"/>'
    return RoverXml(
        "\n".join("    " + line for line in lines), "\n".join("    " + a for a in actuators), "    " + equality
    )


def spawn_pose(world, design: RoverDesign, x: float = rp.SPAWN_X_M, y: float = 0.0) -> tuple[tuple[float, float, float], float]:
    r, half_w = design.wheel_radius_m, design.wheel_width_m / 2
    top = -np.inf
    for cx, cy, _ in design.wheel_centers.values():
        xs, ys = np.meshgrid(np.linspace(x + cx - r, x + cx + r, 9), np.linspace(y + cy - half_w, y + cy + half_w, 5))
        top = max(top, float(np.max(world.terrain.sample(xs, ys))))
        for rock in world.rocks:
            reach = max(rock.sx, rock.sy)
            if abs(rock.x - (x + cx)) < r + reach and abs(rock.y - (y + cy)) < half_w + reach:
                top = max(top, rock.z + rock.sz)
    return (x, y, top + rp.SPAWN_CLEARANCE_M), 0.0

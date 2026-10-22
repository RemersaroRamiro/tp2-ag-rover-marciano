"""
headless.py: el modelo físico de la pista sin nada visual, para evaluar rovers más rápido.

Para qué existe
    La escena completa (scene) trae texturas, luces, cámaras y terreno lejano que no afectan la física.
    Este archivo arma, a partir del mismo mundo generado, un modelo que conserva solo lo que puede
    tocar a un rover, y así cada simulación es mucho más liviana.

Qué contiene
    - generate_headless_world: el terreno y las rocas de una semilla, recortados a la zona alcanzable.
    - HeadlessWorld: todo lo que una evaluación necesita de una semilla.
    - build_model: compila esa zona con un rover en su posición de salida.
    - RockWindows y transfer_state: cargan solo las rocas cercanas al rover y cambian de modelo a mitad
      de la corrida sin perder el estado de la simulación.

Cómo funciona
    Se conservan intactos el terreno de colisión, su fricción y las rocas alcanzables. Se quita lo
    visual y lo inalcanzable: la corrida termina si el centro del rover se aleja más de 10 m del eje
    del carril, y ninguna parte del rover está a más de ROVER_REACH_M de su centro.

Dónde encaja
    Etapa: simulación. Usa track_terrain, track_rocks, track_sections, mars_terrain, scene, rover y
    mars_params. Lo usa evaluate.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

import mars_params as mp
import mars_terrain as mt
import rover
import scene
import track_rocks as tr
import track_sections as ts
import track_terrain as tt

ROVER_REACH_M = mp.ROVER_REACH_M
CROP_MARGIN_M = 2.0
LANE_HALF_M = mp.LANE_WIDTH_M / 2.0
LATERAL_CROP_M = LANE_HALF_M + ROVER_REACH_M + CROP_MARGIN_M
X_BEHIND_M = ROVER_REACH_M + CROP_MARGIN_M + 2.0
X_AHEAD_M = ROVER_REACH_M + CROP_MARGIN_M

HFIELD_BASE_M = 2.0
HYSTERESIS_M = 2.0


def design_reach_bound_m(design: rover.RoverDesign) -> float:
    top = design.chassis_bottom_z_m + design.chassis_height_m
    chassis = math.sqrt((design.chassis_length_m / 2) ** 2 + (design.chassis_width_m / 2) ** 2 + top**2)
    bogie_arm = math.hypot(design.bogie_length_m / 2, design.bogie_pivot_l[2] - design.wheel_radius_m)
    chain = max(design.front_arm_m, design.rear_arm_m + bogie_arm)
    pivot = math.dist(design.rocker_pivot_l, (0.0, 0.0, 0.0))
    wheel = math.hypot(design.wheel_radius_m, design.wheel_width_m / 2)
    return max(chassis, pivot + chain + wheel)


@lru_cache(maxsize=1)
def load_strip() -> tt.Strip:
    return tt.load_strip()


def lane_length_m(strip: tt.Strip | None = None) -> float:
    strip = strip or load_strip()
    return strip.length_m - mp.FINISH_MARGIN_M - mp.START_X_M


@dataclass(frozen=True)
class HeadlessTile:
    name: str
    asset_xml: str
    geom_xml: str
    data: np.ndarray


@dataclass(frozen=True)
class HeadlessWorld:
    seed: int
    terrain: tt.TrackTerrain
    rocks: list[mt.Rock]
    tiles: tuple[HeadlessTile, ...]
    head_xml: str
    rock_lines: tuple[str, ...]
    rock_x_lo: np.ndarray
    rock_x_hi: np.ndarray

    @property
    def sections(self) -> tuple[ts.Section, ...]:
        return self.terrain.sections

    @property
    def finish_x(self) -> float:
        return tt.finish_x(self.terrain)

    def section_at(self, x: float) -> ts.Section:
        return ts.section_at(x, self.sections)

    def rocks_between(self, x_lo: float, x_hi: float) -> list[mt.Rock]:
        keep = (self.rock_x_hi >= x_lo) & (self.rock_x_lo <= x_hi)
        return [r for r, k in zip(self.rocks, keep) if k]

    def rock_xml(self, rock_range: tuple[float, float] | None = None) -> str:
        if rock_range is None:
            return "\n".join(self.rock_lines)
        keep = (self.rock_x_hi >= rock_range[0]) & (self.rock_x_lo <= rock_range[1])
        return "\n".join(line for line, k in zip(self.rock_lines, keep) if k)


@dataclass(frozen=True)
class RockWindows:
    width_m: float
    origin_m: float = -X_BEHIND_M

    def index(self, x: float) -> int:
        return math.floor((x - self.origin_m) / self.width_m)

    def zone(self, k: int) -> tuple[float, float]:
        return self.origin_m + k * self.width_m, self.origin_m + (k + 1) * self.width_m

    def needs_switch(self, k: int, x: float) -> bool:
        lo, hi = self.zone(k)
        return x < lo - HYSTERESIS_M or x >= hi + HYSTERESIS_M

    def rock_range(self, k: int) -> tuple[float, float]:
        lo, hi = self.zone(k)
        margin = ROVER_REACH_M + CROP_MARGIN_M + HYSTERESIS_M
        return lo - margin, hi + margin


def _rock_in_window(r: mt.Rock, x_lo: float, x_hi: float) -> bool:
    extent = max(r.sx, r.sy, r.sz)
    return abs(r.y) - extent <= LATERAL_CROP_M and x_lo <= r.x + extent and r.x - extent <= x_hi


def _rock_lines(rocks: list[mt.Rock]) -> list[str]:
    friction = f"{mp.ROCK_FRICTION} {mp.SPIN_FRICTION} {mp.ROLL_FRICTION}"
    lines = []
    for r in rocks:
        q = scene._rock_quat(r)
        lines.append(
            f'<geom type="ellipsoid" size="{r.sx:.4f} {r.sy:.4f} {r.sz:.4f}" pos="{r.x:.4f} {r.y:.4f} {r.z:.4f}" '
            f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" friction="{friction}" priority="{scene.GROUND_PRIORITY}"/>'
        )
    return lines


def _window_tiles(terrain: tt.TrackTerrain, x_lo: float, x_hi: float) -> list[HeadlessTile]:
    n_rows, n_cols = terrain.heights.shape
    cell_x = (terrain.x_max - terrain.x_min) / (n_cols - 1)
    cell_y = (terrain.y_max - terrain.y_min) / (n_rows - 1)
    row0 = max(0, math.floor((-LATERAL_CROP_M - terrain.y_min) / cell_y))
    row1 = min(n_rows - 1, math.ceil((LATERAL_CROP_M - terrain.y_min) / cell_y))
    window0 = max(0, math.floor((x_lo - terrain.x_min) / cell_x))
    window1 = min(n_cols - 1, math.ceil((x_hi - terrain.x_min) / cell_x))
    tiles = []
    for i, tile in enumerate(tt.plan_friction_tiles(terrain)):
        col0, col1 = max(tile.col0, window0), min(tile.col1, window1)
        if col0 >= col1:
            continue
        whole = terrain.heights[:, tile.col0 : tile.col1 + 1]
        low = float(whole.min())
        span = max(float(whole.max()) - low, 1e-6)
        part = terrain.heights[row0 : row1 + 1, col0 : col1 + 1]
        rect = scene._grid_rect(terrain, col0, col1, row0, row1)
        name = f"terrain_tile_{i}"
        tiles.append(HeadlessTile(
            name=name,
            asset_xml=f'<hfield name="{name}" nrow="{part.shape[0]}" ncol="{part.shape[1]}" '
                      f'size="{rect["hx"]:.6f} {rect["hy"]:.6f} {span:.5f} {HFIELD_BASE_M}"/>',
            geom_xml=f'<geom name="{name}" type="hfield" hfield="{name}" pos="{rect["cx"]:.6f} {rect["cy"]:.6f} {low:.5f}" '
                     f'friction="{tile.friction:.4f} {mp.SPIN_FRICTION} {mp.ROLL_FRICTION}" priority="{scene.GROUND_PRIORITY}"/>',
            data=((part - low) / span).astype(np.float32).ravel(),
        ))
    return tiles


def generate_headless_world(seed: int = 0, cell_m: float = mp.TRACK_CELL_M, strip: tt.Strip | None = None,
                            max_lane_rock_height_m: float | None = None) -> HeadlessWorld:
    strip = strip or load_strip()
    terrain = tt.generate_track_heightmap(strip, cell_m=cell_m, seed=seed)
    x_lo, x_hi = -X_BEHIND_M, tt.finish_x(terrain) + X_AHEAD_M
    rocks = [r for r in tr.plan_track_rocks(terrain, seed, max_lane_rock_height_m) if _rock_in_window(r, x_lo, x_hi)]
    tiles = _window_tiles(terrain, x_lo, x_hi)
    head = f"""<mujoco model="jezero_headless">
  <option gravity="0 0 {scene.MARS_GRAVITY}" timestep="{scene.TIMESTEP_S}" integrator="implicitfast"/>
  <statistic center="{terrain.length_m / 2:.1f} 0 0" extent="100"/>
  <asset>
{chr(10).join("    " + t.asset_xml for t in tiles)}
  </asset>
  <worldbody>
{chr(10).join("    " + t.geom_xml for t in tiles)}
"""
    extent = np.array([max(r.sx, r.sy, r.sz) for r in rocks])
    x = np.array([r.x for r in rocks])
    return HeadlessWorld(seed, terrain, rocks, tuple(tiles), head, tuple(_rock_lines(rocks)), x - extent, x + extent)


def build_model(
    world: HeadlessWorld, design: rover.RoverDesign, prefix: str, pos, yaw: float, timestep_s: float,
    rock_range: tuple[float, float] | None = None,
) -> mujoco.MjModel:
    parts = rover.rover_mjcf(design, prefix, pos, yaw, visual=False)
    xml = (
        f"{world.head_xml}{world.rock_xml(rock_range)}\n{parts.body}\n  </worldbody>\n  <actuator>\n{parts.actuator}\n  </actuator>\n"
        f"  <equality>\n{parts.equality}\n  </equality>\n</mujoco>\n"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    for tile in world.tiles:
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, tile.name)
        adr = model.hfield_adr[hid]
        model.hfield_data[adr : adr + tile.data.size] = tile.data
    model.opt.timestep = timestep_s
    return model


def transfer_state(old_model: mujoco.MjModel, old_data: mujoco.MjData, new_model: mujoco.MjModel) -> mujoco.MjData:
    if (old_model.nq, old_model.nv, old_model.nu, old_model.na, old_model.nbody) != (new_model.nq, new_model.nv, new_model.nu, new_model.na, new_model.nbody):
        raise ValueError("the models do not hold the same rover")
    new = mujoco.MjData(new_model)
    new.time = old_data.time
    for name in ("qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart"):
        getattr(new, name)[:] = getattr(old_data, name)
    return new

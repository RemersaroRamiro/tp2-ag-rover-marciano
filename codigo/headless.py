"""The headless, physics-only model of the Jezero track, used to evaluate rovers without any graphics.

``scene.build_model`` makes a model for *looking*: textures, lights, cameras, a far backdrop, decimated visual
terrain, sand tint layers, markers, and full-resolution collision terrain and rocks over the whole 121 m wide
map. Evaluating a rover needs none of that. This module builds, from the very same generated world
(``track_terrain`` + ``track_rocks``), a model that keeps exactly what can touch a rover and nothing else:

* **Kept, unchanged:** the collision terrain at full resolution, cut into the same friction tiles, each node
  bitwise equal to the full model's (the heights are normalised with the *whole* tile's range, only the rows and
  columns outside the window are dropped); every rock that can be reached, with the same pose, size, friction
  and contact priority; gravity, solver and integrator (implicit-in-velocity, like ``rover.build_model_with_rovers``).
* **Dropped, because it cannot affect physics:** textures, materials, lights, cameras, the skybox, the visual
  terrain patches, the sand tint layers, the distance / section markers (all ``contype = conaffinity = 0``),
  and the rover's own visual-only geoms (``rover.rover_mjcf(visual=False)``).
* **Dropped, because it is physically unreachable:** the run ends (``dead(out_of_lane)``) the first time the
  rover's centre is more than ``LANE_WIDTH_M / 2 = 10 m`` off the lane axis. Nothing of the rover is further than
  ``ROVER_REACH_M`` from its centre (``design_reach_bound_m``, proven for the whole genome range by a test), and a
  run is checked 50 times per simulated second, so the furthest any part ever gets is well inside
  ``LATERAL_CROP_M = 10 + ROVER_REACH_M + CROP_MARGIN_M``. Terrain rows and rocks beyond that, the far collision
  terrain (which lies under the track terrain and beyond the strip), and the terrain west of the start / east of
  the finish by more than a rover's reach cannot ever be touched, so they are not in the model.

``tests/test_headless.py`` checks the stripped model against the full one: same terrain nodes, same friction, and
the same outcome of a closed-loop run.
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

# Upper bound on the distance from a rover's chassis origin to any point of it, for any genome (see
# ``design_reach_bound_m``: the largest value over the genome ranges is ~5.3 m).
ROVER_REACH_M = mp.ROVER_REACH_M  # defined with the other track constants: the rock rules of ``track_rocks`` use it too
CROP_MARGIN_M = 2.0  # extra ground kept beyond the reach, so no contact ever happens at the window's edge
LANE_HALF_M = mp.LANE_WIDTH_M / 2.0  # the tracker's kill line (``rover_run.RunLimits.on_track``)
LATERAL_CROP_M = LANE_HALF_M + ROVER_REACH_M + CROP_MARGIN_M  # terrain and rocks beyond |y| = this are not modelled
X_BEHIND_M = ROVER_REACH_M + CROP_MARGIN_M + 2.0  # ground kept west of x = 0 (rovers spawn at x = 4 and only drive +x)
X_AHEAD_M = ROVER_REACH_M + CROP_MARGIN_M  # ground kept east of the finish line (a finished run stops)

HFIELD_BASE_M = 2.0  # thickness of the hfield below its lowest node (same as ``scene``)
HYSTERESIS_M = 2.0  # a rock window is only changed once the rover is this far outside its zone (no thrashing at a border)


def design_reach_bound_m(design: rover.RoverDesign) -> float:
    """Upper bound on how far any point of the rover can be from its chassis origin, in any suspension pose.

    The chassis box is bounded by its farthest corner. A wheel hangs at the end of a rigid chain from the rocker
    pivot (the front arm, or the rear arm plus the bogie half-arm), so by the triangle inequality it is at most
    ``|pivot| + chain + wheel half-diagonal`` from the origin however the two hinges rotate."""
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
    """Distance from the start line to the finish line along +x (373.45 m on the committed strip)."""
    strip = strip or load_strip()
    return strip.length_m - mp.FINISH_MARGIN_M - mp.START_X_M


@dataclass(frozen=True)
class HeadlessTile:
    """One friction tile of the window: its asset / geom XML and the normalised node heights to write."""

    name: str
    asset_xml: str
    geom_xml: str
    data: np.ndarray  # float32, row-major (row = +y, column = +x), in [0, 1] of the whole tile's range


@dataclass(frozen=True)
class HeadlessWorld:
    """Everything a rover evaluation needs of one seed: the generated terrain and the compiled-ready window."""

    seed: int
    terrain: tt.TrackTerrain  # the full heightmap (spawning samples it); only ``tiles`` go into the model
    rocks: list[mt.Rock]  # the rocks inside the window
    tiles: tuple[HeadlessTile, ...]
    head_xml: str  # options, hfield assets and terrain geoms, up to the rocks
    rock_lines: tuple[str, ...]  # one geom element per rock, in the order of ``rocks``
    rock_x_lo: np.ndarray  # x extent of every rock (centre -+ its largest semi-axis)
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
        """The rocks whose x extent overlaps ``[x_lo, x_hi]``."""
        keep = (self.rock_x_hi >= x_lo) & (self.rock_x_lo <= x_hi)
        return [r for r, k in zip(self.rocks, keep) if k]

    def rock_xml(self, rock_range: tuple[float, float] | None = None) -> str:
        if rock_range is None:
            return "\n".join(self.rock_lines)
        keep = (self.rock_x_hi >= rock_range[0]) & (self.rock_x_lo <= rock_range[1])
        return "\n".join(line for line, k in zip(self.rock_lines, keep) if k)


@dataclass(frozen=True)
class RockWindows:
    """Cuts the lane into zones ``width_m`` long; a rover in a zone only needs the rocks within its reach of that zone.

    Every rock is a geom, and MuJoCo updates the pose of every geom on every step (about 10 ns each): the ~3500 rocks
    of the lane cost as much as the six wheels' contact handling. A model that holds only the rocks near the rover
    (``rock_range``) does the same physics for less. The rover may move ``HYSTERESIS_M`` beyond its zone before the
    model is swapped, and a window holds ``ROVER_REACH_M + CROP_MARGIN_M + HYSTERESIS_M`` of rocks on each side of
    the zone, so a rock that any part of the rover can touch is always there (``tests/test_headless.py``)."""

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
    """Collision-only rock geoms: the same numbers, formats, friction and priority as ``scene._rock_xml``."""
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
    """The friction tiles of ``scene._collision_tiles_xml`` cut to the window (rows |y| <= LATERAL_CROP_M, x in
    [x_lo, x_hi]), with node heights bitwise equal to the full model's."""
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
        whole = terrain.heights[:, tile.col0 : tile.col1 + 1]  # normalise with the whole tile, like the full model
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
    """The track of ``seed`` as the physics sees it: the same terrain and rocks as ``scene.generate_track_world``
    (same generators, same arguments), without the far backdrop and the visual copies, cut to the reachable window.
    ``max_lane_rock_height_m`` is ``track_rocks.plan_track_rocks``' explicit cap (tests craft one instead of patching
    the module default)."""
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
    """Compile the world window with one rover of ``design`` at ``pos`` / ``yaw`` (see ``rover.spawn_pose``).
    ``rock_range = (x_lo, x_hi)`` keeps only the rocks overlapping that x range (see ``RockWindows``)."""
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
    """A fresh ``MjData`` of ``new_model`` (same rover, other rocks) holding the dynamic state of ``old_data``: time,
    positions, velocities, controls, actuator state and the solver warm start. Only the static geoms differ between
    the two models, so every body, joint, dof and actuator id is the same."""
    if (old_model.nq, old_model.nv, old_model.nu, old_model.na, old_model.nbody) != (new_model.nq, new_model.nv, new_model.nu, new_model.na, new_model.nbody):
        raise ValueError("the models do not hold the same rover")
    new = mujoco.MjData(new_model)
    new.time = old_data.time
    for name in ("qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart"):
        getattr(new, name)[:] = getattr(old_data, name)
    return new

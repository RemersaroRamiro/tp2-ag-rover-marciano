"""MuJoCo Mars scenes: heightfield terrain, rocks, sky, haze, sun and fixed cameras.

``build_model(seed)`` is the single entry point used by ``view.py``, ``screenshot.py`` and, later,
the rover simulation. It builds the Jezero *track* by default (``kind="track"``: a long, narrow
lane over a real HiRISE elevation strip with eight named mixed-terrain sections, see
``track_terrain.py`` and ``track_sections.py``); ``kind="sandbox"`` keeps the original square
procedural map (``mars_terrain.py``). Everything is deterministic per seed and
self-contained: the ground texture is embedded through MuJoCo's in-memory asset dictionary, so it
works from any working directory.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import mars_params as mp
import mars_terrain as mt
import track_rocks as tr
import track_sections as ts
import track_terrain as tt
import track_texture as ttex

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
REGOLITH_TILE = "ground_regolith_tile.jpg"
REGOLITH_ASSET_NAME = "regolith.png"

DEFAULT_SIZE = 513  # heightfield nodes per side (0.23 m cells): 257 shows visible faceting on small craters
MARS_GRAVITY = -mp.MARS_GRAVITY_M_S2  # m/s^2 (z component)
TIMESTEP_S = 0.002
GROUND_FRICTION = (0.6, 0.005, 0.0001)  # sliding, torsional, rolling: loose regolith
TILE_WORLD_WIDTH_M = 4.0  # physical width covered by one repetition of the regolith tile
FAR_TILE_SCALE = 2.0  # the far terrain repeats the regolith tile this many times larger
# Fog distances in units of model extent (100 m). Fog is a scene render flag, not part of the model:
# screenshot.py switches it on; in the interactive viewer toggle it in the Rendering panel.
FOG_START, FOG_END = 1.0, 6.0

# Shadow map coverage in units of model extent. Larger values produced shadow acne on the heightfield.
SHADOW_CLIP = 0.3

# Geom group of geoms that only collide. MuJoCo's viewer and Renderer draw groups 0-2 by default.
COLLISION_GEOM_GROUP = 3

SANDBOX_CAMERA_NAMES = ("overview", "ground", "top")
TRACK_CAMERA_NAMES = ("start", "overview", "side", "top")
CAMERA_NAMES = TRACK_CAMERA_NAMES  # the overview cameras of the default (track) scene
SECTION_CAMERA_PREFIX = "section_"  # plus one camera per section: "section_floor", "section_crater", ...

# Colours (linear-ish RGB in [0, 1]).
SKY_ZENITH = (0.40, 0.27, 0.20)
SKY_LOW = (0.98, 0.72, 0.50)  # MuJoCo's skybox gradient puts the horizon at the midpoint of the two colors
FOG_COLOR = tuple((a + b) / 2 for a, b in zip(SKY_ZENITH, SKY_LOW))  # so fog matches the horizon
SUN_COLOR = (1.0, 0.82, 0.62)
SUN_DIRECTION = (-0.62, -0.42, -0.42)  # direction the light travels; about 29 degrees above the horizon
ROCK_COLORS = {
    "basalt": (0.27, 0.20, 0.17),
    "rust": (0.48, 0.26, 0.16),
    "dust": (0.38, 0.29, 0.23),
    "ochre": (0.56, 0.37, 0.23),
}


@dataclass(frozen=True)
class World:
    """Everything generated for one seed, before it is turned into MJCF."""

    seed: int
    extent_m: float
    heights: np.ndarray
    far_heights: np.ndarray
    rocks: list[mt.Rock]
    craters: tuple[mt.Crater, ...]
    dunes: mt.DuneField


def generate_world(
    seed: int = 0, size: int = DEFAULT_SIZE, extent_m: float = 120.0, rock_count: int = 150
) -> World:
    heights = mt.generate_heightmap(size=size, extent_m=extent_m, seed=seed)
    craters = mt.plan_craters(extent_m, seed)
    rocks = mt.scatter_rocks(heights, extent_m, seed, count=rock_count, craters=craters)
    far_heights = mt.generate_far_heightmap(extent_m, seed)
    return World(seed, extent_m, heights, far_heights, rocks, craters, mt.plan_dune_field(extent_m, seed))


@lru_cache(maxsize=1)
def load_assets() -> dict[str, bytes]:
    """Texture bytes for ``MjModel.from_xml_string`` (MuJoCo reads PNG, so the JPEG is converted)."""
    buffer = io.BytesIO()
    Image.open(ASSETS_DIR / REGOLITH_TILE).convert("RGB").save(buffer, format="PNG")
    return {REGOLITH_ASSET_NAME: buffer.getvalue()}


# -- cameras -------------------------------------------------------------------------------


# MuJoCo uses the higher-priority geom's friction in a contact (the max of both only on equal priority): the ground
# and rocks outrank default geoms, so a wheel's own friction never masks the soft-sand zone.
GROUND_PRIORITY = 1


def _rgb(color: tuple[float, float, float]) -> str:
    return " ".join(f"{c:.4f}" for c in color)


def _look_at_axes(position: np.ndarray, target: np.ndarray) -> str:
    """MJCF ``xyaxes`` of a camera at ``position`` looking at ``target`` (world +z is up)."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return " ".join(f"{v:.5f}" for v in (*right, *up))


def _sandbox_camera_layout(world: World) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(position, target) of the sandbox cameras, derived from the generated layout."""
    h = world.heights
    ext = world.extent_m
    biggest = max(world.craters, key=lambda c: c.radius)
    dunes = world.dunes

    def ground_z(x: float, y: float) -> float:
        return mt.sample_height(h, ext, x, y)

    # overview: high and oblique, looking at the middle of the interesting stuff.
    focus_xy = (np.array([biggest.x, biggest.y]) + np.array([dunes.x, dunes.y])) / 2 * 0.7
    # Look across the sun direction so crater walls and dune crests are lit from the side.
    sun_h = np.array(SUN_DIRECTION[:2]) / np.linalg.norm(SUN_DIRECTION[:2])
    away = np.array([-sun_h[1], sun_h[0]])
    if away @ -focus_xy < 0:
        away = -away
    overview_pos = np.array([*(focus_xy + away * 72.0), ground_z(*focus_xy) + 27.0])
    overview_target = np.array([*focus_xy, ground_z(*focus_xy)])

    # ground: eye height next to the spawn zone, looking across the plain to the biggest crater rim.
    to_crater = np.array([biggest.x, biggest.y])
    to_crater = to_crater / np.linalg.norm(to_crater)
    eye_xy = -to_crater * 3.5
    rim_xy = np.array([biggest.x, biggest.y]) - to_crater * biggest.radius
    ground_pos = np.array([*eye_xy, ground_z(*eye_xy) + 1.0])
    ground_target = np.array([*rim_xy, ground_z(*rim_xy) + 1.5])

    top_pos = np.array([0.0, 0.0, 150.0])
    top_target = np.array([0.0, 0.0, 0.0])
    return {
        "overview": (overview_pos, overview_target),
        "ground": (ground_pos, ground_target),
        "top": (top_pos, top_target),
    }


# -- MJCF ----------------------------------------------------------------------------------


def _camera_xml(world: World | TrackWorld) -> str:
    fovy = {"overview": 50, "ground": 60, "top": 48, "start": 60, "side": 50}
    if isinstance(world, TrackWorld):
        fovy["overview"] = 55
    lines = []
    for name, (pos, target) in camera_layout(world).items():
        axes = "1 0 0 0 1 0" if name == "top" else _look_at_axes(pos, target)
        lines.append(
            f'    <camera name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'xyaxes="{axes}" fovy="{fovy.get(name, 60)}"/>'
        )
    return "\n".join(lines)


def _rock_quat(r: mt.Rock) -> np.ndarray:
    """Orientation: yaw about z, then pitch about y, then roll about x (all small but the yaw)."""

    def about(axis: int, angle: float) -> np.ndarray:
        q = np.zeros(4)
        q[0], q[1 + axis] = np.cos(angle / 2), np.sin(angle / 2)
        return q

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    return mul(mul(about(2, r.yaw), about(1, r.pitch)), about(0, r.roll))


def _rock_xml(rocks: list[mt.Rock], friction: str = "0.8 0.005 0.0001") -> str:
    variants = list(ROCK_COLORS)
    lines = []
    for i, r in enumerate(rocks):
        quat = _rock_quat(r)
        variant = variants[i % len(variants)]
        # A per-rock rgba (only when shaded) overrides the material colour but keeps its gloss.
        shaded = tuple(min(c * r.shade, 1.0) for c in ROCK_COLORS[variant])
        rgba = "" if r.shade == 1.0 else f'rgba="{_rgb(shaded)} 1" '
        lines.append(
            f'    <geom name="rock_{i}" type="ellipsoid" group="1" material="rock_{variant}" {rgba}'
            f'size="{r.sx:.4f} {r.sy:.4f} {r.sz:.4f}" pos="{r.x:.4f} {r.y:.4f} {r.z:.4f}" '
            f'quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}" friction="{friction}" priority="{GROUND_PRIORITY}"/>'
        )
    return "\n".join(lines)


def _sandbox_scene_xml(world: World) -> str:
    heights = world.heights
    n_rows, n_cols = heights.shape
    half = world.extent_m / 2
    h_min, h_max = float(heights.min()), float(heights.max())
    elevation = max(h_max - h_min, 1e-6)
    tile_aspect = 224 / 416  # the regolith tile is 416 x 224 px
    repeat_x = world.extent_m / TILE_WORLD_WIDTH_M
    repeat_y = world.extent_m / (TILE_WORLD_WIDTH_M * tile_aspect)
    far_half = mt.FAR_EXTENT_M / 2
    far_min, far_max = float(world.far_heights.min()), float(world.far_heights.max())
    far_repeat_x = mt.FAR_EXTENT_M / (TILE_WORLD_WIDTH_M * FAR_TILE_SCALE)
    far_repeat_y = far_repeat_x / tile_aspect
    friction = " ".join(str(f) for f in GROUND_FRICTION)
    rock_materials = "\n".join(
        f'    <material name="rock_{name}" rgba="{_rgb(color)} 1" specular="0.05" shininess="0.05"/>'
        for name, color in ROCK_COLORS.items()
    )
    return f"""<mujoco model="mars_scene">
  <option gravity="0 0 {MARS_GRAVITY}" timestep="{TIMESTEP_S}"/>
  <statistic center="0 0 0" extent="100"/>

  <visual>
    <global offwidth="1600" offheight="900"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.34 0.26 0.21" diffuse="0.12 0.09 0.07" specular="0 0 0"/>
    <map fogstart="{FOG_START}" fogend="{FOG_END}" znear="0.0005" zfar="60" haze="0.4" shadowclip="{SHADOW_CLIP}"/>
    <rgba fog="{_rgb(FOG_COLOR)} 1" haze="{_rgb(FOG_COLOR)} 1"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{_rgb(SKY_ZENITH)}" rgb2="{_rgb(SKY_LOW)}"
             width="512" height="3072"/>
    <texture name="regolith" type="2d" file="{REGOLITH_ASSET_NAME}"/>
    <material name="regolith" texture="regolith" texrepeat="{repeat_x:.3f} {repeat_y:.3f}"
              specular="0.02" shininess="0.02"/>
    <material name="regolith_far" texture="regolith" texrepeat="{far_repeat_x:.2f} {far_repeat_y:.2f}"
              specular="0.02" shininess="0.02"/>
{rock_materials}
    <hfield name="terrain" nrow="{n_rows}" ncol="{n_cols}" size="{half:.3f} {half:.3f} {elevation:.5f} 2.0"/>
    <hfield name="far_terrain" nrow="{world.far_heights.shape[0]}" ncol="{world.far_heights.shape[1]}"
            size="{far_half:.1f} {far_half:.1f} {max(far_max - far_min, 1e-6):.5f} 12.0"/>
  </asset>

  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="0 0 100" dir="{_rgb(SUN_DIRECTION)}"
           diffuse="{_rgb(SUN_COLOR)}" specular="0.08 0.06 0.05"/>
    <geom name="terrain" type="hfield" hfield="terrain" material="regolith" pos="0 0 {h_min:.5f}"
          friction="{friction}"/>
    <geom name="far_terrain" type="hfield" hfield="far_terrain" material="regolith_far" group="2"
          pos="0 0 {far_min:.5f}" friction="{friction}"/>
{_rock_xml(world.rocks)}
{_camera_xml(world)}
  </worldbody>
</mujoco>
"""


def visual_scene_option() -> mujoco.MjvOption:
    """Render options that draw the visual groups only (0-2): the full-resolution collision terrain of
    the track lives in ``COLLISION_GEOM_GROUP`` and would only cost triangles."""
    option = mujoco.MjvOption()
    hide_collision_geoms(option)
    return option


def hide_collision_geoms(option: mujoco.MjvOption) -> None:
    """Switch the collision-only geom group off in ``option`` (e.g. the passive viewer's ``viewer.opt``)."""
    option.geomgroup[COLLISION_GEOM_GROUP] = 0


def _write_hfield(model: mujoco.MjModel, name: str, heights: np.ndarray) -> None:
    """Fill a compiled hfield with heights normalised to [0, 1] (row-major: row = +y, column = +x)."""
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    adr = model.hfield_adr[hid]
    span = max(float(heights.max() - heights.min()), 1e-6)
    model.hfield_data[adr : adr + heights.size] = ((heights - heights.min()) / span).ravel()


# -- the Jezero track ------------------------------------------------------------------------------

TRACK_TILE_WORLD_WIDTH_M = 5.0  # physical width of one photograph tile on the track
TRACK_TEXTURE_CELLS = (10, 5)  # tiles in the big anti-repetition texture: 50 m x 13.5 m before it repeats
# The far backdrop repeats the plain photograph tile at the track's own grain size: the big track
# texture would show its own period as stripes at grazing angles. Same mean colour and grain.
TRACK_FAR_TEXTURE_SCALE = 1.0
TRACK_TEXTURE_ASSET_NAME = "regolith_track.png"
TILE_ASPECT = 224 / 416  # the regolith tile is 416 x 224 px
TRACK_FOG_START, TRACK_FOG_END = 2.5, 12.0  # in units of model extent (100 m): 250 m to 1.2 km
# Light travels +y (the lane is lit from its right side) and +x: the x component grazes the faces of ripple
# crests and bedrock ledges (their crests run across the lane), which the earlier (0.2, 0.8, -0.5) left flat.
TRACK_SUN_DIRECTION = (0.55, 0.70, -0.50)
TRACK_SHADOW_CLIP = 0.3  # 1.0 produced shadow acne (dotted black patterns) along the lane
MARKER_COLORS = {
    "start": (0.80, 0.74, 0.62),  # pale cream
    "finish": (0.55, 0.22, 0.15),  # muted brick red
    "distance": (0.72, 0.45, 0.20),  # dull ochre
}
# Subtle, distinguishable colours of the stake pairs that mark the start of each section.
SECTION_MARKER_COLORS = {
    "rock_field": (0.50, 0.44, 0.38),  # stone grey-brown
    "ripples": (0.78, 0.60, 0.30),  # ochre
    "soft_sand": (0.86, 0.78, 0.55),  # pale sand
    "bedrock": (0.36, 0.40, 0.46),  # slate
    "cross_slope": (0.32, 0.52, 0.50),  # muted teal
    "crater": (0.62, 0.30, 0.20),  # rust
    "climb": (0.82, 0.78, 0.70),  # cream
}
SAND_DECAL_LIFT_M = 0.03  # the tint layers hover this far over the visual terrain (their alpha hides the edges)
SAND_DECAL_THICKNESS_M = 0.01
SAND_TINT_ASSET = "ground_sand_tile.jpg"  # its mean colour is the tint of the soft-sand zone
MARKER_SINK_M = 0.05  # posts are planted this deep, so they never float on the rough ground
START_CAMERA_BACK_M = 4.0
START_CAMERA_HEIGHT_M = 2.2


@dataclass(frozen=True)
class TrackWorld:
    """Everything generated for one seed of the track, before it is turned into MJCF."""

    seed: int
    terrain: tt.TrackTerrain
    far: tt.FarTerrain
    rocks: list[mt.Rock]
    markers: list[tt.Marker]
    visual_terrain: tt.VisualTerrain  # decimated copy of ``terrain`` that is drawn (collision stays full-res)
    visual_far: tt.FarTerrain

    @property
    def sections(self) -> tuple[ts.Section, ...]:
        """The named sections of the track in order (metadata for per-section analysis)."""
        return self.terrain.sections

    def section_at(self, x: float) -> ts.Section:
        """The section under true-ground ``x`` (positions beyond the ends belong to the first / last)."""
        return ts.section_at(x, self.sections)

    def section_weights(self, x) -> np.ndarray:
        """Blend weights of every section at ``x`` (they sum to 1)."""
        return ts.section_weights(x, self.sections)


def generate_track_world(
    seed: int = 0,
    cell_m: float = mp.TRACK_CELL_M,
    far_pad_cells: int = tt.FAR_DEFAULT_PAD_CELLS,
    strip: tt.Strip | None = None,
    max_lane_rock_height_m: float | None = None,
) -> TrackWorld:
    """Real DTM relief (same for every seed) + seeded roughness and rocks. Larger ``cell_m`` and
    smaller ``far_pad_cells`` make the world coarser and faster, handy for tests. ``max_lane_rock_height_m`` is
    ``track_rocks.plan_track_rocks``' explicit cap on the height of the rocks in the lane (default: the module's)."""
    strip = strip or tt.load_strip()
    terrain = tt.generate_track_heightmap(strip, cell_m=cell_m, seed=seed)
    far = tt.generate_far_terrain(strip, terrain, seed=seed, pad_cells=far_pad_cells)
    visual_terrain, visual_far = tt.make_visual_terrains(terrain, far)
    rocks, markers = tr.plan_track_rocks(terrain, seed, max_lane_rock_height_m), tt.plan_markers(terrain)
    return TrackWorld(seed, terrain, far, rocks, markers, visual_terrain, visual_far)


def lane_length_m(world: TrackWorld) -> float:
    """Distance from the start line to the finish line along +x."""
    return tt.finish_x(world.terrain) - mp.START_X_M


@lru_cache(maxsize=1)
def load_track_assets() -> dict[str, bytes]:
    """Big seamless ground texture of the track (``track_texture.make_track_texture``) as PNG bytes."""
    tile = np.asarray(Image.open(ASSETS_DIR / REGOLITH_TILE).convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(ttex.make_track_texture(tile, cells=TRACK_TEXTURE_CELLS)).save(buffer, format="PNG")
    return {**load_assets(), TRACK_TEXTURE_ASSET_NAME: buffer.getvalue()}


SECTION_CAMERA_BACK_M = 24.0  # section cameras stand this far before the section centre ...
SECTION_CAMERA_AHEAD_M = 18.0  # ... and look this far past it
SECTION_CAMERA_HEIGHT_M = 3.2


def _track_camera_layout(world: TrackWorld) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    terrain = world.terrain

    def ground(x: float, y: float = 0.0) -> float:
        return float(terrain.sample(x, y))

    eye_x = mp.START_X_M - START_CAMERA_BACK_M
    ahead_x = mp.START_X_M + 60.0
    mid_x = 0.5 * terrain.length_m
    top_x = 45.0
    layout = {
        # rover-eye-ish, behind the start line, looking down the lane
        "start": (
            np.array([eye_x, 0.0, ground(eye_x) + START_CAMERA_HEIGHT_M]),
            np.array([ahead_x, 0.0, ground(ahead_x) + 0.8]),
        ),
        # high and oblique from behind the start: the whole lane runs away in perspective
        "overview": (
            np.array([-70.0, -70.0, ground(0.0) + 50.0]),
            np.array([170.0, 0.0, ground(170.0)]),
        ),
        # from the side, mid-track
        "side": (
            np.array([mid_x - 45.0, -32.0, ground(mid_x - 45.0, -30.0) + 6.0]),
            np.array([mid_x, 0.0, ground(mid_x) + 0.5]),
        ),
        # straight down on a section of the lane (square frame, 62 m across)
        "top": (
            np.array([top_x, 0.0, ground(top_x) + 70.0]),
            np.array([top_x, 0.0, ground(top_x)]),
        ),
    }
    for section in world.sections:  # chase-like views: from behind and above, looking along the lane at the section
        eye_x = section.center - SECTION_CAMERA_BACK_M
        aim_x = min(section.center + SECTION_CAMERA_AHEAD_M, section.x_end + 5.0)
        layout[SECTION_CAMERA_PREFIX + section.name] = (
            np.array([eye_x, 0.0, ground(eye_x) + SECTION_CAMERA_HEIGHT_M]),
            np.array([aim_x, 0.0, ground(aim_x) + 0.3]),
        )
    return layout


def _marker_xml(markers: list[tt.Marker]) -> str:
    lines = []
    for m in markers:
        half = m.height / 2
        material = f"marker_section_{m.section}" if m.kind == "section" else f"marker_{m.kind}"
        lines.append(
            f'    <geom name="{m.name}" type="cylinder" group="1" contype="0" conaffinity="0" '
            f'material="{material}" size="{m.radius:.3f} {half:.3f}" '
            f'pos="{m.x:.3f} {m.y:.3f} {m.z_base + half - MARKER_SINK_M:.3f}"/>'
        )
    return "\n".join(lines)


def _hfield_geometry(heights: np.ndarray, half_x: float, half_y: float, base_m: float) -> dict[str, str]:
    """Attribute strings shared by a heightfield asset and its geom: ``size`` and the geom's z position."""
    low, high = float(heights.min()), float(heights.max())
    return {
        "nrow": str(heights.shape[0]),
        "ncol": str(heights.shape[1]),
        "size": f"{half_x:.6f} {half_y:.6f} {max(high - low, 1e-6):.5f} {base_m}",
        "z": f"{low:.5f}",
    }


@lru_cache(maxsize=1)
def sand_tint_color() -> tuple[float, float, float]:
    """Mean colour of the NASA sand photograph: the tint the soft-sand zone is blended towards."""
    mean = np.asarray(Image.open(ASSETS_DIR / SAND_TINT_ASSET).convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    return tuple(float(c) / 255.0 for c in mean)


def _grid_rect(terrain: tt.TrackTerrain, col0: int, col1: int, row0: int = 0, row1: int | None = None) -> dict[str, float]:
    """Centre and half extents (true meters) of the node rectangle ``col0..col1`` x ``row0..row1``."""
    n_rows, n_cols = terrain.heights.shape
    row1 = n_rows - 1 if row1 is None else row1
    cell_x = (terrain.x_max - terrain.x_min) / (n_cols - 1)
    cell_y = (terrain.y_max - terrain.y_min) / (n_rows - 1)
    return {
        "cx": terrain.x_min + 0.5 * (col0 + col1) * cell_x, "cy": terrain.y_min + 0.5 * (row0 + row1) * cell_y,
        "hx": 0.5 * (col1 - col0) * cell_x, "hy": 0.5 * (row1 - row0) * cell_y,
    }


def _collision_tiles_xml(world: TrackWorld) -> tuple[str, str]:
    """(hfield assets, geoms) of the collision terrain: slices along x, each with the friction of its section."""
    terrain = world.terrain
    assets, geoms = [], []
    for i, tile in enumerate(tt.plan_friction_tiles(terrain)):
        heights = terrain.heights[:, tile.col0 : tile.col1 + 1]
        rect = _grid_rect(terrain, tile.col0, tile.col1)
        geometry = _hfield_geometry(heights, rect["hx"], rect["hy"], 2.0)
        assets.append(
            f'    <hfield name="terrain_tile_{i}" nrow="{geometry["nrow"]}" ncol="{geometry["ncol"]}" size="{geometry["size"]}"/>'
        )
        geoms.append(
            f'    <geom name="terrain_tile_{i}" type="hfield" hfield="terrain_tile_{i}" group="{COLLISION_GEOM_GROUP}" '
            f'pos="{rect["cx"]:.6f} {rect["cy"]:.6f} {geometry["z"]}" '
            f'friction="{tile.friction:.4f} {mp.SPIN_FRICTION} {mp.ROLL_FRICTION}" priority="{GROUND_PRIORITY}"/>'
        )
    return "\n".join(assets), "\n".join(geoms)


def _decal_heights(world: TrackWorld, decal: tt.Decal) -> np.ndarray:
    """Heights of a tint layer: the drawn ground sampled on the visual grid's own spacing (0.25 m along x,
    3 rows across), lifted by ``SAND_DECAL_LIFT_M`` so the layer never sinks into the surface."""
    terrain = world.terrain
    cell_x = (terrain.x_max - terrain.x_min) / (terrain.heights.shape[1] - 1)
    cell_y = (terrain.y_max - terrain.y_min) / (terrain.heights.shape[0] - 1)
    xs = np.linspace(decal.x0, decal.x1, max(2, round((decal.x1 - decal.x0) / cell_x) + 1))
    ys = np.linspace(-decal.half_width, decal.half_width, max(2, round(2 * decal.half_width / (3 * cell_y)) + 1))
    return world.visual_terrain.sample(*np.meshgrid(xs, ys)) + SAND_DECAL_LIFT_M


def _visual_terrain_xml(world: TrackWorld) -> tuple[str, str, str]:
    """(materials, hfield assets, geoms) of the drawn ground: one hfield per segment along x.

    Each segment maps the ground texture from its own west edge with a repeat count equal to its length
    in texture periods; the seams lie on multiples of the period (``tt.VISUAL_SEGMENT_ALIGN_M``), so the
    texture continues across them.
    """
    texture_w = TRACK_TEXTURE_CELLS[0] * TRACK_TILE_WORLD_WIDTH_M  # world size of the whole texture
    texture_h = TRACK_TEXTURE_CELLS[1] * TRACK_TILE_WORLD_WIDTH_M * TILE_ASPECT
    materials, assets, geoms = [], [], []
    for i, segment in enumerate(world.visual_terrain.segments):
        rect = _grid_rect(segment, 0, segment.heights.shape[1] - 1)
        geometry = _hfield_geometry(segment.heights, rect["hx"], rect["hy"], 2.0)
        repeat_x, repeat_y = 2 * rect["hx"] / texture_w, 2 * rect["hy"] / texture_h
        materials.append(
            f'    <material name="regolith_{i}" texture="regolith" texrepeat="{repeat_x:.4f} {repeat_y:.4f}" '
            f'specular="0.02" shininess="0.02"/>'
        )
        assets.append(
            f'    <hfield name="terrain_visual_{i}" nrow="{geometry["nrow"]}" ncol="{geometry["ncol"]}" size="{geometry["size"]}"/>'
        )
        geoms.append(
            f'    <geom name="terrain_visual_{i}" type="hfield" hfield="terrain_visual_{i}" material="regolith_{i}" '
            f'contype="0" conaffinity="0" pos="{rect["cx"]:.6f} {rect["cy"]:.6f} {geometry["z"]}"/>'
        )
    return "\n".join(materials), "\n".join(assets), "\n".join(geoms)


def _sand_decals_xml(world: TrackWorld) -> tuple[str, str, str]:
    """(material, hfield assets, geoms) of the translucent soft-sand tint layers (visual only)."""
    decals = tt.plan_sand_decals(world.sections)
    if not decals:
        return "", "", ""
    material = (
        f'    <material name="sand_tint" rgba="{_rgb(sand_tint_color())} {decals[0].alpha:.4f}" '
        f'specular="0" shininess="0"/>'
    )
    assets, geoms = [], []
    for i, d in enumerate(decals):
        heights = _decal_heights(world, d)
        size = f'{0.5 * (d.x1 - d.x0):.4f} {d.half_width:.4f} {max(float(np.ptp(heights)), 1e-6):.5f} {SAND_DECAL_THICKNESS_M}'
        assets.append(f'    <hfield name="sand_decal_{i}" nrow="{heights.shape[0]}" ncol="{heights.shape[1]}" size="{size}"/>')
        geoms.append(
            f'    <geom name="sand_decal_{i}" type="hfield" hfield="sand_decal_{i}" material="sand_tint" contype="0" '
            f'conaffinity="0" pos="{0.5 * (d.x0 + d.x1):.4f} 0 {float(heights.min()):.5f}"/>'
        )
    return material, "\n".join(assets), "\n".join(geoms)


def _track_scene_xml(world: TrackWorld) -> str:
    terrain, far = world.terrain, world.far
    n_rows, n_cols = terrain.heights.shape
    h_min, h_max = float(terrain.heights.min()), float(terrain.heights.max())
    far_n_rows, far_n_cols = far.heights.shape
    far_half_x, far_half_y = 0.5 * far.cell_x * (far_n_cols - 1), 0.5 * far.cell_y * (far_n_rows - 1)
    far_min, far_max = float(far.heights.min()), float(far.heights.max())
    far_repeat_x = 2 * far_half_x / (TRACK_TILE_WORLD_WIDTH_M * TRACK_FAR_TEXTURE_SCALE)
    far_repeat_y = 2 * far_half_y / (TRACK_TILE_WORLD_WIDTH_M * TRACK_FAR_TEXTURE_SCALE * TILE_ASPECT)
    vfar = world.visual_far
    vfar_half_x = 0.5 * vfar.cell_x * (vfar.heights.shape[1] - 1)
    vfar_half_y = 0.5 * vfar.cell_y * (vfar.heights.shape[0] - 1)
    vfar_x, vfar_y = vfar.x0 + vfar_half_x, vfar.y0 + vfar_half_y
    visual_far = _hfield_geometry(vfar.heights, vfar_half_x, vfar_half_y, 12.0)
    friction = f"{mp.SOIL_FRICTION} {mp.SPIN_FRICTION} {mp.ROLL_FRICTION}"  # the far backdrop is plain soil
    rock_friction = f"{mp.ROCK_FRICTION} {mp.SPIN_FRICTION} {mp.ROLL_FRICTION}"
    rock_materials = "\n".join(
        f'    <material name="rock_{name}" rgba="{_rgb(color)} 1" specular="0.05" shininess="0.05"/>'
        for name, color in ROCK_COLORS.items()
    )
    marker_materials = "\n".join(
        f'    <material name="marker_{kind}" rgba="{_rgb(color)} 1" specular="0.05" shininess="0.05"/>'
        for kind, color in MARKER_COLORS.items()
    ) + "\n" + "\n".join(
        f'    <material name="marker_section_{name}" rgba="{_rgb(color)} 1" specular="0.05" shininess="0.05"/>'
        for name, color in SECTION_MARKER_COLORS.items()
    )
    tile_assets, tile_geoms = _collision_tiles_xml(world)
    ground_materials, ground_assets, ground_geoms = _visual_terrain_xml(world)
    sand_material, sand_assets, sand_geoms = _sand_decals_xml(world)
    return f"""<mujoco model="jezero_track">
  <option gravity="0 0 {MARS_GRAVITY}" timestep="{TIMESTEP_S}"/>
  <statistic center="{terrain.length_m / 2:.1f} 0 0" extent="100"/>

  <visual>
    <global offwidth="1600" offheight="900"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.34 0.26 0.21" diffuse="0.12 0.09 0.07" specular="0 0 0"/>
    <map fogstart="{TRACK_FOG_START}" fogend="{TRACK_FOG_END}" znear="0.0005" zfar="60" haze="0.4" shadowclip="{TRACK_SHADOW_CLIP}"/>
    <rgba fog="{_rgb(FOG_COLOR)} 1" haze="{_rgb(FOG_COLOR)} 1"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{_rgb(SKY_ZENITH)}" rgb2="{_rgb(SKY_LOW)}"
             width="512" height="3072"/>
    <texture name="regolith" type="2d" file="{TRACK_TEXTURE_ASSET_NAME}"/>
    <texture name="regolith_tile" type="2d" file="{REGOLITH_ASSET_NAME}"/>
{ground_materials}
    <material name="regolith_far" texture="regolith_tile" texrepeat="{far_repeat_x:.2f} {far_repeat_y:.2f}"
              specular="0.02" shininess="0.02"/>
{rock_materials}
{marker_materials}
{sand_material}
{tile_assets}
    <hfield name="far_terrain" nrow="{far_n_rows}" ncol="{far_n_cols}"
            size="{far_half_x:.3f} {far_half_y:.3f} {max(far_max - far_min, 1e-6):.5f} 12.0"/>
{ground_assets}
    <hfield name="far_terrain_visual" nrow="{visual_far['nrow']}" ncol="{visual_far['ncol']}"
            size="{visual_far['size']}"/>
{sand_assets}
  </asset>

  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="{terrain.length_m / 2:.1f} 0 100"
           dir="{_rgb(TRACK_SUN_DIRECTION)}" diffuse="{_rgb(SUN_COLOR)}" specular="0.08 0.06 0.05"/>
    <!-- Physics: full-resolution heightfields in a group that is not drawn. The near terrain is cut into
         tiles along x (friction is per geom): neighbours share their edge heights, so there is no step. -->
{tile_geoms}
    <geom name="far_terrain" type="hfield" hfield="far_terrain" group="{COLLISION_GEOM_GROUP}"
          pos="{far.x0 + far_half_x:.3f} {far.y0 + far_half_y:.3f} {far_min:.5f}" friction="{friction}"
          priority="{GROUND_PRIORITY}"/>
    <!-- Looks: decimated copies with the same material; they never collide. -->
{ground_geoms}
    <geom name="far_terrain_visual" type="hfield" hfield="far_terrain_visual" material="regolith_far" group="2"
          contype="0" conaffinity="0" pos="{vfar_x:.3f} {vfar_y:.3f} {visual_far['z']}"/>
{sand_geoms}
{_rock_xml(world.rocks, rock_friction)}
{_marker_xml(world.markers)}
{_camera_xml(world)}
  </worldbody>
</mujoco>
"""


# -- dispatch: track (default) or sandbox -----------------------------------------------------------


def camera_layout(world: World | TrackWorld) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(position, target) of the named fixed cameras of either kind of world."""
    if isinstance(world, TrackWorld):
        return _track_camera_layout(world)
    return _sandbox_camera_layout(world)


def camera_names(world: World | TrackWorld) -> tuple[str, ...]:
    if isinstance(world, TrackWorld):
        return TRACK_CAMERA_NAMES + tuple(SECTION_CAMERA_PREFIX + s.name for s in world.sections)
    return SANDBOX_CAMERA_NAMES


def build_scene_xml(
    world: World | TrackWorld, bodies_xml: str = "", actuator_xml: str = "", equality_xml: str = ""
) -> str:
    """Scene MJCF. The optional fragments add bodies (e.g. rovers, see ``rover.rover_mjcf``) at the end of
    ``<worldbody>`` and the matching ``<actuator>`` / ``<equality>`` sections; all empty = the plain scene."""
    xml = _track_scene_xml(world) if isinstance(world, TrackWorld) else _sandbox_scene_xml(world)
    if bodies_xml:
        xml = xml.replace("</worldbody>", f"{bodies_xml}\n  </worldbody>", 1)
    extra = ""
    for tag, content in (("actuator", actuator_xml), ("equality", equality_xml)):
        if content:
            extra += f"  <{tag}>\n{content}\n  </{tag}>\n"
    return xml.replace("</mujoco>", f"{extra}</mujoco>", 1) if extra else xml


def generate(kind: str = "track", seed: int = 0, **options) -> World | TrackWorld:
    """Generate the world of ``kind`` ("track" or "sandbox"); ``options`` go to its generator."""
    if kind == "track":
        return generate_track_world(seed, **options)
    if kind == "sandbox":
        return generate_world(seed, **options)
    raise ValueError(f"unknown scene kind {kind!r}: expected 'track' or 'sandbox'")


def build_model_from_world(
    world: World | TrackWorld, bodies_xml: str = "", actuator_xml: str = "", equality_xml: str = ""
) -> mujoco.MjModel:
    assets = load_track_assets() if isinstance(world, TrackWorld) else load_assets()
    model = mujoco.MjModel.from_xml_string(build_scene_xml(world, bodies_xml, actuator_xml, equality_xml), assets)
    if isinstance(world, TrackWorld):
        for i, tile in enumerate(tt.plan_friction_tiles(world.terrain)):
            _write_hfield(model, f"terrain_tile_{i}", world.terrain.heights[:, tile.col0 : tile.col1 + 1])
        _write_hfield(model, "far_terrain", world.far.heights)
        for i, d in enumerate(tt.plan_sand_decals(world.sections)):
            _write_hfield(model, f"sand_decal_{i}", _decal_heights(world, d))
        for i, segment in enumerate(world.visual_terrain.segments):
            _write_hfield(model, f"terrain_visual_{i}", segment.heights)
        _write_hfield(model, "far_terrain_visual", world.visual_far.heights)
    else:
        _write_hfield(model, "terrain", world.heights)
        _write_hfield(model, "far_terrain", world.far_heights)
    return model


def build_model(seed: int = 0, kind: str = "track", **options) -> mujoco.MjModel:
    """The scene as a compiled model: the Jezero track, or ``kind="sandbox"`` for the square map."""
    return build_model_from_world(generate(kind, seed, **options))

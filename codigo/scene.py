"""
scene.py: arma la escena MuJoCo completa de Marte (terreno, rocas, cielo, luz y cámaras).

Para qué existe
    Es el modelo pensado para mirar: la pista de Jezero con texturas, iluminación, cámaras y terreno
    lejano. También define las piezas de la escena (gravedad, paso de simulación, fricción, rocas) que
    el modelo sin gráficos de headless.py copia para obtener exactamente la misma física.

Qué contiene
    - generate_track_world: genera el mundo de una semilla (relieve real, rugosidad y rocas).
    - TrackWorld: todo lo generado para una semilla, antes de convertirlo en MJCF.
    - build_scene_xml: escribe la escena en MJCF, con lugar para agregar rovers.
    - build_model y build_model_from_world: compilan la escena; con kind="sandbox" arman el mapa
      cuadrado procedural original.

Cómo funciona
    Todo es determinista por semilla y autocontenido: la textura del suelo se carga en memoria, así la
    escena funciona desde cualquier carpeta.

Dónde encaja
    Etapa: simulación. Usa track_terrain, track_rocks, track_sections, track_texture, mars_terrain y
    mars_params. Lo usan rover.py (para la escena con rovers), headless.py y evaluate.py.
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

DEFAULT_SIZE = 513
MARS_GRAVITY = -mp.MARS_GRAVITY_M_S2
TIMESTEP_S = 0.002
GROUND_FRICTION = (0.6, 0.005, 0.0001)
TILE_WORLD_WIDTH_M = 4.0
FAR_TILE_SCALE = 2.0
FOG_START, FOG_END = 1.0, 6.0

SHADOW_CLIP = 0.3

COLLISION_GEOM_GROUP = 3

SANDBOX_CAMERA_NAMES = ("overview", "ground", "top")
TRACK_CAMERA_NAMES = ("start", "overview", "side", "top")
CAMERA_NAMES = TRACK_CAMERA_NAMES
SECTION_CAMERA_PREFIX = "section_"

SKY_ZENITH = (0.40, 0.27, 0.20)
SKY_LOW = (0.98, 0.72, 0.50)
FOG_COLOR = tuple((a + b) / 2 for a, b in zip(SKY_ZENITH, SKY_LOW))
SUN_COLOR = (1.0, 0.82, 0.62)
SUN_DIRECTION = (-0.62, -0.42, -0.42)
ROCK_COLORS = {
    "basalt": (0.27, 0.20, 0.17),
    "rust": (0.48, 0.26, 0.16),
    "dust": (0.38, 0.29, 0.23),
    "ochre": (0.56, 0.37, 0.23),
}


@dataclass(frozen=True)
class World:
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
    buffer = io.BytesIO()
    Image.open(ASSETS_DIR / REGOLITH_TILE).convert("RGB").save(buffer, format="PNG")
    return {REGOLITH_ASSET_NAME: buffer.getvalue()}


GROUND_PRIORITY = 1


def _rgb(color: tuple[float, float, float]) -> str:
    return " ".join(f"{c:.4f}" for c in color)


def _look_at_axes(position: np.ndarray, target: np.ndarray) -> str:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return " ".join(f"{v:.5f}" for v in (*right, *up))


def _sandbox_camera_layout(world: World) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    h = world.heights
    ext = world.extent_m
    biggest = max(world.craters, key=lambda c: c.radius)
    dunes = world.dunes

    def ground_z(x: float, y: float) -> float:
        return mt.sample_height(h, ext, x, y)

    focus_xy = (np.array([biggest.x, biggest.y]) + np.array([dunes.x, dunes.y])) / 2 * 0.7
    sun_h = np.array(SUN_DIRECTION[:2]) / np.linalg.norm(SUN_DIRECTION[:2])
    away = np.array([-sun_h[1], sun_h[0]])
    if away @ -focus_xy < 0:
        away = -away
    overview_pos = np.array([*(focus_xy + away * 72.0), ground_z(*focus_xy) + 27.0])
    overview_target = np.array([*focus_xy, ground_z(*focus_xy)])

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
    tile_aspect = 224 / 416
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
    option = mujoco.MjvOption()
    hide_collision_geoms(option)
    return option


def hide_collision_geoms(option: mujoco.MjvOption) -> None:
    option.geomgroup[COLLISION_GEOM_GROUP] = 0


def _write_hfield(model: mujoco.MjModel, name: str, heights: np.ndarray) -> None:
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    adr = model.hfield_adr[hid]
    span = max(float(heights.max() - heights.min()), 1e-6)
    model.hfield_data[adr : adr + heights.size] = ((heights - heights.min()) / span).ravel()


TRACK_TILE_WORLD_WIDTH_M = 5.0
TRACK_TEXTURE_CELLS = (10, 5)
TRACK_FAR_TEXTURE_SCALE = 1.0
TRACK_TEXTURE_ASSET_NAME = "regolith_track.png"
TILE_ASPECT = 224 / 416
TRACK_FOG_START, TRACK_FOG_END = 2.5, 12.0
TRACK_SUN_DIRECTION = (0.55, 0.70, -0.50)
TRACK_SHADOW_CLIP = 0.3
MARKER_COLORS = {
    "start": (0.80, 0.74, 0.62),
    "finish": (0.55, 0.22, 0.15),
    "distance": (0.72, 0.45, 0.20),
}
SECTION_MARKER_COLORS = {
    "rock_field": (0.50, 0.44, 0.38),
    "ripples": (0.78, 0.60, 0.30),
    "soft_sand": (0.86, 0.78, 0.55),
    "bedrock": (0.36, 0.40, 0.46),
    "cross_slope": (0.32, 0.52, 0.50),
    "crater": (0.62, 0.30, 0.20),
    "climb": (0.82, 0.78, 0.70),
}
SAND_DECAL_LIFT_M = 0.03
SAND_DECAL_THICKNESS_M = 0.01
SAND_TINT_ASSET = "ground_sand_tile.jpg"
MARKER_SINK_M = 0.05
START_CAMERA_BACK_M = 4.0
START_CAMERA_HEIGHT_M = 2.2


@dataclass(frozen=True)
class TrackWorld:
    seed: int
    terrain: tt.TrackTerrain
    far: tt.FarTerrain
    rocks: list[mt.Rock]
    markers: list[tt.Marker]
    visual_terrain: tt.VisualTerrain
    visual_far: tt.FarTerrain

    @property
    def sections(self) -> tuple[ts.Section, ...]:
        return self.terrain.sections

    def section_at(self, x: float) -> ts.Section:
        return ts.section_at(x, self.sections)

    def section_weights(self, x) -> np.ndarray:
        return ts.section_weights(x, self.sections)


def generate_track_world(
    seed: int = 0,
    cell_m: float = mp.TRACK_CELL_M,
    far_pad_cells: int = tt.FAR_DEFAULT_PAD_CELLS,
    strip: tt.Strip | None = None,
    max_lane_rock_height_m: float | None = None,
) -> TrackWorld:
    strip = strip or tt.load_strip()
    terrain = tt.generate_track_heightmap(strip, cell_m=cell_m, seed=seed)
    far = tt.generate_far_terrain(strip, terrain, seed=seed, pad_cells=far_pad_cells)
    visual_terrain, visual_far = tt.make_visual_terrains(terrain, far)
    rocks, markers = tr.plan_track_rocks(terrain, seed, max_lane_rock_height_m), tt.plan_markers(terrain)
    return TrackWorld(seed, terrain, far, rocks, markers, visual_terrain, visual_far)


def lane_length_m(world: TrackWorld) -> float:
    return tt.finish_x(world.terrain) - mp.START_X_M


@lru_cache(maxsize=1)
def load_track_assets() -> dict[str, bytes]:
    tile = np.asarray(Image.open(ASSETS_DIR / REGOLITH_TILE).convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(ttex.make_track_texture(tile, cells=TRACK_TEXTURE_CELLS)).save(buffer, format="PNG")
    return {**load_assets(), TRACK_TEXTURE_ASSET_NAME: buffer.getvalue()}


SECTION_CAMERA_BACK_M = 24.0
SECTION_CAMERA_AHEAD_M = 18.0
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
        "start": (
            np.array([eye_x, 0.0, ground(eye_x) + START_CAMERA_HEIGHT_M]),
            np.array([ahead_x, 0.0, ground(ahead_x) + 0.8]),
        ),
        "overview": (
            np.array([-70.0, -70.0, ground(0.0) + 50.0]),
            np.array([170.0, 0.0, ground(170.0)]),
        ),
        "side": (
            np.array([mid_x - 45.0, -32.0, ground(mid_x - 45.0, -30.0) + 6.0]),
            np.array([mid_x, 0.0, ground(mid_x) + 0.5]),
        ),
        "top": (
            np.array([top_x, 0.0, ground(top_x) + 70.0]),
            np.array([top_x, 0.0, ground(top_x)]),
        ),
    }
    for section in world.sections:
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
    low, high = float(heights.min()), float(heights.max())
    return {
        "nrow": str(heights.shape[0]),
        "ncol": str(heights.shape[1]),
        "size": f"{half_x:.6f} {half_y:.6f} {max(high - low, 1e-6):.5f} {base_m}",
        "z": f"{low:.5f}",
    }


@lru_cache(maxsize=1)
def sand_tint_color() -> tuple[float, float, float]:
    mean = np.asarray(Image.open(ASSETS_DIR / SAND_TINT_ASSET).convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    return tuple(float(c) / 255.0 for c in mean)


def _grid_rect(terrain: tt.TrackTerrain, col0: int, col1: int, row0: int = 0, row1: int | None = None) -> dict[str, float]:
    n_rows, n_cols = terrain.heights.shape
    row1 = n_rows - 1 if row1 is None else row1
    cell_x = (terrain.x_max - terrain.x_min) / (n_cols - 1)
    cell_y = (terrain.y_max - terrain.y_min) / (n_rows - 1)
    return {
        "cx": terrain.x_min + 0.5 * (col0 + col1) * cell_x, "cy": terrain.y_min + 0.5 * (row0 + row1) * cell_y,
        "hx": 0.5 * (col1 - col0) * cell_x, "hy": 0.5 * (row1 - row0) * cell_y,
    }


def _collision_tiles_xml(world: TrackWorld) -> tuple[str, str]:
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
    terrain = world.terrain
    cell_x = (terrain.x_max - terrain.x_min) / (terrain.heights.shape[1] - 1)
    cell_y = (terrain.y_max - terrain.y_min) / (terrain.heights.shape[0] - 1)
    xs = np.linspace(decal.x0, decal.x1, max(2, round((decal.x1 - decal.x0) / cell_x) + 1))
    ys = np.linspace(-decal.half_width, decal.half_width, max(2, round(2 * decal.half_width / (3 * cell_y)) + 1))
    return world.visual_terrain.sample(*np.meshgrid(xs, ys)) + SAND_DECAL_LIFT_M


def _visual_terrain_xml(world: TrackWorld) -> tuple[str, str, str]:
    texture_w = TRACK_TEXTURE_CELLS[0] * TRACK_TILE_WORLD_WIDTH_M
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
    friction = f"{mp.SOIL_FRICTION} {mp.SPIN_FRICTION} {mp.ROLL_FRICTION}"
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


def camera_layout(world: World | TrackWorld) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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
    xml = _track_scene_xml(world) if isinstance(world, TrackWorld) else _sandbox_scene_xml(world)
    if bodies_xml:
        xml = xml.replace("</worldbody>", f"{bodies_xml}\n  </worldbody>", 1)
    extra = ""
    for tag, content in (("actuator", actuator_xml), ("equality", equality_xml)):
        if content:
            extra += f"  <{tag}>\n{content}\n  </{tag}>\n"
    return xml.replace("</mujoco>", f"{extra}</mujoco>", 1) if extra else xml


def generate(kind: str = "track", seed: int = 0, **options) -> World | TrackWorld:
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
    return build_model_from_world(generate(kind, seed, **options))

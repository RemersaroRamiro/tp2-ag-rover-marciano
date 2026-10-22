"""
track_texture.py: la textura del suelo de la pista, sin repeticiones visibles.

Para qué existe
    La foto de regolito de la NASA es una baldosa pequeña (5 m sobre la pista); repetida a lo largo de
    380 m, el patrón se nota. Este archivo arma con ella una textura mucho más grande que no se repite
    a simple vista. Es solo visual: no cambia la física.

Qué contiene
    - make_track_texture: arma la textura grande a partir de la baldosa, de forma determinista por semilla.

Cómo funciona
    Cada nodo de una grilla toma la baldosa con un desplazamiento al azar y cada píxel mezcla sus cuatro
    nodos vecinos. Encima se agregan manchas suaves de brillo y de tono.

Dónde encaja
    Etapa: simulación (escena, solo visual). No importa otros archivos del proyecto. Lo usa scene.py.

Fuentes
    - Foto PIA24627 (Mastcam-Z, NASA/JPL-Caltech/ASU/MSSS); ver codigo/assets/CREDITS.md.
"""

from __future__ import annotations

import numpy as np

_STREAM_SHIFTS = 104
_STREAM_MACRO = 105

BRIGHTNESS_VARIATION = 0.02
TINT_VARIATION = 0.01
MACRO_WAVELENGTH_MIN_TILES = 0.6


def _periodic_noise(rng: np.random.Generator, shape: tuple[int, int], wavelength_min_px: float) -> np.ndarray:
    height, width = shape
    spectrum = np.fft.rfft2(rng.standard_normal(shape))
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.rfftfreq(width)[None, :]
    k = np.hypot(kx, ky)
    with np.errstate(divide="ignore"):
        shaping = np.where(k > 0, 1.0 / k, 0.0) * np.exp(-((k * wavelength_min_px) ** 2))
    field = np.fft.irfft2(spectrum * shaping, s=shape)
    return (field - field.mean()) / field.std()


def make_track_texture(tile: np.ndarray, cells: tuple[int, int] = (10, 5), seed: int = 0) -> np.ndarray:
    tile_h, tile_w = tile.shape[:2]
    cells_x, cells_y = cells
    height, width = cells_y * tile_h, cells_x * tile_w
    rng = np.random.default_rng([seed, _STREAM_SHIFTS])
    shift_x = rng.integers(0, tile_w, size=(cells_y, cells_x))
    shift_y = rng.integers(0, tile_h, size=(cells_y, cells_x))

    mean = tile.reshape(-1, 3).mean(axis=0)
    detail = tile.astype(np.float32) - mean.astype(np.float32)
    xs, ys = np.arange(width), np.arange(height)
    u, v = xs / tile_w, ys / tile_h
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - i0).astype(np.float32), (v - j0).astype(np.float32)

    blended = np.zeros((height, width, 3), dtype=np.float32)
    norm = np.zeros((height, width), dtype=np.float32)
    for dj in (0, 1):
        for di in (0, 1):
            weight = (fv if dj else 1.0 - fv)[:, None] * (fu if di else 1.0 - fu)[None, :]
            node_j, node_i = ((j0 + dj) % cells_y)[:, None], ((i0 + di) % cells_x)[None, :]
            px = (xs[None, :] + shift_x[node_j, node_i]) % tile_w
            py = (ys[:, None] + shift_y[node_j, node_i]) % tile_h
            blended += weight[..., None] * detail[py, px]
            norm += weight**2
    image = mean.astype(np.float32) + blended / np.sqrt(norm)[..., None]

    macro = np.random.default_rng([seed, _STREAM_MACRO])
    wavelength_px = MACRO_WAVELENGTH_MIN_TILES * tile_w
    brightness = 1.0 + BRIGHTNESS_VARIATION * _periodic_noise(macro, (height, width), wavelength_px)
    tint = TINT_VARIATION * _periodic_noise(macro, (height, width), wavelength_px)
    image = image * brightness[..., None] * (1.0 + tint[..., None] * np.array([1.0, 0.0, -1.5], dtype=np.float32))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)

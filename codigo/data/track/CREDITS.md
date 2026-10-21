# Track elevation data credits

## jezero_floor_strip.npy / jezero_floor_strip.json

- **Product:** USGS Mars 2020 Terrain Relative Navigation (TRN) HiRISE DTM mosaic of Jezero crater
  (1 m/px, MOLA-tied, `JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif`).
- **Credit:** USGS Astrogeology Science Center; HiRISE imagery: NASA/JPL/University of Arizona.
- **What is committed:** one 400 x 61 pixel window (about 379.5 m x 61 m on the ground) along the
  2D demo's `jezero_floor` line, extracted by `tools/extract_track.py`. The full provenance (source
  URL, projected bounds, corner coordinates, pixel sizes, extraction date) is in the JSON next to it.
- **Note:** pixels are not square on the ground (0.9486 m in x, 1 m in y); the code always reads the
  true sizes from the JSON.

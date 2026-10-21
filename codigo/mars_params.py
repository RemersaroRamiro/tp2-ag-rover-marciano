"""Physical and modelling constants of the Jezero track, each with its source or a flag.

Legend used in the comments:
  [source]      value taken from a published measurement or model
  [calibrated]  free parameter tuned so that a measurable statistic matches the literature
  [assumption]  our own choice; no published value for this exact place / quantity
Pure constants, no imports: the README's realism table is generated from the same numbers.
"""

# -- environment ---------------------------------------------------------------------------------
MARS_GRAVITY_M_S2 = 3.71  # [source] standard Mars surface gravity
# [source, derived] rho = P / (R_specific * T) with the MEDA-measured ~735 Pa and 183-266 K.
# Irrelevant for rigid-body contact; recorded only. MuJoCo's fluid model is left off.
AIR_DENSITY_KG_M3 = 0.017

# -- real terrain: USGS Mars 2020 TRN HiRISE DTM ---------------------------------------------------
DTM_PIXEL_M = 1.0  # [source] product grid spacing (1 m/px)
DTM_VERTICAL_ERROR_M = 0.5  # [source] about 0.5 px (ISPRS 2020 evaluation of the product)
DTM_EFFECTIVE_RESOLUTION_M = (10.0, 20.0)  # [source] effective horizontal resolution, ~10-20 px (ISPRS 2020)
# [assumption, from the line above] Gaussian sigma of the low-pass applied to the DTM: features
# finer than the effective resolution are partly noise (max 1 m slopes of 21-28 degrees in this strip).
DTM_LOWPASS_SIGMA_M = 4.0
# [assumption] the border of the strip relaxes to a smoother (sigma 10 m) copy of the DTM so that it
# joins the far backdrop; also the sigma used to extend the DTM outwards.
DTM_BORDER_SIGMA_M = 10.0
# The terrain heightfield extends this far beyond the strip on every side. Over these margins the
# real-DTM detail (sigma 4 m) is blended into the smooth extension and the synthetic roughness fades
# to zero, so that the heightfield meets the coarse far backdrop exactly; inside the strip both are
# at full strength, so the roughness crosses the strip border without any visible change.
TERRAIN_MARGIN_M = 30.0

# -- sub-DTM roughness -----------------------------------------------------------------------------
HURST = 0.7  # [source] Mars HiRISE DTM statistics, +-0.1 (PSJ abbfac); rms dh(L) ~ L**HURST
# [source] rms slope at a 1 m baseline is 2-3 degrees (Kirk et al. 2008). Target: the middle of
# the range [calibrated]. Equivalent to rms dh(1 m) = tan(2.4 deg) = 0.042 m, i.e. the
# 0.035 * L**0.7 relation quoted for HiRISE, within its uncertainty.
ROUGHNESS_RMS_SLOPE_1M_DEG = 2.4
ROUGHNESS_RMS_SLOPE_1M_RANGE_DEG = (2.0, 3.0)
# [assumption] Gaussian roll-off of the synthetic spectrum below this wavelength: content at 2-4 cells
# per wave renders as a quilt of flat facets and is irrelevant to a 0.5 m wheel. The rms-slope
# calibration above is done on the rolled-off spectrum, so it still holds.
ROUGHNESS_CUTOFF_M = 0.5

# -- rocks: Golombek et al. cumulative fractional area model ---------------------------------------
ROCK_FRACTIONAL_AREA_K = 0.05  # [assumption] Jezero-specific k is not published; landing sites are 3-7 %
ROCK_HEIGHT_TO_DIAMETER = 0.4  # [source] typical rock height / diameter ratio
ROCK_EMBEDDED_FRACTION = 0.27  # [assumption] share of the ellipsoid's height buried in the ground
ROCK_D_MAX_M = 2.0  # [assumption] truncation of the size distribution (N(>=2 m) is ~1e-6 per m^2 anyway)
# [assumption] MuJoCo budget: at most ~6000 rock geoms (the render bench stays above 45 fps). Smaller rocks are not modelled as geoms
# (the fractal roughness above covers them). The diameter cut-off is lower inside the driving
# lane, where rocks matter for the rover, than in the flanks that only dress the scene.
MAX_ROCK_GEOMS = 6000
# Rocks continue beyond the lane and the strip (so the strip border is invisible): the minimum diameter rises smoothly from the lane's
# D_MIN_LANE to D_MIN_OUTSIDE between these |y| values, and outside the strip the density of the
# remaining boulders fades to zero over ROCK_FADE_OUT_M (the border of the strip is then invisible).
ROCK_GRADED_BAND_M = (10.0, 14.0)
ROCK_FADE_OUT_M = 30.0
ROCK_D_MIN_LANE_M = 0.17
ROCK_D_MIN_OUTSIDE_M = 0.45
# Reach of any evolvable rover from its chassis origin (``headless.design_reach_bound_m``, proven over the whole genome
# range by a test: the largest value is ~5.3 m). Used by the headless crop and by the unsurmountable-rock rule of the
# lane (``track_rocks.LANE_ROCK_BAND_M``).
ROVER_REACH_M = 6.0

# -- contact ------------------------------------------------------------------------------------
# [source] JEZ-1 simulant interface friction angles 18.9-36.2 degrees -> mu 0.35-0.75; mid value.
# Rigid-body contact cannot model sinkage or slip: real wheels on loose regolith do both.
SOIL_FRICTION = 0.5
ROCK_FRICTION = 0.7  # [assumption] wheel-rock (basalt-like) friction
SPIN_FRICTION = 0.005  # torsional and rolling friction: [assumption]
ROLL_FRICTION = 0.0001

# -- Perseverance, for scale only (no rover model yet) ---------------------------------------------
ROVER_LENGTH_M = 3.0  # [source] NASA/JPL Mars 2020 fact sheet (approximate body)
ROVER_WIDTH_M = 2.7
ROVER_HEIGHT_M = 2.2
ROVER_MASS_KG = 1025.0
ROVER_WHEEL_DIAMETER_M = 0.525
ROVER_TOP_SPEED_M_S = 0.042
ROVER_OBSTACLE_M = 0.4  # [source] obstacles the rover is meant to drive over
ROVER_TILT_LIMIT_DEG = 30.0

# -- track layout ---------------------------------------------------------------------------------
TRACK_CELL_M = 0.25  # default heightfield cell (1500 x 240 cells over the strip)
LANE_WIDTH_M = 20.0  # playable lane centred on the strip's centre row
START_X_M = 3.0  # start line; rovers spawn here and drive +x
FINISH_MARGIN_M = 3.0  # the finish line stands this far before the strip's far end
SPAWN_CLEAR_X_M = 6.0  # no rocks before this x
MARKER_SPACING_M = 50.0  # distance posts along the lane

# -- mixed track: named sections with gradual transitions (T10) --------------------------------------------
# Values with a citation come from the T9 research pass; "# ASSUMPTION (design choice)" marks shape details
# with no published value. Change them here only: nothing else in the code or the tests holds a copy.
#
# The strip is split into 8 sections along +x (floor, rock_field, ripples, soft_sand, bedrock,
# cross_slope, crater, climb). Each feature has a smooth weight (smoothstep over SECTION_TRANSITION_M,
# centred on the boundary), so nothing ever changes abruptly: fitness = distance always has a gradient.
SECTION_TRANSITION_M = 16.0  # ASSUMPTION (design choice): blend width; 10-20 m keeps every ramp gentle
# ASSUMPTION (design choice): x of the start of sections 2..8 (the first starts at 0, the last ends at
# the strip's end). Lengths 40 / 60 / 55 / 55 / 55 / 40 / 40 / 34 m, ordered from gentle to hard.
SECTION_BOUNDARIES_M = (40.0, 100.0, 155.0, 210.0, 265.0, 305.0, 345.0)

# rock_field: Golombek abundance ramps from ROCK_FRACTIONAL_AREA_K up to this value and back
ROCK_K_DENSE = 0.15  # Jezero rockiest areas CFA up to 15-20 % (ACE planner paper, arXiv:1808.00031, citing Golombek 2015)
ROCK_K_BEDROCK_FACTOR = 0.4  # ASSUMPTION (design choice): bedrock is swept clean: k = 5 % * 0.4
ROCK_D_MIN_DENSE_M = 0.20  # ASSUMPTION (design choice): lane cut-off diameter at ROCK_K_DENSE (budget)

# ripples: aeolian ripples / megaripples, asymmetric ridges roughly perpendicular to the lane
RIPPLE_HEIGHT_M = 0.18  # Maaz-floor bedforms 15-20 cm tall (Bell et al. 2022, Sci. Adv. 8, eabo4856, Fig. 4)
RIPPLE_WAVELENGTH_M = 4.0  # ASSUMPTION: no Jezero value found; Gale Hidden Valley ripples 3-4 m (Bretzfelder 2021)
RIPPLE_ASYMMETRY = 0.3  # ASSUMPTION (design choice): 0 = symmetric; larger = steeper lee face (+x side)
RIPPLE_OBLIQUITY_DEG = 8.0  # ASSUMPTION (design choice): crests deviate from perpendicular by up to this
RIPPLE_WARP_RAD = 0.7  # ASSUMPTION (design choice): waviness of the crest lines (phase noise amplitude)

# soft_sand: lower friction (per collision tile) plus smaller ripples
SOFT_SAND_FRICTION = 0.2  # ASSUMPTION (calibrated): effective mu 0.15-0.25 matching severe slip at ~4 deg (Curiosity,
# Hidden Valley) to ~20 deg (Perseverance rim, >94 % slip; JPL 2026 AAS paper). Rigid contact has no sinkage.
SAND_RIPPLE_FRACTION = 0.4  # ASSUMPTION (design choice): sand ripples are this fraction of RIPPLE_HEIGHT_M
FRICTION_TILE_M = 4.0  # collision hfield tile width in the friction zone: mu changes by small steps every tile
# Visual tint of the sand (the mean colour of assets/ground_sand_tile.jpg, blended in as translucent layers)
SAND_TINT_ALPHA = 0.5  # ASSUMPTION (design choice): opacity of the tint where the sand weight is 1

# bedrock: outcrop with low ledges; plates slightly tilted
BEDROCK_STEP_M = 0.30  # Maaz paver relief < 0.3 m, Seitah cap-rock steps 0.1-0.5 m (Bell et al. 2022)
BEDROCK_PLATE_LENGTH_M = (7.0, 13.0)  # ASSUMPTION (design choice): distance between ledges (min, max)
BEDROCK_LEDGE_EDGE_M = 1.0  # ASSUMPTION (design choice): horizontal width of a ledge face
BEDROCK_PLATE_TILT_DEG = 0.8  # ASSUMPTION (design choice): maximum tilt of a plate
BEDROCK_LEDGE_OBLIQUITY_DEG = 5.0  # ASSUMPTION (design choice): ledges deviate from perpendicular by up to this

# cross_slope: lateral tilt across the lane (positive: the +y side is higher)
CROSS_SLOPE_DEG = 12.0  # ASSUMPTION: no published cross-slope limit beyond the 30 deg tilt guard; typical slopes 5-15 deg

# crater: one small fresh crater centred on the lane (cosine bowl plus a gaussian rim)
CRATER_DIAMETER_M = 12.0  # within small-crater range 5-50 m (Daubar 2014); rim-to-rim diameter
# Degraded Jezero-floor craters d/D 0.023-0.067, fresh ~0.2 (Warner LPSC 2020 #1608; Daubar 2014). 0.15 would give 25 degree
# walls that, added to the local terrain slope, exceed the rover's 30 degree tilt limit; 0.10 keeps it below it.
CRATER_DEPTH_TO_DIAMETER = 0.10
CRATER_RIM_HEIGHT_TO_DIAMETER = 0.02  # below Warner 2020's ~0.032 (kept low for the tilt limit); rim crest / D
CRATER_RIM_WIDTH_FRACTION = 0.4  # ASSUMPTION (design choice): gaussian rim width / crater radius
CRATER_CENTER_Y_M = 0.0  # ASSUMPTION (design choice): the crater is centred on the lane

# climb: final longitudinal slope (crater rim / delta front); the slope ramps in over SECTION_TRANSITION_M
CLIMB_TOP_EASE_M = 20.0  # ASSUMPTION (design choice): the slope eases to zero over this distance after the strip's end
# ASSUMPTION (design choice): the scarp line is not straight: where the rise starts wanders by up to about this
# many meters along y (zero inside the lane, so the lane profile is exact). Real rim / delta scarps are regional.
CLIMB_SCARP_WANDER_M = 12.0
# ASSUMPTION (design choice): away from the lane the same rise is spread over this many times the distance (gentler
# slope, same total gain): 1 in the lane, fully broadened beyond ~160 m off it. The far backdrop is drawn with coarse
# flat-shaded cells, on which a steep scarp would show as a checkerboard.
CLIMB_SCARP_BROADENING = 4.0
CLIMB_SLOPE_DEG = 18.0  # Jezero rim climb (Summerland Trail) 15-20 deg, max traversed ~23 deg (JPL 2026 AAS paper)

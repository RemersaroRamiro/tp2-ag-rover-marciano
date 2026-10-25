"""
mars_params.py: las constantes físicas y de modelado de la pista de Jezero.

Para qué existe
    Reúne en un solo lugar todos los valores del entorno marciano y de la pista, cada uno con su fuente
    o marcado como supuesto. Ningún otro archivo guarda una copia de estos números.

Qué contiene
    - Entorno y terreno real: gravedad, suavizado del DTM y el margen de 30 m que rodea a la franja real.
    - Rugosidad y rocas: exponente de Hurst, pendiente rms, modelo de rocas de Golombek y
      ROVER_REACH_M = 6 m, una cota del alcance de cualquier rover desde su centro (máximo ~5.3 m).
    - Contacto y trazado de la pista: celda de 0.25 m, carril de 20 m, salida en x = 3 m y meta 3 m
      antes del final de la franja.
    - Secciones de la pista: límites, transiciones y la forma de cada sección (ondulaciones, arena
      blanda, roca firme, pendiente lateral, cráter y subida final).

Dónde encaja
    Etapa: simulación. No importa otros archivos del proyecto. Lo usan track_terrain, track_features,
    track_rocks, track_sections, headless, evaluate y rover_run.

Fuentes
    Valores medidos o publicados:
    - MARS_GRAVITY_M_S2 = 3.71: gravedad estándar en la superficie de Marte.
    - El relieve es el DTM HiRISE Mars 2020 TRN del USGS: 1 m/px, error vertical ~0.5 px y resolución
      efectiva de 10-20 px (evaluación ISPRS 2020).
    - HURST = 0.7: estadística de los DTM HiRISE de Marte, ±0.1 (PSJ abbfac).
    - ROUGHNESS_RMS_SLOPE_1M_DEG = 2.4: la pendiente rms a 1 m es de 2-3° (Kirk et al. 2008); se calibró
      al centro del rango, lo que equivale a la relación 0.035 · L^0.7 citada para HiRISE.
    - ROCK_HEIGHT_TO_DIAMETER = 0.4: relación típica entre altura y diámetro de una roca.
    - ROCK_K_DENSE = 0.15: las zonas más rocosas de Jezero llegan a 15-20 % (artículo del planificador
      ACE, arXiv:1808.00031, que cita a Golombek 2015).
    - SOIL_FRICTION = 0.5: ángulos de fricción del simulante JEZ-1 de 18.9-36.2° (μ 0.35-0.75), valor
      medio. El contacto rígido no modela ni hundimiento ni patinaje.
    - RIPPLE_HEIGHT_M = 0.18 y BEDROCK_STEP_M = 0.30: formas del piso de Maaz de 15-20 cm, relieve de las
      losas de Maaz < 0.3 m y escalones de Seitah de 0.1-0.5 m (Bell et al. 2022, Sci. Adv. 8, eabo4856).
    - CRATER_DIAMETER_M = 12: dentro del rango de cráteres pequeños de 5-50 m (Daubar 2014).
    - CRATER_DEPTH_TO_DIAMETER = 0.10 y CRATER_RIM_HEIGHT_TO_DIAMETER = 0.02: los cráteres degradados
      del piso de Jezero tienen d/D de 0.023-0.067 y los frescos ~0.2 (Warner, LPSC 2020 #1608; Daubar
      2014); el borde que indica Warner 2020 es ~0.032. Se usan valores bajos: 0.15 daría paredes de 25°
      que, sumadas a la pendiente local, superarían el límite de inclinación de 30° del rover.
    - CLIMB_SLOPE_DEG = 18: subida al borde de Jezero (Summerland Trail) de 15-20°, máximo recorrido
      ~23° (JPL, artículo AAS 2026).
    Valores calibrados:
    - SOFT_SAND_FRICTION = 0.2: μ efectivo de 0.15-0.25, que reproduce el patinaje severo observado
      desde ~4° (Curiosity, Hidden Valley) hasta ~20° (Perseverance en el borde, más de 94 % de
      patinaje; JPL, artículo AAS 2026). El contacto rígido no modela hundimiento.
    Supuestos propios (no hay un valor publicado para este lugar o esta magnitud):
    - DTM_LOWPASS_SIGMA_M = 4 (deducido de la resolución efectiva; los detalles más finos son en parte
      ruido), DTM_BORDER_SIGMA_M = 10 y ROUGHNESS_CUTOFF_M = 0.5.
    - ROCK_FRACTIONAL_AREA_K = 0.05: el k de Jezero no está publicado; los sitios de aterrizaje tienen 3-7 %.
    - ROCK_EMBEDDED_FRACTION = 0.27 y ROCK_D_MAX_M = 2. Para que el modelo MuJoCo no pase de unas 6000
      rocas, las más pequeñas que ROCK_D_MIN_LANE_M (dentro del carril) o ROCK_D_MIN_OUTSIDE_M (fuera de
      él) no se modelan.
    - ROCK_FRICTION = 0.7 (rueda contra roca tipo basalto), SPIN_FRICTION y ROLL_FRICTION.
    - RIPPLE_WAVELENGTH_M = 4: no se encontró un valor de Jezero; las ondulaciones de Hidden Valley, en
      Gale, miden 3-4 m (Bretzfelder 2021).
    - CROSS_SLOPE_DEG = 12: no hay un límite publicado de pendiente lateral más allá de la inclinación
      de 30°; las pendientes típicas son de 5-15°.
    - Decisiones de diseño: SECTION_TRANSITION_M, SECTION_BOUNDARIES_M, ROCK_K_BEDROCK_FACTOR,
      ROCK_D_MIN_DENSE_M, RIPPLE_ASYMMETRY, RIPPLE_OBLIQUITY_DEG, RIPPLE_WARP_RAD, SAND_RIPPLE_FRACTION,
      BEDROCK_PLATE_LENGTH_M, BEDROCK_LEDGE_EDGE_M, BEDROCK_PLATE_TILT_DEG, BEDROCK_LEDGE_OBLIQUITY_DEG,
      CRATER_RIM_WIDTH_FRACTION, CRATER_CENTER_Y_M, CLIMB_TOP_EASE_M, CLIMB_SCARP_WANDER_M y
      CLIMB_SCARP_BROADENING.
"""

MARS_GRAVITY_M_S2 = 3.71

DTM_LOWPASS_SIGMA_M = 4.0
DTM_BORDER_SIGMA_M = 10.0
TERRAIN_MARGIN_M = 30.0

HURST = 0.7
ROUGHNESS_RMS_SLOPE_1M_DEG = 2.4
ROUGHNESS_CUTOFF_M = 0.5

ROCK_FRACTIONAL_AREA_K = 0.05
ROCK_HEIGHT_TO_DIAMETER = 0.4
ROCK_EMBEDDED_FRACTION = 0.27
ROCK_D_MAX_M = 2.0
ROCK_GRADED_BAND_M = (10.0, 14.0)
ROCK_FADE_OUT_M = 30.0
ROCK_D_MIN_LANE_M = 0.17
ROCK_D_MIN_OUTSIDE_M = 0.45
ROVER_REACH_M = 6.0

SOIL_FRICTION = 0.5
ROCK_FRICTION = 0.7
SPIN_FRICTION = 0.005
ROLL_FRICTION = 0.0001


TRACK_CELL_M = 0.25
LANE_WIDTH_M = 20.0
START_X_M = 3.0
FINISH_MARGIN_M = 3.0
SPAWN_CLEAR_X_M = 6.0

SECTION_TRANSITION_M = 16.0
SECTION_BOUNDARIES_M = (40.0, 100.0, 155.0, 210.0, 265.0, 305.0, 345.0)

ROCK_K_DENSE = 0.15
ROCK_K_BEDROCK_FACTOR = 0.4
ROCK_D_MIN_DENSE_M = 0.20

RIPPLE_HEIGHT_M = 0.18
RIPPLE_WAVELENGTH_M = 4.0
RIPPLE_ASYMMETRY = 0.3
RIPPLE_OBLIQUITY_DEG = 8.0
RIPPLE_WARP_RAD = 0.7

SOFT_SAND_FRICTION = 0.2
SAND_RIPPLE_FRACTION = 0.4
FRICTION_TILE_M = 4.0

BEDROCK_STEP_M = 0.30
BEDROCK_PLATE_LENGTH_M = (7.0, 13.0)
BEDROCK_LEDGE_EDGE_M = 1.0
BEDROCK_PLATE_TILT_DEG = 0.8
BEDROCK_LEDGE_OBLIQUITY_DEG = 5.0

CROSS_SLOPE_DEG = 12.0

CRATER_DIAMETER_M = 12.0
CRATER_DEPTH_TO_DIAMETER = 0.10
CRATER_RIM_HEIGHT_TO_DIAMETER = 0.02
CRATER_RIM_WIDTH_FRACTION = 0.4
CRATER_CENTER_Y_M = 0.0

CLIMB_TOP_EASE_M = 20.0
CLIMB_SCARP_WANDER_M = 12.0
CLIMB_SCARP_BROADENING = 4.0
CLIMB_SLOPE_DEG = 18.0

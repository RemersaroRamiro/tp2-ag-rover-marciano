"""
rover_params.py: las constantes del rover (geometría, masas, restricciones, control y energía).

Para qué existe
    Reúne en un solo lugar todos los valores fijos del rover que el AG no evoluciona, cada uno con su
    fuente o marcado como supuesto. Ningún otro archivo los repite: cambiar un valor aquí es suficiente.

Qué contiene
    - Reparación de la geometría: separaciones mínimas entre ruedas, chasis y brazos de la suspensión.
    - Modelo de masas: densidades y masas específicas con las que cada pieza obtiene su masa a partir
      de los genes, más la carga útil fija.
    - Restricciones de la misión: masa máxima, ángulo mínimo de vuelco, volumen plegado y despeje.
    - Modelo MuJoCo: recorrido y amortiguación de las articulaciones, motores y filtro de colisiones
      (los rovers chocan con el terreno y las rocas, nunca entre sí ni consigo mismos).
    - Salida, controlador (50 Hz), energía (RTG y batería) y fin de corrida (vuelco y rover atascado).

Dónde encaja
    Etapa: simulación. No importa otros archivos del proyecto.
    Lo usan rover, controller, energy, rover_run y evaluate.

Fuentes
    Valores medidos, publicados o calibrados (investigación de constructibilidad T13):
    - ALUMINUM_DENSITY_KG_M3 = 2700: propiedad de las aleaciones de aluminio.
    - WHEEL_SHELL_THICKNESS_M = 0.0053: espesor efectivo que agrupa piel, garras, llanta, cubo y
      flexores, calibrado para que una rueda del tamaño de la de Perseverance (D 0.523 m, ancho 0.336 m;
      Rankin et al. 2022, JPL, Tabla 1) pese ~9.4 kg (fuente secundaria, sin verificar).
    - MOTOR_SPECIFIC_TORQUE_NM_PER_KG = 30: los actuadores espaciales dan de 24-26 N·m/kg (planetarios)
      a 49-76 (harmonic drive), IDD de MER (Fleischner, ESMATS 2003); 30 es un valor reducido por freno,
      rodamientos y encoder, elegido dentro de ese rango (supuesto).
    - BATTERY_SPECIFIC_ENERGY_WH_PER_KG = 83: batería de Perseverance, 2408 Wh en un paquete de 29 kg
      (West, Smart y Herman, JPL 2021).
    - CHASSIS_AREAL_DENSITY_KG_M2 = 7: panel de panal de aluminio de 25 mm, 5.4 kg/m² (datos de
      fabricante) × 1.3 por insertos y refuerzos.
    - PAYLOAD_MASS_KG = 790: todo lo que no sale de los genes (MMRTG ~40-45 kg, carga científica 59 kg,
      brazo y torreta, aviónica, control térmico, cableado, mástil, telecomunicaciones); calibrado para
      que el rover tipo Perseverance sume ~1025 kg, la masa real de Mars 2020 (sus partes modeladas
      pesan ~234 kg).
    - MAX_TOTAL_MASS_KG = 1050: techo blando del sistema de entrada y descenso de MSL/M2020 (MSL: requisito
      de 850 kg, voló con 899 kg; M2020 aterrizó con 1025 kg).
    - MIN_TIPOVER_DEG = 45: requisito de MER de 45° de estabilidad estática en cabeceo y en rolido
      (Harrington y Voorhees, ESMATS 2004).
    - TILT_REF_DEG = 20: inclinación con la que el freno en pendientes actúa por completo; la pendiente
      más empinada de la pista es la subida final de 18° (JPL 2026, ver mars_params).
    - RTG_POWER_W = 110: potencia eléctrica del MMRTG al inicio de su vida útil (decae ~4 % por año, se
      ignora).
    - DRIVE_EFFICIENCY = 0.65: eficiencia de eléctrica a mecánica en la rueda (motor, caja reductora y
      electrónica), estimada en 0.5-0.8; la fuente primaria no está registrada, así que es una estimación.
    - BATTERY_MAX_DEPTH_OF_DISCHARGE = 0.60: parte usable de la capacidad, límite de vida útil por
      ciclos de Mars 2020.
    Supuestos propios (no hay un valor publicado):
    - MAX_ENVELOPE_M = (3.0, 2.7, 2.2): el volumen plegado no está publicado; se usa como límite el de
      Perseverance desplegado (~2.9 × 2.7 × 2.2 m, dentro de un escudo de 4.5 m).
    - MIN_GROUND_CLEARANCE_M = 0.30: MER exigía superar obstáculos de 0.20 m; Perseverance supera rocas
      de ~0.40 m.
    - HOTEL_LOAD_W = 100: aviónica, control térmico y sensores; a propósito por debajo del RTG, así un
      rover detenido nunca descarga la batería: solo se descarga al manejar.
    - ARM_LINEAR_DENSITY_KG_M = 8 y WHEEL_SIDE_AREA_FRACTION = 0.25.
    - HEADING_KP_MPS_PER_RAD, HEADING_KD_MPS_PER_RAD_S y LANE_KP_RAD_PER_M: ganancias ajustadas en terreno
      plano (con LANE_KP = 0.10, un desvío de 3 m se corrige en ~20 s a 0.5 m/s; con 0.04 hacían falta 50 s).
    - Decisiones de diseño: MIN_WHEEL_GAP_M, CHASSIS_SIDE_GAP_M, BOGIE_PIVOT_HEIGHT_M, ROCKER_RANGE_DEG,
      BOGIE_RANGE_DEG, CONTROL_DT_S, MAX_TURN_DIFF_MPS, LANE_MAX_HEADING_RAD, MIN_SPEED_FRACTION,
      TIP_LIMIT_DEG, TIP_TIME_S, STUCK_PROGRESS_M y STUCK_WINDOW_S.
"""

from __future__ import annotations

MIN_WHEEL_GAP_M = 0.05
CHASSIS_SIDE_GAP_M = 0.05
MIN_ARM_REACH_M = 0.15
BOGIE_PIVOT_HEIGHT_M = 0.15
MIN_CHASSIS_BOTTOM_ABOVE_AXLE_M = 0.05
ARM_TUBE_RADIUS_M = 0.04

ALUMINUM_DENSITY_KG_M3 = 2700.0
WHEEL_SHELL_THICKNESS_M = 0.0053
WHEEL_SIDE_AREA_FRACTION = 0.25
MOTOR_SPECIFIC_TORQUE_NM_PER_KG = 30.0
BATTERY_SPECIFIC_ENERGY_WH_PER_KG = 83.0
CHASSIS_AREAL_DENSITY_KG_M2 = 7.0
ARM_LINEAR_DENSITY_KG_M = 8.0
PAYLOAD_MASS_KG = 790.0
PAYLOAD_HEIGHT_FRACTION = 0.35
BATTERY_HEIGHT_FRACTION = 0.20
MAX_LOAD_OFFSET_FRACTION = 0.45
PAYLOAD_SIZE_FRACTION = 0.6

MAX_TOTAL_MASS_KG = 1050.0
MIN_TIPOVER_DEG = 45.0
MAX_ENVELOPE_M = (3.0, 2.7, 2.2)
MIN_GROUND_CLEARANCE_M = 0.30

ROCKER_RANGE_DEG = 40.0
BOGIE_RANGE_DEG = 45.0
ROCKER_DAMPING_NMS = 20.0
BOGIE_DAMPING_NMS = 10.0
WHEEL_DAMPING_NMS = 0.5
WHEEL_ARMATURE_KG_M2 = 0.05
DRIVE_SATURATION_RAD_S = 1.0
MAX_WHEEL_SPEED_MPS = 2.0
WHEEL_FRICTION = (1.0, 0.005, 0.0001)
ROVER_CONTYPE = 2
ROVER_CONAFFINITY = 1
ROVER_GROUP = 1

SPAWN_X_M = 4.0
SPAWN_CLEARANCE_M = 0.06

CONTROL_DT_S = 0.02
HEADING_KP_MPS_PER_RAD = 1.2
HEADING_KD_MPS_PER_RAD_S = 0.6
MAX_TURN_DIFF_MPS = 0.4
LANE_KP_RAD_PER_M = 0.10
LANE_MAX_HEADING_RAD = 0.35
TILT_REF_DEG = 20.0
MIN_SPEED_FRACTION = 0.25

RTG_POWER_W = 110.0
DRIVE_EFFICIENCY = 0.65
BATTERY_MAX_DEPTH_OF_DISCHARGE = 0.60
HOTEL_LOAD_W = 100.0

TIP_LIMIT_DEG = 60.0
TIP_TIME_S = 1.0
STUCK_PROGRESS_M = 0.2
STUCK_WINDOW_S = 20.0
TRAJECTORY_LOG_DT_S = 1.0

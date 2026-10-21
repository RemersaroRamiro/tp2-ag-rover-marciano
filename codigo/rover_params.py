"""Every constant of the rover decoder in one place (SI units).

Values come from the T13 constructibility research (cited inline); the ones with no published value carry
``# ASSUMPTION``. Nothing else in the code base hard-codes
them, and the tests read them from here, so changing a value here is enough.
"""

from __future__ import annotations

# -- geometry repair (the decoder makes every genome buildable, see ``rover.design_from_genome``) --------
MIN_WHEEL_GAP_M = 0.05  # ASSUMPTION (design choice): minimum gap between neighbouring wheel rims along x
CHASSIS_SIDE_GAP_M = 0.05  # ASSUMPTION (design choice): gap between a chassis side and the wheel inner face
MIN_ARM_REACH_M = 0.15  # horizontal reach below which a rocker arm would be a vertical strut; arms are lengthened to it
BOGIE_PIVOT_HEIGHT_M = 0.15  # ASSUMPTION (design choice): bogie pivot height above the axle line
MIN_CHASSIS_BOTTOM_ABOVE_AXLE_M = 0.05  # the chassis underside never drops below the axle line plus this
ARM_TUBE_RADIUS_M = 0.04  # tube radius used for the inertia of the suspension arms (and for drawing them)

# -- mass model (every body's mass is derived from the genes; the payload is fixed) ---------------------
ALUMINUM_DENSITY_KG_M3 = 2700.0  # material property of aluminium alloys
WHEEL_SHELL_THICKNESS_M = 0.0053  # EFFECTIVE thickness lumping skin, grousers, rims, hub and flexures: calibrated so a
# Perseverance-size wheel (D 0.523 m, W 0.336 m; Rankin et al. 2022, JPL, Table 1) weighs ~9.4 kg (secondary source, unverified)
WHEEL_SIDE_AREA_FRACTION = 0.25  # ASSUMPTION (design choice): share of the two side discs that is solid
MOTOR_SPECIFIC_TORQUE_NM_PER_KG = 30.0  # space-rated actuators 24-26 N*m/kg (planetary) to 49-76 (harmonic), MER IDD
# (Fleischner, ESMATS 2003); 30 derated for brake, bearings and encoder (assumption within that range)
BATTERY_SPECIFIC_ENERGY_WH_PER_KG = 83.0  # Perseverance battery: 2408 Wh / 29 kg pack (West, Smart, Herman, JPL 2021)
CHASSIS_AREAL_DENSITY_KG_M2 = 7.0  # 25 mm aluminium honeycomb panel 5.4 kg/m^2 (vendor data) x 1.3 for inserts/doublers
ARM_LINEAR_DENSITY_KG_M = 8.0  # ASSUMPTION (design choice): no published value for rocker/bogie tubes
PAYLOAD_MASS_KG = 790.0  # everything NOT derived from genes (MMRTG ~40-45 kg, science payload 59 kg, arm/turret,
# avionics, thermal, harness, mast, telecom): calibrated so the Perseverance-like genome totals ~1025 kg, the real
# Mars 2020 rover mass (the modelled parts of that design weigh ~234 kg)
PAYLOAD_HEIGHT_FRACTION = 0.35  # payload centre above the chassis underside, as a fraction of the chassis height
BATTERY_HEIGHT_FRACTION = 0.20  # battery centre above the chassis underside, as a fraction of the chassis height
MAX_LOAD_OFFSET_FRACTION = 0.45  # payload and battery stay within this fraction of the chassis length from its centre
PAYLOAD_SIZE_FRACTION = 0.6  # the payload lump fills this fraction of the chassis length and width (for its inertia)

# -- mission constraints (``rover.check_constraints``; thresholds from the Mars 2020 class of rover) ------
MAX_TOTAL_MASS_KG = 1050.0  # soft ceiling of the MSL/M2020 EDL system: MSL req. 850 kg, flew 899 kg; M2020 landed 1025 kg
MIN_TIPOVER_DEG = 45.0  # MER requirement: 45 deg static stability in pitch and roll (Harrington & Voorhees, ESMATS 2004)
MAX_ENVELOPE_M = (3.0, 2.7, 2.2)  # ASSUMPTION: stowed envelope not published; Perseverance deployed ~2.9 x 2.7 x 2.2 m
# inside a 4.5 m aeroshell, used as the limit (length, width, height)
MIN_GROUND_CLEARANCE_M = 0.30  # ASSUMPTION: MER required 0.20 m obstacles; Perseverance handles ~0.40 m rocks

# -- simulation model (MJCF) ------------------------------------------------------------------------
ROCKER_RANGE_DEG = 40.0  # ASSUMPTION (design choice): rocker travel about its pivot, each way
BOGIE_RANGE_DEG = 45.0  # ASSUMPTION (design choice): bogie travel about its pivot, each way
ROCKER_DAMPING_NMS = 20.0  # joint damping, keeps the free hinges from ringing
BOGIE_DAMPING_NMS = 10.0
WHEEL_DAMPING_NMS = 0.5  # rolling drag of the drive train
WHEEL_ARMATURE_KG_M2 = 0.05  # rotor inertia reflected to the wheel through the gearbox
DRIVE_SATURATION_RAD_S = 1.0  # the velocity servo saturates at the peak torque for a speed error of this many rad/s
MAX_WHEEL_SPEED_MPS = 2.0  # control range of a drive actuator, as a rim speed
WHEEL_FRICTION = (1.0, 0.005, 0.0001)  # the terrain has contact priority, so its friction is the one applied
# Collision filtering. MuJoCo: two geoms collide when (contype1 & conaffinity2) or (contype2 & conaffinity1).
# The scene's terrain and rocks keep the defaults (contype 1, conaffinity 1); a rover geom has contype 2 and
# conaffinity 1, so rover vs terrain/rocks collides (terrain.contype & rover.conaffinity) while rover vs rover
# never does (2 & 1 == 0 both ways), and neither do the parts of one rover.
ROVER_CONTYPE = 2
ROVER_CONAFFINITY = 1
ROVER_GROUP = 1  # drawn by default in the viewer (groups 0-2)

# -- placing a rover on the track ----------------------------------------------------------------------
SPAWN_X_M = 4.0  # ahead of the start line (x = 3 m), inside the first section
SPAWN_CLEARANCE_M = 0.06  # wheels start this far above the highest ground under the rover and settle by themselves

# -- controller (``controller.py``): heading hold toward +x, cruise speed, slope brake --------------------
# All of these are design choices of the fixed, explainable controller (not evolved, not sourced): the three
# evolved controller genes are ``target_speed_mps``, ``torque_limit_fraction`` and ``slope_brake``.
CONTROL_DT_S = 0.02  # ASSUMPTION (design choice): the controller runs at 50 Hz; energy is integrated every physics step
HEADING_KP_MPS_PER_RAD = 1.2  # ASSUMPTION (tuned on flat ground): wheel-speed difference (half) per radian of yaw error
HEADING_KD_MPS_PER_RAD_S = 0.6  # ASSUMPTION (tuned on flat ground): damping on the yaw rate
MAX_TURN_DIFF_MPS = 0.4  # ASSUMPTION (design choice): the left/right rim speeds differ by at most twice this
LANE_KP_RAD_PER_M = 0.10  # ASSUMPTION (tuned on flat ground): heading offset asked per metre off the lane centre (a 3 m
# offset is closed in ~1/(v*K) = 20 s at 0.5 m/s; 0.04 needed 50 s)
LANE_MAX_HEADING_RAD = 0.35  # ASSUMPTION (design choice): the lateral term never asks for more than ~20 deg off +x
TILT_REF_DEG = 20.0  # tilt at which the slope brake is fully applied: the steepest designed slope of the track is the
# 18 deg final climb (JPL 2026, see ``mars_params``)
MIN_SPEED_FRACTION = 0.25  # ASSUMPTION (design choice): the slope brake never slows the rover below this share of its target

# -- energy (``energy.py``): RTG + battery, no regeneration ---------------------------------------------
RTG_POWER_W = 110.0  # MMRTG electrical output at beginning of life (T13 research; it decays ~4 %/yr, ignored)
DRIVE_EFFICIENCY = 0.65  # electrical -> mechanical at the wheel (motor + gearbox + drive electronics): T13 research
# estimate, range 0.5-0.8; the primary source is not recorded in the repository, so treat it as an estimate
BATTERY_MAX_DEPTH_OF_DISCHARGE = 0.60  # usable share of the capacity: Mars 2020 cycle-life limit (T13 research)
HOTEL_LOAD_W = 100.0  # ASSUMPTION: avionics, thermal control and sensors while awake; below the RTG output on purpose,
# so an idle rover never drains the battery (only driving does). No published value for this model.

# -- run tracking (``rover_run.py``) ------------------------------------------------------------------------
TIP_LIMIT_DEG = 60.0  # ASSUMPTION (design choice): |roll| or |pitch| beyond this for TIP_TIME_S means the rover tipped over
TIP_TIME_S = 1.0  # ASSUMPTION (design choice)
STUCK_PROGRESS_M = 0.2  # ASSUMPTION (design choice): less progress than this in STUCK_WINDOW_S means stuck
STUCK_WINDOW_S = 20.0  # ASSUMPTION (design choice)
TRAJECTORY_LOG_DT_S = 1.0  # one trajectory row per simulated second

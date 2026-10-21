"""Every constant of the evolution loop in one place: fitness offset, recording, validation (SI units).

Nothing here is a physical quantity of the rover (see ``rover_params``) nor a rule of the simulation (see
``evaluate.EvalConfig``); these are the experiment's own choices, and the ones a report has to state.
"""

from __future__ import annotations

# -- fitness (``fitness.py``) --------------------------------------------------------------------------------------
INFEASIBLE_OFFSET = 1.0  # Deb (2000) feasibility rules folded into one scalar (see ``fitness``): a legal design scores its
# distance in metres (>= 0); an illegal one scores -(INFEASIBLE_OFFSET + total violation) < 0, where the total violation is
# the sum of the RELATIVE overshoots of the broken mission constraints (``rover.Violation.magnitude`` = (value - limit) /
# limit, dimensionless). Any positive value gives the same ordering (every illegal design below every legal one, illegal
# ones by smaller violation first, legal ones by larger distance first); 1.0 just keeps the illegal scores readable.
# Replaces the former soft penalty PENALTY_M_PER_UNIT = 500 m per unit of violation (retired: champions sat 4-7 % over
# the envelope because the penalty was worth paying; see the T18 results).

# -- seeds (``seed_schedule.py``) ----------------------------------------------------------------------------------
TERRAIN_SEED_LIMIT = 2**31 - 1  # terrain seeds are drawn from [1, this)
SCHEDULE_TAG = 0x52_4F_56_33  # mixed with the run seed so the terrain seeds are a stream apart from the GA's own rng
VALIDATION_SEEDS = 5  # unseen terrain seeds every run is validated on at the end
VALIDATION_GENOMES = 3  # how many of the final population's best genomes are validated

# -- success criterion (partial delivery, section 6.3) --------------------------------------------------------------
SUCCESS_MIN_FINISHES = 3  # a run succeeds when its best design is legal and finishes on at least 3 ...
SUCCESS_OF_SEEDS = 5  # ... of 5 validation terrains; with another number of validation terrains the same share is
# required, rounded up (``evolve.required_finishes``)

# -- recording (``replay.py``) -------------------------------------------------------------------------------------
REPLAY_HZ = 10.0  # frames per simulated second, per rover
REPLAY_MAX_FRAMES = 6000  # per rover: longer runs are thinned by a whole factor. A finisher needs ~590 s = 5900 frames
# at 10 Hz, so finishers keep the full rate; only a rover that keeps creeping along for the 1556 s cap is thinned (to
# ~4 Hz). This bounds a generation: 50 rovers x 6000 frames = 21.6 MB raw, 7.7 MB stored (measured with
# real recorded frames, JSON included; see the storage note in ``replay``), the worst case of any preset. Typical: 0.3-0.8 MB
# for a 12-rover generation.
REPLAY_BEST_K = 5  # rovers per generation recorded by the "best" mode

"""Central configuration for the CS2 cheat-detection experiments."""
from pathlib import Path

# --- Dataset -----------------------------------------------------------------
HF_REPO_ID = "CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection"
# Only matches with at least one VAC-banned player were manually labelled,
# so the case study uses this partition exclusively.
HF_SUBSET = "with_cheater_present"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Game constants ----------------------------------------------------------
TICK_RATE = 64                 # CS2 demos in CS2CD are recorded at 64 ticks/s
EYE_HEIGHT = 64.0              # approximate standing eye height (game units)
TARGET_HEIGHT = 40.0           # approximate chest height of the victim

# --- Kill window ---------------------------------------------------------------
WINDOW_BEFORE_TICKS = 64       # 1.0 s of aim before the kill
WINDOW_AFTER_TICKS = 16        # 0.25 s after the kill
SPOT_LOOKBACK_TICKS = 192      # 3.0 s to search for the first "spotted" tick

# Weapons whose kills carry no aiming information.
NON_AIM_WEAPONS = {
    "knife", "knife_t", "bayonet", "hegrenade", "inferno", "molotov",
    "incgrenade", "flashbang", "smokegrenade", "decoy", "world", "taser",
    "planted_c4", "",
}

# --- Experiment settings -----------------------------------------------------
MIN_KILLS_PER_PLAYER = 3
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
TARGET_FPR = 0.05              # decision threshold is chosen so that FPR <= 5 %
RANDOM_SEED = 42

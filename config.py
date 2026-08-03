"""Central configuration for the curvature / tree-depth study."""
from pathlib import Path

# --- paths ---
DATA_PATH = Path(r"C:/Users/asudj/Downloads/hour.csv")   # UCI bikeshare hourly
OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- target & features ---
TARGET = "cnt"
LOG1P_TARGET = True

# Dropped, with reason:
#   casual, registered -> components of cnt (target leakage)
#   instant, dteday    -> identifiers
#   atemp              -> ~0.99 correlated with temp
#   season             -> redundant with mnth
DROP = ["casual", "registered", "instant", "dteday", "atemp", "season"]

# Treated as categorical by CatBoost (integer-coded)
CAT_FEATURES = ["mnth", "hr", "weekday", "weathersit"]

# --- reference model that defines the geometry (kernel + curvature) ---
REF_DEPTH = 2
REF_ITERS = 500
REF_LR = 0.05

# --- Nystrom kernel approximation ---
N_LANDMARKS = 500
LANDMARK_SPACE = "embedding"   # k-means on the leaf-indicator space Phi

# --- graph / Ollivier-Ricci ---
KNN = 15
OR_ALPHA = 0.0                 # idleness 0: measure is all on neighbors, allows
                               # negative curvature (boundary/high-complexity regions)

# --- depth sweep for "required depth" (independent axis) ---
SWEEP_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # extended so best depth isn't censored
ELBOW_FRAC = 0.90              # required depth = shallowest reaching this frac of gain
SWEEP_ITERS = 500
SWEEP_LR = 0.05
STAB_TOL_FRAC = 0.05           # tol = frac * std(y_log): prediction "settled"
SWEEP_CV = 5                   # cross-fitted OOF predictions (None = in-sample)
SWEEP_JOBS = None              # concurrent CatBoost fits (None = auto: ~cores)

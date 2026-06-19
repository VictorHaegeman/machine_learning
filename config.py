# ── OpenPlanet Plugin ─────────────────────────────────────────────────────────
OPENPLANET_PORT: int = 9000

# ── Boucle de contrôle ────────────────────────────────────────────────────────
CONTROL_TIMESTEP: float = 0.05   # 20 Hz (cadence tmrl)

# ── Récompense par trajectoire (méthode tmrl) ─────────────────────────────────
# La récompense = nombre de points de la trajectoire enregistrée franchis par step.
# C'est LA méthode qui fait apprendre vite : signal dense + détection d'échec.
#
# reward.pkl par défaut = la ligne centrale de tmrl-train (marche tout de suite
# sur cette map). Pour une AUTRE map : lance `python record_reward.py`, conduis
# un tour, puis mets REWARD_PKL = "reward.pkl".
REWARD_PKL: str = "reward.pkl"   # ta ligne enregistrée (python record_reward.py)

# Paramètres de RewardFunction (valeurs éprouvées de tmrl)
CHECK_FORWARD:     int   = 500     # autorise/récompense les coupes
CHECK_BACKWARD:    int   = 10
FAILURE_COUNTDOWN: int   = 10      # steps sans progrès avant échec
MIN_STEPS:         int   = 70      # pas d'échec avant ce nb de steps (3.5 s)
MAX_STRAY:         float = 100.0   # reward nul si trop loin de la trajectoire

K_TRAJ:       float = 1.0          # poids de la récompense de trajectoire
K_SPEED:      float = 0.1          # petit bonus de vitesse (densifie le signal)
FINISH_BONUS: float = 100.0        # bonus d'arrivée (= END_OF_TRACK de tmrl)

# ── Observation ───────────────────────────────────────────────────────────────
FRAME_STACK: int = 4    # empile 4 observations → le réseau perçoit le mouvement

# ── Episode ───────────────────────────────────────────────────────────────────
EPISODE_TIMEOUT: float = 120.0   # garde-fou (l'échec vient surtout de RewardFunction)
RESET_WAIT_S:    float = 1.5     # attente après respawn (= SLEEP_TIME_AT_RESET tmrl)

# ── SAC ───────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS:    int   = 5_000_000
LEARNING_RATE:      float = 3e-4
BUFFER_SIZE:        int   = 300_000
LEARNING_STARTS:    int   = 2_000     # ~100 s d'exploration avant d'apprendre
BATCH_SIZE:         int   = 256
TAU:                float = 0.005     # = POLYAK 0.995 de tmrl
GAMMA:              float = 0.995     # horizon long (tmrl)
TRAIN_FREQ:         int   = 1
GRADIENT_STEPS:     int   = 2
ENT_COEF:           str   = "auto"
# target_entropy moins négatif que le défaut (-3 pour 3 actions) → garde plus
# d'exploration et évite que ent_coef s'effondre à 0 (politique figée qui retape
# le même mur sans jamais essayer de freiner).
TARGET_ENTROPY:     float = -1.0

USE_SDE:          bool = True
SDE_SAMPLE_FREQ:  int  = 64
NET_ARCH:         list = [256, 256]

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR:      str = "models"
LOGS_DIR:        str = "logs"
MODEL_NAME:      str = "sac_trackmania"
MODEL_SAVE_FREQ: int = 10_000

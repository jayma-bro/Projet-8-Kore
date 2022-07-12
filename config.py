# %%writefile config.py
import numpy as np
from kaggle_environments import make

# Read env specification
ENV_SPECIFICATION = make("kore_fleets").specification
SHIP_COST = ENV_SPECIFICATION.configuration.spawnCost.default
SHIPYARD_COST = ENV_SPECIFICATION.configuration.convertCost.default
GAME_CONFIG = {
    "episodeSteps": ENV_SPECIFICATION.configuration.episodeSteps.default,  # You might want to start with smaller values ENV_SPECIFICATION.configuration.episodeSteps.default
    "size": ENV_SPECIFICATION.configuration.size.default,
    "maxLogLength": None,
}

# Define your opponent. We'll use the starter bot in the notebook environment for this baseline.
OPPONENT = "opponent/main.py"
GAME_AGENTS = [None, OPPONENT]

# Define our parameters
N_FEATURES = 4
ACTION_SIZE = (7,)
DTYPE = np.float64
MAX_OBSERVABLE_KORE = 500
MAX_OBSERVABLE_SHIPS = 200
MAX_ACTION_FLEET_SIZE = 150
MAX_KORE_IN_RESERVE = 40000
WIN_REWARD = 1000
MAX_SHIP_IN_SHIPYARD = 200
OPTIMAL_SHIP_COUNT = 1000
WEIGHT_SHIP = 1500
WEIGHT_KORE = 100
WEIGHT_CARGO = 0.5
WEIGHT_SHIPYARD = 100
WEIGHT_SHIPS_IN_SHIPYARD = 100

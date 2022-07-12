# %%writefile opponent.py
# All this syspath wranglig is needed to make sure that the agent runs on the target environment and can load both the external dependencies
# and the saved model. Dear kaggle, if possible, please make this easier!
import kaggle_environments.envs.kore_fleets.helpers as kr
import os
import sys
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    # We're in the kaggle target system
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, 'lib'))
    agent_path = os.path.join(KAGGLE_AGENT_PATH, 'opponent_agent')
else:
    # We're somewhere else
    sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))
    agent_path = 'opponent_agent'

# Now for the actual agent
from stable_baselines3 import PPO
from environment import KoreGymEnv

model = PPO.load(agent_path)
kore_env = KoreGymEnv()

def agent(obs, config):
    kore_env.raw_obs = obs
    action = {}
    for SY_id in kr.Board(obs, config).current_player.shipyard_ids:
        kore_env.shipyard_id = SY_id
        state = kore_env.obs_as_gym_state
        mod_act, _ = model.predict(state)
        action.update(kore_env.gym_to_kore_action(mod_act))
    return action

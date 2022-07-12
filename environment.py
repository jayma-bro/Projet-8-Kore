# %%writefile environment.py
import gym
import numpy as np
from gym import spaces
from math import floor
from kaggle_environments import make
import kaggle_environments.envs.kore_fleets.helpers as kr
from typing import Union, Tuple, Optional, Dict
from reward_utils import get_board_value
from config import (
    N_FEATURES,
    ACTION_SIZE,
    GAME_AGENTS,
    GAME_CONFIG,
    DTYPE,
    MAX_OBSERVABLE_KORE,
    MAX_OBSERVABLE_SHIPS,
    MAX_ACTION_FLEET_SIZE,
    MAX_KORE_IN_RESERVE,
    WIN_REWARD,
    MAX_SHIP_IN_SHIPYARD,
)


class KoreGymEnv(gym.Env):
    """An openAI-gym env wrapper for kaggle's kore environment. Can be used with stable-baselines3.

    There are three fundamental components to this class which you would want to customize for your own agents:
        The action space is defined by `action_space` and `gym_to_kore_action()`
        The state space (observations) is defined by `state_space` and `obs_as_gym_state()`
        The reward is computed with `compute_reward()`

    Note that the action and state spaces define the inputs and outputs to your model *as numpy arrays*. Use the
    functions mentioned above to translate these arrays into actual kore environment observations and actions.

    The rest is basically boilerplate and makes sure that the kaggle environment plays nicely with stable-baselines3.

    Usage:
        >>> from stable_baselines3 import PPO
        >>>
        >>> kore_env = KoreGymEnv()
        >>> model = PPO('MlpPolicy', kore_env, verbose=1)
        >>> model.learn(total_timesteps=100000)
    """

    def __init__(self, config=None, agents=None, debug=None):
        super(KoreGymEnv, self).__init__()

        if not config:
            config = GAME_CONFIG
        if not agents:
            agents = GAME_AGENTS
        if not debug:
            debug = True
        self.agents = agents
        self.env = make("kore_fleets", configuration=config, debug=debug)
        self.config = self.env.configuration
        self.trainer = None
        self.raw_obs = None
        self.previous_obs = None

        # Define the action and state space
        # Change these to match your needs. Normalization to the [-1, 1] interval is recommended. See:
        # https://araffin.github.io/slides/rlvs-tips-tricks/#/13/0/0
        # See https://www.gymlibrary.ml/content/spaces/ for more info on OpenAI-gym spaces.
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=ACTION_SIZE,
            dtype=DTYPE
        )

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.config.size ** 2 * N_FEATURES + 7,),
            dtype=DTYPE
        )

        self.strict_reward = config.get('strict', False)

        # Debugging info - Enable or disable as needed
        self.reward = 0
        self.n_steps = 0
        self.n_resets = 0
        self.n_dones = 0
        self.last_action = None
        self.last_done = False
        
        self.shipyard_id: Optional[kr.ShipyardId] = None
        self.shipyards: list[kr.ShipyardId] = []
        self.kore_action = {}

    def reset(self) -> np.ndarray:
        """Resets the trainer and returns the initial observation in state space.

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
        """
        # agents = self.agents if np.random.rand() > .5 else self.agents[::-1]  # Randomize starting position
        self.trainer = self.env.train(self.agents)
        self.raw_obs = self.trainer.reset()
        board: kr.Board = self.board
        self.shipyards = board.current_player.shipyard_ids
        self.kore_action = {}
        self.shipyard_id = self.shipyards.pop()
        self.n_resets += 1
        return self.obs_as_gym_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in the trainer and return the results.

        Args:
            action: The action in action space, i.e. the output of the stable-baselines3 agent

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
            reward: The agent's reward
            done: If True, the episode is over
            info: A dictionary with additional debugging information
        """
        done = False
        info = {}
        shipyard_action = self.gym_to_kore_action(action)
        self.kore_action.update(shipyard_action)
        if self.shipyards == []:
            self.previous_obs = self.raw_obs
            self.raw_obs, _, done, info = self.trainer.step(self.kore_action)  # Ignore trainer reward, which is just delta kore
            self.reward = self.compute_reward(done)
            # print(self.kore_action)
            # print(str(self.board.step) + 'tour de jeu')

            # Debugging info
            # print(self.kore_action)
            # with open('logs/tmp.log', 'a') as log:
            # #    print(self.kore_action.action_type, self.kore_action.num_ships, self.kore_action.flight_plan, file=log)
            #    if done:
            #        print(done, file=log)
            #        print(info, file=log)
            #    if info:
            #        print('info', file=log)
            self.n_steps += 1
            self.last_done = done
            self.last_action = self.kore_action
            self.kore_action = {}
            self.n_dones += 1 if done else 0
            self.shipyards = self.board.current_player.shipyard_ids
            # print(self.shipyards)
            # print(self.board.step)
            if len(self.shipyards) > 0:
                self.shipyard_id = self.shipyards.pop()
            else:
                self.shipyard_id = None
        else:
            self.shipyard_id = self.shipyards.pop()
        
        return self.obs_as_gym_state, self.reward, done, info  # type: ignore

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    @property
    def board(self):
        return kr.Board(self.raw_obs, self.config)

    @property
    def previous_board(self):
        return kr.Board(self.previous_obs, self.config)

    def gym_to_kore_action(self, gym_action: np.ndarray) -> Dict[str, str]:
        """Decode an action in action space as a kore action.

        In other words, transform a stable-baselines3 action into an action compatible with the kore environment.

        This method is central - It defines how the agent output is mapped to kore actions.
        You can modify it to suit your needs.

        Let's start with an übereasy mapping. Our gym_action is a 1-dimensional vector of size 2 (as defined in
        self.action_space). We will interpret the values as follows:
        if gym_action[0] > 0 launch a fleet, elif < 0 build ships, else wait.
        abs(gym_action[0]) encodes the number of ships to build/launch.
        gym_action[1] represents the direction in which to launch the fleet.

        Notes: The same action is sent to all shipyards, though we make sure that the actions are valid.

        Args:
            gym_action: The action produces by our stable-baselines3 agent.

        Returns:
            The corresponding kore environment actions or None if the agent wants to wait.

        """
        action_type = np.array([gym_action[0], gym_action[1], gym_action[2], gym_action[3]])
        number_of_ships = round(
            clip_normalize(
                x=gym_action[4],
                low_in=-1,
                high_in=1,
                low_out=1,
                high_out=MAX_ACTION_FLEET_SIZE
            )  # type: ignore
        )
        action_dir = np.array([round(
            clip_normalize(
                x=gym_action[5],
                low_in=-1,
                high_in=1,
                low_out=-9,
                high_out=9
            )), round(  # type: ignore
            clip_normalize(
                x=gym_action[6],
                low_in=-1,
                high_in=1,
                low_out=-9,
                high_out=9
            ))])  # type: ignore
        # Mapping the number of ships is an interesting exercise. Here we chose a linear mapping to the interval
        # [1, MAX_ACTION_FLEET_SIZE], but you could use something else. With a linear mapping, all values are
        # evenly spaced. An exponential mapping, however, would space out lower values, making them easier for the agent
        # to distinguish and choose, at the cost of needing more precision to accurately select higher values.

        # Broadcast the same action to all shipyards
        # print(str(self.board.step) + 'tour de SY', self.shipyard_id)
        board: kr.Board = self.board
        me: kr.Player = board.current_player
        action = None
        if self.shipyard_id == None:
            return {}
        shipyard: kr.Shipyard = board.shipyards[self.shipyard_id]  # type: ignore
        if action_type.argmax() == 0: # craft ships
            # Limit the number of ships to the maximum that can be actually built
            max_spawn = shipyard.max_spawn
            max_purchasable = floor(me.kore / self.config["spawnCost"])
            number_of_ships = min(max_spawn, max_purchasable)
            if number_of_ships:
                action = kr.ShipyardAction.spawn_ships(number_ships=number_of_ships)
        elif action_type.argmax() == 1: # lauch linear
            # Limit the number of ships to the amount that is actually present in the shipyard
            shipyard_count = shipyard.ship_count
            number_of_ships = max(number_of_ships, 3)
            number_of_ships = min(number_of_ships, shipyard_count)
            if number_of_ships >= 2:
                direction, _ = self.direction(action_dir=action_dir)
                flight_plan = kr.Direction.from_index(direction).to_char()  # int between 0 (North) and 3 (West)
                action = kr.ShipyardAction.launch_fleet_with_flight_plan(number_of_ships, flight_plan)
        elif action_type.argmax() == 2: # lauch circular
            # Limit the number of ships to the amount that is actually present in the shipyard
            shipyard_count = shipyard.ship_count
            number_of_ships = max(number_of_ships, 21)
            number_of_ships = min(number_of_ships, shipyard_count)
            if number_of_ships >= 21:
                flight_plan = ""
                direction, length = self.direction(action_dir=action_dir, first=True)
                flight_plan += kr.Direction.from_index(direction).to_char()
                if length > 0:
                    flight_plan += str(length)
                direction, length = self.direction(action_dir=action_dir, first=False)
                if length > 0:
                    flight_plan += kr.Direction.from_index(direction).to_char()
                    flight_plan += str(length)
                direction, length = self.direction(action_dir=action_dir, first=True)
                if length > 0:
                    flight_plan += kr.Direction.from_index((direction + 2) % 4).to_char()
                    flight_plan += str(length)
                direction, length = self.direction(action_dir=action_dir, first=False)
                if length > 0:
                    flight_plan += kr.Direction.from_index((direction + 2) % 4).to_char()
                action = kr.ShipyardAction.launch_fleet_with_flight_plan(number_of_ships, flight_plan)
        else: # lauch built
            # Limit the number of ships to the amount that is actually present in the shipyard
            shipyard_count = shipyard.ship_count
            number_of_ships = max(number_of_ships, 50)
            number_of_ships = min(number_of_ships, shipyard_count)
            if number_of_ships >= 50:
                flight_plan = ""
                direction, length = self.direction(action_dir=action_dir, first=True)
                flight_plan += kr.Direction.from_index(direction).to_char()
                if length > 0:
                    flight_plan += str(length)
                direction, length = self.direction(action_dir=action_dir, first=False)
                if length > 0:
                    flight_plan += kr.Direction.from_index(direction).to_char()
                    flight_plan += str(length)
                flight_plan += "C"
                action = kr.ShipyardAction.launch_fleet_with_flight_plan(number_of_ships, flight_plan)
        shipyard.next_action = action
        return me.next_actions
    
    def direction(self, action_dir: np.ndarray, first: bool = True) -> tuple:
        if first:
            axe = abs(action_dir).argmax()
        else:
            axe = (abs(action_dir).argmax() + 1) % 2
        sens = action_dir[axe] > 0
        if not axe and sens:
            direction = 0
        elif axe and sens:
            direction = 1
        elif not axe and not sens:
            direction = 2
        else:
            direction = 3
        return direction, abs(action_dir[axe])
        
    @property
    def obs_as_gym_state(self) -> np.ndarray:
        """Return the current observation encoded as a state in state space.

        In other words, transform a kore observation into a stable-baselines3-compatible np.ndarray.

        This property is central - It defines how the kore board is mapped to our state space.
        You can modify it to include as many features as you see convenient.

        Let's keep start with something easy: Define a 21x21x4+3 state (size x size x n_features and 3 extra features).
        # Feature 0: How much kore there is in a cell
        # Feature 1: How many ships there are in a cell (>0: friendly, <0: enemy)
        # Feature 2: Fleet direction
        # Feature 3: Is a shipyard present? (1: friendly, -1: enemy, 0: no)
        # Feature 4: Progress - What turn is it?
        # Feature 5: How much kore do I have?
        # Feature 6: How much kore does the opponent have?
        # Feature 7: How much ships does the shipyard have?
        # Feature 8: How much ships do you have?
        # Feature 9: X coordonate of shipyard
        # Feature 10: Y coordonate of shipyard

        We'll make sure that all features are in the range [-1, 1] and as close to a normal distribution as possible.

        Note: This mapping doesn't tackle a critical issue in kore: How to encode (full) flight plans?
        """
        # Init output state
        gym_state = np.ndarray(shape=(self.config.size, self.config.size, N_FEATURES))
        # Get our player ID
        board: kr.Board = self.board
        our_id: kr.PlayerId = board.current_player_id

        for point, cell in board.cells.items():
            # Feature 0: How much kore
            gym_state[point.y, point.x, 0] = cell.kore

            # Feature 1: How many ships (>0: friendly, <0: enemy)
            # Feature 2: Fleet direction
            fleet = cell.fleet
            if fleet:
                modifier = 1 if fleet.player_id == our_id else -1
                gym_state[point.y, point.x, 1] = modifier * fleet.ship_count
                gym_state[point.y, point.x, 2] = fleet.direction.value
            else:
                # The current cell has no fleet
                gym_state[point.y, point.x, 1] = gym_state[point.y, point.x, 2] = 0

            # Feature 3: Shipyard present (1: friendly, -1: enemy)
            shipyard = cell.shipyard  # type: ignore
            if shipyard:
                gym_state[point.y, point.x, 3] = 1 if shipyard.player_id == our_id else -1
            else:
                # The current cell has no shipyard
                gym_state[point.y, point.x, 3] = 0

        # Normalize features to interval [-1, 1]
        # Feature 0: Logarithmic scale, kore in range [0, MAX_OBSERVABLE_KORE]
        gym_state[:, :, 0] = clip_normalize(
            x=np.log2(gym_state[:, :, 0] + 1),
            low_in=0,
            high_in=np.log2(MAX_OBSERVABLE_KORE)
        )

        # Feature 1: Ships in range [-MAX_OBSERVABLE_SHIPS, MAX_OBSERVABLE_SHIPS]
        gym_state[:, :, 1] = clip_normalize(
            x=gym_state[:, :, 1],
            low_in=-MAX_OBSERVABLE_SHIPS,
            high_in=MAX_OBSERVABLE_SHIPS
        )

        # Feature 2: Fleet direction in range (1, 4)
        gym_state[:, :, 2] = clip_normalize(
            x=gym_state[:, :, 2],
            low_in=1,
            high_in=4
        )

        # Feature 3 is already as normal as it gets

        # Flatten the input (recommended by stable_baselines3.common.env_checker.check_env)
        player = board.current_player
        player_fleets, player_shipyards = list(player.fleets), list(player.shipyards)
        
        output_state = gym_state.flatten()
        num_ships = sum(fleet.ship_count for fleet in player_fleets) + sum(shipyard.ship_count for shipyard in player_shipyards)

        # Extra Features: Progress, how much kore do I have, how much kore does opponent have

        if self.shipyard_id==None:
            ship_count = 0
            shipyard_x = 0
            shipyard_y = 0
        else:
            shipyard: kr.Shipyard = board.shipyards[self.shipyard_id]
            ship_count = shipyard.ship_count
            shipyard_x = shipyard.position[0]
            shipyard_y = shipyard.position[1]
        player: kr.Player = board.current_player
        opponent = board.opponents[0]
        progress = clip_normalize(board.step, low_in=0, high_in=GAME_CONFIG['episodeSteps'])
        my_kore = clip_normalize(np.log2(player.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))
        opponent_kore = clip_normalize(np.log2(opponent.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))
        shipyard_ships = clip_normalize(ship_count, low_in=0, high_in=MAX_SHIP_IN_SHIPYARD)
        num_ships = clip_normalize(num_ships, low_in=0, high_in=2000)
        shipyard_x = clip_normalize(shipyard_x, low_in=0, high_in=21)
        shipyard_y = clip_normalize(shipyard_y, low_in=0, high_in=21)

        return np.append(output_state, [progress, my_kore, opponent_kore, shipyard_ships, num_ships, shipyard_x, shipyard_y])  # type: ignore

    def compute_reward(self, done: bool, strict=False) -> float:
        """Compute the agent reward. Welcome to the fine art of RL.

         We'll compute the reward as the current board value and a final bonus if the episode is over. If the player
          wins the episode, we'll add a final bonus that increases with shorter time-to-victory.
        If the player loses, we'll subtract that bonus.

        Args:
            done: True if the episode is over
            strict: If True, count only wins/loses (Useful for evaluating a trained agent)

        Returns:
            The agent's reward
        """
        board: kr.Board = self.board
        previous_board: kr.Board = self.previous_board

        if strict:
            if done:
                # Who won?
                # Ugly but 99% sure correct, see https://www.kaggle.com/competitions/kore-2022/discussion/324150#1789804
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                return int(agent_reward > opponent_reward)
            else:
                return 0
        else:
            if done:
                # Who won?
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                if agent_reward is None or opponent_reward is None:
                    we_won = -1
                else:
                    we_won = 1 if agent_reward > opponent_reward else -1
                win_reward = we_won * (WIN_REWARD + 5 * (GAME_CONFIG['episodeSteps'] - board.step))
            else:
                win_reward = 0

            return get_board_value(board) - get_board_value(previous_board) + win_reward


def clip_normalize(x: Union[np.ndarray, float],
                   low_in: float,
                   high_in: float,
                   low_out=-1.,
                   high_out=1.) -> Union[np.ndarray, float]:
    """Clip values in x to the interval [low_in, high_in] and then MinMax-normalize to [low_out, high_out].

    Args:
        x: The array of float to clip and normalize
        low_in: The lowest possible value in x
        high_in: The highest possible value in x
        low_out: The lowest possible value in the output
        high_out: The highest possible value in the output

    Returns:
        The clipped and normalized version of x

    Raises:
        AssertionError if the limits are not consistent

    Examples:
        >>> clip_normalize(50, low_in=0, high_in=100)
        0.0

        >>> clip_normalize(np.array([-1, .5, 99]), low_in=-1, high_in=1, low_out=0, high_out=2)
        array([0., 1.5, 2.])
    """
    assert high_in > low_in and high_out > low_out, "Wrong limits"

    # Clip outliers
    try:
        x[x > high_in] = high_in
        x[x < low_in] = low_in
    except TypeError:
        x = high_in if x > high_in else x
        x = low_in if x < low_in else x

    # y = ax + b
    a = (high_out - low_out) / (high_in - low_in)
    b = high_out - high_in * a

    return a * x + b

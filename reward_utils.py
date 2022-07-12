# %%writefile reward_utils.py
from config import GAME_CONFIG, SHIP_COST, SHIPYARD_COST, OPTIMAL_SHIP_COUNT, WEIGHT_SHIP, WEIGHT_KORE, WEIGHT_CARGO, WEIGHT_SHIPYARD, WEIGHT_SHIPS_IN_SHIPYARD
from kaggle_environments.envs.kore_fleets.helpers import Board
from typing import Union, Tuple, Optional, Dict
import numpy as np
import math

# Compute weight constants -- See get_board_value's docstring
# _max_steps = GAME_CONFIG['episodeSteps']
# _end_of_asset_value = floor(.5 * _max_steps)
# _weights_assets = np.linspace(start=1, stop=0, num=_end_of_asset_value)
# _weights_kore = np.linspace(start=0, stop=1, num=_end_of_asset_value)
# WEIGHTS_ASSETS = np.append(_weights_assets, np.zeros(_max_steps - _end_of_asset_value))
# WEIGHTS_KORE = np.append(_weights_kore, np.ones(_max_steps - _end_of_asset_value))
WEIGHTS_MAX_SPAWN = {x: (x+3)/4 for x in range(1, 11)}  # Value multiplier of a shipyard as a function of its max spawn
# WEIGHTS_KORE_IN_FLEETS = WEIGHTS_KORE * WEIGHTS_ASSETS/2  # Always equal or smaller than either, almost always smaller
WEIGHTS_KORE_steps = np.linspace(start=0, stop=1, num=GAME_CONFIG['episodeSteps'])



def get_board_value(board: Board) -> float:
    """Computes the board value for the current player.

    The board value captures how are we currently performing, compared to the opponent. Each player's partial board
    value assesses the player's situation, taking into account their current kore, ship count, shipyard count
    (including their max spawn) and kore carried by fleets. We then define the board value as the difference between
    player's partial board values.
    Flight plans and the positioning of fleet and shipyards do not flow into the board value (yet).

    To keep things simple, we'll take a weighted sum as the partial board value. We need weighting since
    the importance of each item changes over time. We don't need to have the most kore at the beginning of the game,
    but we do at the end. Ship count won't help us win games in the latter stages, but it is crucial in the beginning.
    Fleets and shipyards will be accounted for proportionally to their kore cost.

    For efficiency, the weight factors are pre-computed at module level. Here is the logic behind the weighting:
    WEIGHTS_KORE: Applied to the player's kore count. Increases linearly from 0 to 1. It reaches one before
        the maximum game length is reached.
    WEIGHTS_ASSETS: Applied to fleets and shipyards. Decreases linearly from 1 to 0 and reaches zero before the maximum
        length. It emphasizes the need of having ships over kore at the beginning of the game.
    WEIGHTS_MAX_SPAWN: Shipyard value is multiplied by its max spawn. This captures the idea that long-held shipyards
        are more valuable.
    WEIGHTS_KORE_IN_FLEETS: Kore in fleets should be valued, too. But its value must be upper-bounded by WEIGHTS_KORE
        (it can never be better to have kore in cargo than home) and it must decrease in time, since it doesn't
        count towards the end kore count.

    Args:
        board: The board for which we want to compute the value.

    Returns:
        The value of the board.
    """
    board_value: float = 0.0
    if not board:
        return board_value

    # Get the weights as a function of the current game step
    step = board.step

    # Compute the partial board values
    player = board.current_player
    player_fleets, player_shipyards = list(player.fleets), list(player.shipyards)

    value_kore =  math.log(player.kore + 1) * WEIGHT_KORE * WEIGHTS_KORE_steps[step]

    # value_fleets = weight_assets * SHIP_COST * (
    #         sum(fleet.ship_count for fleet in player_fleets)
    #         + sum(shipyard.ship_count for shipyard in player_shipyards)
    # )
    num_ships = sum(fleet.ship_count for fleet in player_fleets) + sum(shipyard.ship_count for shipyard in player_shipyards)

    if num_ships<=OPTIMAL_SHIP_COUNT:
        value_fleets = clip_normalize(num_ships, low_in=0, high_in=OPTIMAL_SHIP_COUNT, low_out=0, high_out=1) * WEIGHT_SHIP
    else:
        value_fleets = (1-clip_normalize(num_ships, low_in=OPTIMAL_SHIP_COUNT, high_in=OPTIMAL_SHIP_COUNT*2, low_out=0, high_out=1)) * WEIGHT_SHIP
    
    value_shipyards = WEIGHT_SHIPYARD * (sum(WEIGHTS_MAX_SPAWN[shipyard.max_spawn] for shipyard in player_shipyards))
    # value_shipyards = weight_assets * SHIPYARD_COST * (
    #     sum(shipyard.max_spawn * WEIGHTS_MAX_SPAWN[shipyard.max_spawn] for shipyard in player_shipyards)
    # )
    value_kore_in_cargo = sum(fleet.kore for fleet in player_fleets) * WEIGHT_CARGO
    # value_kore_in_cargo = weight_cargo * sum(fleet.kore for fleet in player_fleets)
    ships_in_shipyard = 0 - sum(clip_normalize(shipyard.ship_count, low_in=0, high_in=300, low_out=0, high_out=WEIGHT_SHIPS_IN_SHIPYARD) for shipyard in player_shipyards)

    # Add (or subtract) the partial values to the total board value. The current player is always us.
    board_value += value_kore + value_fleets + value_shipyards + value_kore_in_cargo + ships_in_shipyard

    

    #    Debugging info
    # with open('logs/tmp.log', 'a') as log:
    #     print('step '+ str(step), file=log)
    #     print("value_kore {:.2f}".format(value_kore), file=log)
    #     print("num_ships {}".format(num_ships), file=log)
    #     print("value_fleets {:.2f}".format(value_fleets), file=log)
    #     print("value_shipyards {:.2f}".format(value_shipyards), file=log)
    #     print("value_kore_in_cargo {:.2f}".format(value_kore_in_cargo), file=log)


    return board_value # c'est un flotant


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

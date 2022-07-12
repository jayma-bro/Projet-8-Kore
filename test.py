from kaggle_environments.envs.kore_fleets.helpers import *
from random import randint
import pickle

# a flight plan
def build_flight_plan(dir_idx, size):
    flight_plan = ""
    for i in range(4):
        flight_plan += Direction.from_index((dir_idx + i) % 4).to_char()
        if not i == 3:
            flight_plan += str(size)
    return flight_plan

def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player
    turn = board.step
    spawn_cost = board.configuration.spawn_cost
    convert_cost = board.configuration.convert_cost
    kore_left = me.kore
    for shipyard in me.shipyards:
        action = None
        if kore_left >= 500 and shipyard.ship_count >= convert_cost:
            flight_plan = build_flight_plan(randint(0, 3), randint(10, 15))
            flight_plan = flight_plan[:6] + "C"
            action = ShipyardAction.launch_fleet_with_flight_plan(convert_cost, flight_plan)
        elif shipyard.ship_count >= 21:
            flight_plan = build_flight_plan(randint(0, 3), randint(2, 9))
            action = ShipyardAction.launch_fleet_with_flight_plan(21, flight_plan)
        elif kore_left >= spawn_cost:
            action = ShipyardAction.spawn_ships(1)
        shipyard.next_action = action
    return me.next_actions
# %%writefile "opponent/main.py"
# <--->
from board import Board
from offence import capture_shipyards
from defence import defend_shipyards
from expantion import expand
from mining import mine
from control import spawn, greedy_spawn, adjacent_attack, direct_attack

# <--->


def agent(obs, conf):

    board = Board(obs, conf)
    step = board.step
    my_id = obs["player"]
    remaining_time = obs["remainingOverageTime"]

    try:
        a = board.get_player(my_id)
    except KeyError:
        return {}

    if not a.opponents:
        return {}

    defend_shipyards(a)
    capture_shipyards(a)
    adjacent_attack(a)
    direct_attack(a)
    expand(a)
    greedy_spawn(a)
    mine(a)
    spawn(a)

    return a.actions()

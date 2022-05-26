import os
import traceback

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from board import Board
    from logger import logger, init_logger
    from offence import capture_shipyards, coordinate_shipyard_capture
    from defence import defend_shipyards
    from expansion import expand
    from mining import mine
    from control import spawn, greedy_spawn, adjacent_attack, direct_attack, save_kore
    from state import State
else:
    from .board import Board
    from .logger import logger, init_logger
    from .offence import capture_shipyards, coordinate_shipyard_capture
    from .defence import defend_shipyards
    from .expansion import expand
    from .mining import mine
    from .control import spawn, greedy_spawn, adjacent_attack, direct_attack, save_kore
    from .state import State
# <--->

prev_state = State()

def agent(obs, conf):
    global prev_state
    if obs["step"] == 0:
        init_logger(logger)

    board = Board(obs, conf)
    step = board.step
    my_id = obs["player"]
    remaining_time = obs["remainingOverageTime"]
    logger.info(f"<step_{step + 1}>, remaining_time={remaining_time:.1f}")

    try:
        a = board.get_player(my_id)
    except KeyError:
        return {}

    if not a.opponents:
        return {}

    try:
        if prev_state.__repr__() != "State":
            logger.info(f"State: {prev_state}")
        a.state = prev_state

        defend_shipyards(a)
        save_kore(a)
        capture_shipyards(a)
        coordinate_shipyard_capture(a)
        adjacent_attack(a)
        direct_attack(a)
        expand(a)
        greedy_spawn(a)
        mine(a)
        spawn(a)

        prev_state = a.state
    except:
        logger.error(traceback.format_exc())
        exit()

    return a.actions()

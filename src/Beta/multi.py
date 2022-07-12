import os
import traceback
from typing import Set

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from board import Board
    from geometry import Point
    from logger import logger, init_logger
    from offence import capture_shipyards, coordinate_shipyard_capture, whittle_attack
    from defence import defend_shipyards
    from expansion import expand
    from mining import mine
    from control import spawn, greedy_spawn, adjacent_attack, direct_attack, save_kore, conservative_save_kore
    from state import State, Memory
else:
    from .board import Board
    from .geometry import Point
    from .logger import logger, init_logger
    from .offence import capture_shipyards, coordinate_shipyard_capture, whittle_attack
    from .defence import defend_shipyards
    from .expansion import expand
    from .mining import mine
    from .control import spawn, greedy_spawn, adjacent_attack, direct_attack, save_kore, conservative_save_kore
    from .state import State, Memory
# <--->

def make_agent():
    prev_state: State = State()
    self_built_sys: Set[Point] = set()
    lost_sys: Set[Point] = set()
    memory: Memory = Memory()
    initialized = False

    def agent(obs, conf):
        nonlocal prev_state
        nonlocal self_built_sys
        nonlocal lost_sys
        nonlocal memory
        nonlocal initialized
        if not initialized:
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
            if not initialized:
                for sy in a.shipyards:
                    self_built_sys.add(sy.point)
            else:
                for sy in a.shipyards:
                    if sy.point in lost_sys:
                        lost_sys.remove(sy.point)
                for sy in self_built_sys:
                    if sy not in lost_sys and not any(sy == x.point for x in a.all_shipyards):
                        lost_sys.add(sy)
            if prev_state.__repr__() != "State":
                logger.info(f"State: {prev_state}")
            a.state = prev_state

            memory.update_memory(a)
            a.memory = memory

            conservative_save_kore(a)
            defend_shipyards(a, self_built_sys)
            save_kore(a)
            coordinate_shipyard_capture(a)
            capture_shipyards(a)
            expand(a, step, self_built_sys, lost_sys)
            whittle_attack(a, step)
            adjacent_attack(a)
            direct_attack(a)
            greedy_spawn(a)
            mine(a, remaining_time)
            spawn(a)

            prev_state = a.state
            memory = a.memory
        except:
            logger.error(traceback.format_exc())
            exit()

        if not initialized:
            initialized = True

        return a.actions()
    return agent


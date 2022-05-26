from enum import Enum
import os
from typing import Dict, Tuple

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from board import Shipyard, Player, Launch
    from geometry import Point
    from helpers import find_shortcut_routes
    from logger import logger
else:
    from .board import Shipyard, Player, Launch
    from .geometry import Point
    from .helpers import find_shortcut_routes
    from .logger import logger


class State:
    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def act(self, agent: Player):
        pass

    def is_finished(self):
        return False

    def next_state(self):
        raise NotImplementedError


class CoordinatedAttack(State):
    def __init__(self, shipyard_to_launch: Dict[Shipyard, Tuple[int, int]], target: Point):
        super().__init__()
        self.shipyard_to_launch = shipyard_to_launch
        self.target = target

    def __repr__(self):
        return f"{self.__class__.__name__}({self.shipyard_to_launch}, {self.target})"

    def act(self, agent: Player):
        board = agent.board
        new_shipyard_to_launch = {}
        for sy, (power, wait_time) in self.shipyard_to_launch.items():
            for shipyard in agent.shipyards:
                if shipyard.game_id == sy.game_id:
                    sy = shipyard
                    break

            num_ships_to_launch = min(sy.available_ship_count, int(power * 1.2))
            if wait_time <= 0:
                if sy.available_ship_count < power:
                    logger.info(f"Error: CoordinatedAttack: {sy} has {sy.available_ship_count} ships, but {power} power")
                routes = find_shortcut_routes(
                    board,
                    sy.point,
                    self.target,
                    agent,
                    num_ships_to_launch,
                    allow_join=True
                )

                if routes:
                    best_route = min(routes, key=lambda route: route.expected_kore(board, num_ships_to_launch))
                    logger.info(
                        f"Coordinated attack shipyard {num_ships_to_launch} {sy.point}->{self.target}"
                    )
                    sy.action = Launch(num_ships_to_launch, best_route)
                else:
                    sy.increase_reserved_ship_count(num_ships_to_launch)
                    new_shipyard_to_launch[sy] = (power, wait_time - 1)
            else:
                # Reserve ships for the attack
                sy.increase_reserved_ship_count(num_ships_to_launch)
                new_shipyard_to_launch[sy] = (power, wait_time - 1)

        self.shipyard_to_launch = new_shipyard_to_launch

    def is_finished(self):
        return not self.shipyard_to_launch

    def next_state(self):
        return State()

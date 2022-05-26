from enum import Enum
import os
from typing import Dict, Tuple

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from basic import min_ship_count_for_flight_plan_len
    from board import Shipyard, Player, Launch, BoardRoute, DontLaunch, Spawn
    from geometry import Point, Convert, PlanRoute, PlanPath
    from helpers import find_shortcut_routes, is_safety_route_to_convert
    from logger import logger
else:
    from .basic import min_ship_count_for_flight_plan_len
    from .board import Shipyard, Player, Launch, BoardRoute, DontLaunch, Spawn
    from .geometry import Point, Convert, PlanRoute, PlanPath
    from .helpers import find_shortcut_routes, is_safety_route_to_convert
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
            found_sy = False
            for shipyard in agent.shipyards:
                if shipyard.game_id == sy.game_id:
                    sy = shipyard
                    found_sy = True
                    break

            if not found_sy:
                logger.error(f"Error: CoordinatedAttack: Could not find shipyard {sy.point}. It may have been taken")
                continue

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
                    self._spawn(agent, sy)
                    new_shipyard_to_launch[sy] = (power, wait_time - 1)
            else:
                self._spawn(agent, sy)
                new_shipyard_to_launch[sy] = (power, wait_time - 1)

        self.shipyard_to_launch = new_shipyard_to_launch
        # logger.info(f"Coordinated attack: {self.shipyard_to_launch}")

    def is_finished(self):
        return not self.shipyard_to_launch

    def next_state(self):
        return State()

    def _spawn(self, agent: Player, shipyard: Shipyard):
        board = agent.board
        num_ships_to_spawn = min(
            int(agent.available_kore() // board.spawn_cost),
            shipyard.max_ships_to_spawn,
        )
        if num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)


class Expansion(State):
    def __init__(self, shipyard_to_target: Dict[Shipyard, Point]):
        super().__init__()
        self.shipyard_to_target = shipyard_to_target

    def __repr__(self):
        return f"{self.__class__.__name__}({self.shipyard_to_target})"

    def act(self, agent: Player):
        board = agent.board
        shipyard_positions = {x.point for x in board.shipyards}
        new_shipyard_to_target = {}
        for sy, target in self.shipyard_to_target.items():
            found_sy = False
            for shipyard in agent.shipyards:
                if shipyard.game_id == sy.game_id:
                    sy = shipyard
                    found_sy = True
                    break

            if not found_sy:
                logger.error(f"Error: Expansion: Could not find shipyard {sy.point}. It may have been taken")
                continue

            if sy.action:
                logger.error(f"Error: Expansion: {sy.point} already has action {sy.action}")
                new_shipyard_to_target[sy] = target
                continue

            if sy.available_ship_count < 63:
                logger.info(f"Expansion: {sy.point} has {sy.available_ship_count} and is waiting to launch")
                new_shipyard_to_target[sy] = target
                sy.action = DontLaunch()
                continue

            target_distance = shipyard.distance_from(target)

            routes = []
            for p in board:
                if p in shipyard_positions:
                    continue

                distance = shipyard.distance_from(p) + p.distance_from(target)
                if distance > target_distance:
                    continue

                plans = shipyard.get_plans_through([p, target])
                rs = [BoardRoute(shipyard.point, plan) for plan in plans]

                for route in rs:
                    if shipyard.available_ship_count < min_ship_count_for_flight_plan_len(
                        len(route.plan.to_str()) + 1
                    ):
                        continue

                    route_points = route.points()
                    if any(x in shipyard_positions for x in route_points):
                        continue

                    if not is_safety_route_to_convert(route_points, agent):
                        continue

                    routes.append(route)

            if routes:
                route = max(routes, key=lambda route: route.expected_kore(board, sy.available_ship_count))
                route = BoardRoute(
                    sy.point, route.plan + PlanRoute([PlanPath(Convert)])
                )
                logger.info(f"Building new sy {sy.point}->{route.end}")
                sy.action = Launch(sy.available_ship_count, route)
            else:
                logger.info(f"No routes for {sy.point}->{target}")
                new_shipyard_to_target[sy] = target

        self.shipyard_to_target = new_shipyard_to_target

    def is_finished(self):
        return not self.shipyard_to_target

    def next_state(self):
        return State()


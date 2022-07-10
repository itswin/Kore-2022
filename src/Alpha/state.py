from collections import defaultdict
import os
from typing import Dict, Tuple, Set, List

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from basic import min_ship_count_for_flight_plan_len
    from board import Shipyard, Player, Launch, BoardRoute, Spawn, AllowMine
    from geometry import Point, Convert, PlanRoute, PlanPath
    from helpers import find_shortcut_routes, is_safety_route_to_convert, _spawn
    from logger import logger
else:
    from .basic import min_ship_count_for_flight_plan_len
    from .board import Shipyard, Player, Launch, BoardRoute, Spawn, AllowMine
    from .geometry import Point, Convert, PlanRoute, PlanPath
    from .helpers import find_shortcut_routes, is_safety_route_to_convert, _spawn
    from .logger import logger

class Memory:
    def __init__(self):
        self.agent = None
        self.sy_to_turn_attacked = defaultdict(int)

    def __repr__(self):
        return f"Memory(sy_to_turn_attacked={self.sy_to_turn_attacked})"

    def update_memory(self, agent: Player):
        self.agent = agent
        new_sy_to_turn_attacked = defaultdict(int)
        for sy in agent.shipyards:
            if sy.incoming_hostile_fleets:
                new_sy_to_turn_attacked[sy.point] = agent.board.step
            else:

                new_sy_to_turn_attacked[sy.point] = self.sy_to_turn_attacked[sy.point]
        self.sy_to_turn_attacked = new_sy_to_turn_attacked

    def recently_attacked_sys(self, turn: int, within_turns: int = 5) -> List[Point]:
        return [
            sy
            for sy in self.agent.shipyards
            if self.sy_to_turn_attacked[sy.point] + within_turns >= turn
        ]

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

    def is_sy_used(self, _: Shipyard):
        return False


class CoordinatedAttack(State):
    def __init__(self, shipyard_to_launch: Dict[Shipyard, Tuple[int, int]], target: Point, max_timeout: int = 5):
        super().__init__()
        self.shipyard_to_launch = shipyard_to_launch
        self.target = target
        self._max_timeout = max_timeout

    def __repr__(self):
        return f"{self.__class__.__name__}({list((sy.point, (power, wait_time)) for (sy, (power, wait_time)) in self.shipyard_to_launch.items())}, {self.target})"

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
                logger.error(f"CoordinatedAttack: Could not find shipyard {sy.point}. It may have been taken")
                continue

            if wait_time <= -self._max_timeout:
                logger.error(f"CoordinatedAttack: Waited too long for {sy.point} to find routes. Skipping")
                continue

            num_ships_to_launch = min(sy.available_ship_count, int(power * 1.2))
            if wait_time <= 0:
                if sy.available_ship_count < power:
                    logger.info(f"CoordinatedAttack: {sy} has {sy.available_ship_count} ships, but {power} power")
                routes = find_shortcut_routes(
                    board,
                    sy.point,
                    self.target,
                    agent,
                    num_ships_to_launch,
                    allow_join=True
                )

                if routes:
                    best_route = min(routes, key=lambda route: (route.expected_kore(board, num_ships_to_launch), len(route)))
                    logger.info(
                        f"Coordinated attack shipyard {num_ships_to_launch} {sy.point}->{self.target}"
                    )
                    sy.action = Launch(num_ships_to_launch, best_route)
                else:
                    _spawn(agent, sy)
                    logger.info(f"CoordinatedAttack: No routes found for {sy.point}->{self.target}")
                    new_shipyard_to_launch[sy] = (power, wait_time - 1)
            else:
                _spawn(agent, sy)
                logger.info(f"CoordinatedAttack: Not time for {sy.point} to send ships {self.target}")
                new_shipyard_to_launch[sy] = (power, wait_time - 1)

        self.shipyard_to_launch = new_shipyard_to_launch

    def is_finished(self):
        return not self.shipyard_to_launch

    def next_state(self):
        return State()

    def is_sy_used(self, sy: Shipyard):
        for shipyard, _ in self.shipyard_to_launch.items():
            if shipyard.game_id == sy.game_id:
                return True
        return False



class PrepCoordinatedAttack(State):
    def __init__(self, max_time_to_wait: int, target: Point = None, fraction: float = 0.5):
        super().__init__()
        self.max_time_to_wait = max_time_to_wait
        self.target = target
        self.fraction = fraction
        self._enough_ships_ready = False
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.max_time_to_wait})({self.target})"

    def act(self, agent: Player):
        self.max_time_to_wait -= 1

        ship_count = agent.ship_count
        ships_ready = 0
        for sy in agent.shipyards:
            ships_ready += sy.available_ship_count
        self._enough_ships_ready = ships_ready >= ship_count * self.fraction

        if self.is_finished():
            return

        for sy in agent.shipyards:
            _spawn(agent, sy)

    def is_finished(self):
        return self.max_time_to_wait <= 0 or self._enough_ships_ready

    def next_state(self):
        return State()

    def is_sy_used(self, _: Shipyard):
        return True


class Expansion(State):
    def __init__(self, shipyard_to_target: Dict[Shipyard, Point], self_built_sys: Set[Shipyard], extra_dist: int = 0):
        super().__init__()
        self.shipyard_to_target = shipyard_to_target
        self.self_built_sys = self_built_sys
        self.extra_distance = extra_dist

    def __repr__(self):
        return f"{self.__class__.__name__}({self.shipyard_to_target})"

    def act(self, agent: Player):
        board = agent.board
        shipyard_positions = {x.point for x in board.all_shipyards}
        new_shipyard_to_target = {}

        min_eta = min((min((x.eta for x in sy.incoming_allied_fleets), default=0) for sy in agent.shipyards), default=0)
        max_opp_sy_power = max((x.ship_count for x in agent.opponents[0].shipyards), default=0)

        for sy, target in self.shipyard_to_target.items():
            found_sy = False
            for shipyard in agent.shipyards:
                if shipyard.game_id == sy.game_id:
                    sy = shipyard
                    found_sy = True
                    break

            if not found_sy:
                logger.error(f"Expansion: Could not find shipyard {sy.point}. It may have been taken")
                continue

            if sy.action:
                logger.error(f"Expansion: {sy.point} already has action {sy.action}")
                new_shipyard_to_target[sy] = target
                continue

            thresh = 50 if max_opp_sy_power >= 50 else 63
            if sy.available_ship_count < thresh:
                logger.info(f"Expansion: {sy.point} has {sy.available_ship_count} and is waiting to launch")
                new_shipyard_to_target[sy] = target
                _spawn(agent, sy)
                if not isinstance(sy.action, Spawn):
                    # Workaround to allow mining if no fleets out now.
                    if min_eta == 0:
                        min_eta = 30
                    sy.action = AllowMine(min_eta // 2, sy.point)
                continue 

            target_distance = shipyard.distance_from(target) + 2 * (self.extra_distance)

            routes = []
            for p in board:
                if p in shipyard_positions:
                    continue

                distance = shipyard.distance_from(p) + p.distance_from(target)
                if distance > target_distance:
                    continue

                plans = shipyard.get_plans_through([p, target])
                rs = [BoardRoute(shipyard.point, plan + PlanRoute([PlanPath(Convert)])) for plan in plans]

                for route in rs:
                    if shipyard.available_ship_count < min_ship_count_for_flight_plan_len(
                        len(route.plan.to_str())
                    ):
                        continue

                    route_points = route.points()
                    if any(x in shipyard_positions for x in route_points):
                        continue

                    if not is_safety_route_to_convert(route_points, agent, sy.available_ship_count):
                        continue

                    routes.append(route)

            if routes:
                route = max(routes, key=lambda route: (-len(route), route.expected_kore(board, sy.available_ship_count)))
                logger.info(f"Building new sy {sy.point}->{route.end}")
                sy.action = Launch(sy.available_ship_count, route)
                self.self_built_sys.add(target)
            else:
                logger.info(f"No routes for {sy.point}->{target} with distance {target_distance}")
                _spawn(agent, sy)
                new_shipyard_to_target[sy] = target
                self.extra_distance += 1

        self.shipyard_to_target = new_shipyard_to_target

    def is_finished(self):
        return not self.shipyard_to_target

    def next_state(self):
        return State()

    def is_sy_used(self, sy: Shipyard):
        for shipyard, _ in self.shipyard_to_target.items():
            if shipyard.game_id == sy.game_id:
                return True
        return False

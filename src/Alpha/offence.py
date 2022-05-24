from math import floor
import numpy as np
import os

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from basic import max_ships_to_spawn
    from board import Player, Shipyard, Launch, DontLaunch
    from helpers import find_shortcut_routes
    from logger import logger
    from state import State, CoordinatedAttack
else:
    from .basic import max_ships_to_spawn
    from .board import Player, Shipyard, Launch, DontLaunch
    from .helpers import find_shortcut_routes
    from .logger import logger
    from .state import State, CoordinatedAttack

# <--->


class _ShipyardTarget:
    def __init__(self, shipyard: Shipyard, dist_from_shipyards: int):
        self.shipyard = shipyard
        self.point = shipyard.point
        self.expected_profit = self._estimate_profit()
        self.reinforcement_distance = self._get_reinforcement_distance()
        self.total_incoming_power = self._get_total_incoming_power()
        self.distance_from_shipyards = dist_from_shipyards

    def __repr__(self):
        return f"Target {self.shipyard}"

    def estimate_shipyard_power(self, time):
        help_power = 0
        player_id = self.shipyard.player_id
        for sy in self.shipyard.board.shipyards:
            if sy.player_id != player_id or sy == self.shipyard:
                continue
            help_time = self.point.distance_from(sy.point)
            help_power += sy.estimate_shipyard_power(time - help_time - 1)

        own_power = self.shipyard.estimate_shipyard_power(time)
        return own_power + help_power

    def _get_total_incoming_power(self):
        return sum(x.ship_count for x in self.shipyard.incoming_allied_fleets)

    def _get_reinforcement_distance(self):
        incoming_allied_fleets = self.shipyard.incoming_allied_fleets
        if not incoming_allied_fleets:
            return np.inf
        return min(x.eta for x in incoming_allied_fleets)

    def _estimate_profit(self):
        board = self.shipyard.board
        spawn_cost = board.spawn_cost
        profit = sum(
            2 * x.expected_kore() - x.ship_count * spawn_cost
            for x in self.shipyard.incoming_allied_fleets
        )
        profit += spawn_cost * board.shipyard_cost
        return profit


def capture_shipyards(agent: Player, max_attack_distance: int = 10,  max_time_to_wait: int = 10):
    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ship_count >= 3 and not x.action
    ]
    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.shipyards:
        if op_sy.player_id == agent.game_id or op_sy.incoming_hostile_fleets:
            continue
        target = _ShipyardTarget(op_sy, sum(op_sy.distance_from(sy.point) for sy in agent_shipyards))
        targets.append(target)

    if not targets:
        return

    targets.sort(key=lambda x: x.distance_from_shipyards)

    my_ship_count = agent.ship_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    if my_ship_count > op_ship_count * 1.5:
        max_attack_distance = 15
    if my_ship_count > op_ship_count * 2:
        max_attack_distance = 20

    for t in targets:
        shipyards = sorted(
            agent_shipyards, key=lambda x: t.point.distance_from(x.point)
        )

        for sy in shipyards:
            if sy.action:
                continue

            distance = sy.point.distance_from(t.point)
            if distance > max_attack_distance:
                continue

            power = t.estimate_shipyard_power(distance)

            if sy.available_ship_count <= power:
                if sy.estimate_shipyard_power(max_time_to_wait) >= t.estimate_shipyard_power(distance + max_time_to_wait):
                    sy.action = DontLaunch()
                    logger.info(f"Saving for capturing shipyard {sy.point} -> {t.point}")
                continue

            num_ships_to_launch = min(sy.available_ship_count, int(power * 1.2))

            routes = find_shortcut_routes(
                board,
                sy.point,
                t.point,
                agent,
                num_ships_to_launch,
            )
            if routes:
                best_route = max(routes, key=lambda route: route.expected_kore(board, num_ships_to_launch))
                logger.info(
                    f"Attack shipyard {sy.point}->{t.point}"
                )
                sy.action = Launch(num_ships_to_launch, best_route)
                break

# Incoporate waiting also?
def coordinate_shipyard_capture(agent: Player, max_attack_distance: int = 10, send_fraction: float = 0.7):
    if not isinstance(agent.state, State):
        return

    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ship_count >= 3 and not x.action
    ]
    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.shipyards:
        if op_sy.player_id == agent.game_id or op_sy.incoming_hostile_fleets:
            continue
        target = _ShipyardTarget(op_sy, sum(op_sy.distance_from(sy.point) for sy in agent_shipyards))
        targets.append(target)

    if not targets:
        return

    targets.sort(key=lambda x: x.distance_from_shipyards)

    my_ship_count = agent.ship_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    if my_ship_count > op_ship_count * 1.5:
        max_attack_distance = 15
    if my_ship_count > op_ship_count * 2:
        max_attack_distance = 20

    for t in targets:
        shipyards = filter(
            lambda x: x.point.distance_from(t.point) <= max_attack_distance,
            agent_shipyards
        )
        shipyards = sorted(shipyards, key=lambda x: t.point.distance_from(x.point))

        loaded_attack = False
        for i in range(1, len(shipyards) + 1):
            shipyard_to_launch = {}
            total_power = 0
            max_sy_dist = max(x.point.distance_from(t.point) for x in shipyards[:i])
            last_max = shipyards[i-1].point.distance_from(t.point)
            assert last_max == max_sy_dist

            for sy in shipyards[:i]:
                wait_time = max_sy_dist - t.point.distance_from(sy.point)
                if sy.can_launch_to_at_time(t.point, wait_time):
                    power = floor(sy.estimate_shipyard_power(wait_time) * send_fraction)
                    total_power += power
                    shipyard_to_launch[sy] = (power, wait_time)

            if total_power >= t.estimate_shipyard_power(max_sy_dist):
                loaded_attack = True
                break

        if loaded_attack:
            logger.info(f"Starting coordinated attack: {t.point}, {shipyard_to_launch.items()}")
            agent.state = CoordinatedAttack(shipyard_to_launch, t.point)
            agent.update_state()
            break

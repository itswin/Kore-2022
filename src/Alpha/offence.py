from math import floor
import numpy as np
import os

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from basic import max_ships_to_spawn
    from board import Player, Shipyard, Launch, DontLaunch
    from geometry import Point
    from helpers import find_shortcut_routes, gaussian
    from logger import logger
    from state import CoordinatedAttack
else:
    from .basic import max_ships_to_spawn
    from .board import Player, Shipyard, Launch, DontLaunch
    from .geometry import Point
    from .helpers import find_shortcut_routes, gaussian
    from .logger import logger
    from .state import CoordinatedAttack

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
        for sy in self.shipyard.board.all_shipyards:
            if sy.player_id != player_id or sy == self.shipyard:
                continue
            help_time = self.point.distance_from(sy.point)
            help_power += sy.estimate_shipyard_power(time - help_time)

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

    def can_attack_from(self, point: Point) -> bool:
        if isinstance(self.shipyard, Shipyard):
            return True
        return self.shipyard.time_to_build < self.point.distance_from(point)


def capture_shipyards(agent: Player, max_attack_distance: int = 10, max_time_to_wait: int = 10):
    if isinstance(agent.state, CoordinatedAttack):
        return

    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ship_count >= 3 and not x.action
    ]
    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.all_shipyards:
        if op_sy.player_id == agent.game_id or op_sy.incoming_hostile_fleets:
            continue
        target = _ShipyardTarget(op_sy, sum(op_sy.distance_from(sy.point) for sy in agent_shipyards))
        targets.append(target)

    if not targets:
        return

    targets.sort(key=lambda x: x.distance_from_shipyards)

    my_ship_count = agent.ship_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    if my_ship_count > 100:
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

            if not t.can_attack_from(sy.point):
                continue

            distance = sy.distance_from(t.point)
            if distance > max_attack_distance:
                continue

            power = t.estimate_shipyard_power(distance)

            if sy.available_ship_count <= power:
                if sy.estimate_shipyard_power(max_time_to_wait) >= t.estimate_shipyard_power(distance + max_time_to_wait):
                    sy.action = DontLaunch()
                    logger.info(f"Saving for capturing shipyard {sy.point} -> {t.point}")
                continue

            num_ships_to_launch = min(sy.available_ship_count, max(int(power * 1.2), 21))

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
                    f"Attack shipyard {sy.point}->{t.point} with {num_ships_to_launch} > {power}"
                )
                sy.action = Launch(num_ships_to_launch, best_route)
                break
            else:
                logger.info(f"No routes for {sy.point}->{t.point}")


# Incoporate waiting also?
def coordinate_shipyard_capture(agent: Player, max_attack_distance: int = 10, send_fraction: float = 1):
    if agent.update_state_if_is(CoordinatedAttack):
        return

    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ship_count >= 3 and not x.action
    ]
    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.all_shipyards:
        if op_sy.player_id == agent.game_id or op_sy.incoming_hostile_fleets:
            continue
        target = _ShipyardTarget(op_sy, sum(op_sy.distance_from(sy.point) for sy in agent_shipyards))
        targets.append(target)

    if not targets:
        return

    targets.sort(key=lambda x: x.distance_from_shipyards)

    my_ship_count = agent.ship_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    mult_factor = 1
    if my_ship_count > 100:
        if my_ship_count > op_ship_count * 1.5:
            max_attack_distance = 15
            mult_factor = 1.5
        if my_ship_count > op_ship_count * 2:
            max_attack_distance = 20
            mult_factor = 2

    for t in targets:
        shipyards = filter(
            lambda x: x.point.distance_from(t.point) <= max_attack_distance,
            agent_shipyards
        )
        shipyards = sorted(shipyards, key=lambda x: t.point.distance_from(x.point))

        loaded_attack = False
        for i in range(2, len(shipyards) + 1):
            shipyard_to_launch = {}
            total_power = 0
            max_sy_dist = max(x.distance_from(t.point) for x in shipyards[:i])
            last_max = shipyards[i-1].distance_from(t.point)
            assert last_max == max_sy_dist

            j = 0
            num_shipyards_used = 0
            while num_shipyards_used < i and j < len(shipyards):
                sy = shipyards[j]
                wait_time = max_sy_dist - t.point.distance_from(sy.point)
                if sy.can_launch_to_at_time(t.point, wait_time) and t.can_attack_from(sy.point):
                    power = floor(sy.estimate_shipyard_power(wait_time) * send_fraction)
                    routes = find_shortcut_routes(
                        board,
                        sy.point,
                        t.point,
                        agent,
                        power,
                        allow_join=True
                    )

                    if routes:
                        total_power += power
                        shipyard_to_launch[sy] = (power, wait_time)
                        num_shipyards_used += 1
                j += 1

            if num_shipyards_used != i:
                break
            # logger.info(f"Coordinate {i} shipyards {t.point}, {total_power}, {t.estimate_shipyard_power(max_sy_dist)}")
            if total_power >= t.estimate_shipyard_power(max_sy_dist) or \
                    (my_ship_count > mult_factor * op_ship_count and \
                    total_power * mult_factor >= t.estimate_shipyard_power(max_sy_dist)):
                loaded_attack = True
                break

        if loaded_attack:
            logger.info(f"Starting coordinated attack: {t.point}, {shipyard_to_launch.items()}")
            agent.state = CoordinatedAttack(shipyard_to_launch, t.point)
            agent.update_state()
            break


WHITTLE_COOLDOWN = 20
last_whittle_attack = -WHITTLE_COOLDOWN

def should_whittle_attack(agent: Player, step: int, min_overage: int = 25):
    board = agent.board
    global last_whittle_attack
    my_ship_count = agent.ship_count
    my_shipyard_count = len(agent.shipyards)

    op_shipyard_positions = {
        x.point for x in board.all_shipyards if x.player_id != agent.game_id
    }
    attacking_count = sum(
        (x.ship_count for x in agent.fleets if x.route.end in op_shipyard_positions),
        start=0
    )

    available_ships = my_ship_count - attacking_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    op_shipyard_count = max(len(x.shipyards) for x in agent.opponents)

    # if step - last_whittle_attack < WHITTLE_COOLDOWN:
    #     return False

    return available_ships > 100 and \
            my_shipyard_count >= 2 and \
            available_ships - min_overage > op_ship_count


def whittle_attack(agent: Player, step: int,
    max_attack_distance: int = 20, max_time_to_wait: int = 10, whittle_power: int = 50
):
    global last_whittle_attack
    if isinstance(agent.state, CoordinatedAttack):
        return

    if not should_whittle_attack(agent, step):
        return

    board = agent.board
    agent_shipyards = [
        x for x in agent.shipyards if x.available_ship_count >= 3 and not x.action
    ]
    if not agent_shipyards:
        return

    targets = []
    for op_sy in board.all_shipyards:
        if op_sy.player_id == agent.game_id or op_sy.incoming_hostile_fleets:
            continue
        target = _ShipyardTarget(op_sy, sum(op_sy.distance_from(sy.point) for sy in agent_shipyards))
        targets.append(target)

    if not targets:
        return

    kore_mu = 0
    kore_sigma = 5
    gaussian_mid = gaussian(0, kore_mu, kore_sigma)
    targets.sort(key=lambda sy: sum(
        (x.kore ** 1.1) * gaussian(sy.point.distance_from(x), kore_mu, kore_sigma) / gaussian_mid
        for x in sy.point.nearby_points(10)
    ))

    # my_ship_count = agent.ship_count
    # op_ship_count = max(x.ship_count for x in agent.opponents)
    # if my_ship_count > 100:
    #     if my_ship_count > op_ship_count * 1.5:
    #         max_attack_distance = 15
    #     if my_ship_count > op_ship_count * 2:
    #         max_attack_distance = 20

    attacked = False
    for t in targets:
        shipyards = sorted(
            agent_shipyards, key=lambda x: t.point.distance_from(x.point)
        )

        for sy in shipyards:
            if sy.action:
                continue

            if not t.can_attack_from(sy.point):
                continue

            distance = sy.distance_from(t.point)
            if distance > max_attack_distance:
                continue

            if sy.available_ship_count <= whittle_power:
                if sy.estimate_shipyard_power(max_time_to_wait) >= t.estimate_shipyard_power(distance + max_time_to_wait):
                    sy.action = DontLaunch()
                    logger.info(f"Saving for whittle attack {sy.point} -> {t.point}")
                continue

            num_ships_to_launch = min(max(whittle_power, int(agent.ship_count / 10)), sy.available_ship_count)
            routes = find_shortcut_routes(
                board,
                sy.point,
                t.point,
                agent,
                num_ships_to_launch,
                max_route_distance=min(distance+4, max_attack_distance)
            )

            if routes:
                best_route = min(routes, key=lambda route: route.expected_kore(board, num_ships_to_launch))
                logger.info(
                    f"Whittle attack shipyard {sy.point}->{t.point}"
                )
                sy.action = Launch(num_ships_to_launch, best_route)
                attacked = True
                break
            else:
                logger.info(f"Whittle: No routes for {sy.point}->{t.point}")

        if attacked:
            last_whittle_attack = step
            # break

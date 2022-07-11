from collections import defaultdict
from math import floor
import numpy as np
import os

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from basic import max_ships_to_spawn
    from board import Player, Shipyard, Launch, FutureShipyard
    from geometry import Point
    from helpers import find_shortcut_routes, _spawn
    from logger import logger
    from state import CoordinatedAttack, PrepCoordinatedAttack, State
else:
    from .basic import max_ships_to_spawn
    from .board import Player, Shipyard, Launch, FutureShipyard
    from .geometry import Point
    from .helpers import find_shortcut_routes, _spawn
    from .logger import logger
    from .state import CoordinatedAttack, PrepCoordinatedAttack, State

# <--->


class _ShipyardTarget:
    def __init__(self, shipyard: Shipyard, dist_from_shipyards: int):
        self.shipyard = shipyard
        self.point = shipyard.point
        self.distance_from_shipyards = dist_from_shipyards

    def __repr__(self):
        return f"Target {self.shipyard}"

    @property
    def max_ships_to_spawn(self) -> int:
        if isinstance(self.shipyard, Shipyard):
            return self.shipyard.max_ships_to_spawn
        return 1

    def estimate_shipyard_power(self, time):
        help_power = 0
        board = self.shipyard.board
        player_id = self.shipyard.player_id
        player = board.get_player(player_id)

        time_to_fleet_kore = defaultdict(int)
        shipyard_reinforcements = defaultdict(lambda: defaultdict(int))
        for sy in player.all_shipyards:
            for f in sy.incoming_allied_fleets:
                time_to_fleet_kore[f.eta] += f.expected_kore()
                shipyard_reinforcements[sy][f.eta] += f.ship_count
            for f in sy.incoming_hostile_fleets:
                shipyard_reinforcements[sy][f.eta] -= f.ship_count

        spawn_cost = board.spawn_cost
        player_kore = player.kore
        own_power = self.shipyard.ship_count
        help_power = 0
        for t in range(0, time):
            # Prioritize spawning at the target
            player_kore += time_to_fleet_kore[t]
            own_power += shipyard_reinforcements[self.shipyard][t]
            can_spawn = max_ships_to_spawn(self.shipyard.turns_controlled + t)
            spawn_count = min(int(player_kore // spawn_cost), can_spawn)
            player_kore -= spawn_count * spawn_cost
            own_power += spawn_count

            for sy in player.all_shipyards:
                if sy == self.shipyard:
                    continue
                if isinstance(sy, FutureShipyard) and sy.time_to_build < t:
                    continue

                help_time = sy.distance_from(self.shipyard)
                if time - t < help_time:
                    continue
                if t == 0 or (isinstance(sy, FutureShipyard) and sy.time_to_build == t):
                    help_power += sy.ship_count

                can_spawn = max_ships_to_spawn(sy.turns_controlled + t)
                spawn_count = min(int(player_kore // spawn_cost), can_spawn)
                player_kore -= spawn_count * spawn_cost
                help_power += shipyard_reinforcements[sy][t]
                if time - t != help_time:
                    help_power += spawn_count

        # logger.error(f"{self.shipyard.point}, Time: {time} Own power: {own_power}, Help power: {help_power}")
        return own_power + help_power

    def can_attack_from(self, point: Point) -> bool:
        if isinstance(self.shipyard, Shipyard):
            return True
        return self.shipyard.time_to_build <= self.point.distance_from(point)


def capture_shipyards(agent: Player, max_attack_distance: int = 10, max_time_to_wait: int = 10):
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

    targets.sort(key=lambda x: (-x.max_ships_to_spawn, x.distance_from_shipyards))

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
                my_power = sy.estimate_shipyard_power(max_time_to_wait)
                op_power = t.estimate_shipyard_power(distance + max_time_to_wait)
                if my_power >= op_power:
                    _spawn(agent, sy)
                    logger.info(f"Saving for capturing shipyard {sy.point} -> {t.point}. {my_power} > {op_power}")
                else:
                    # logger.info(f"NOT saving for capturing shipyard {sy.point} -> {t.point}. {my_power} > {op_power}")
                    pass
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
    was_prepping = False
    prepped_target = None
    if isinstance(agent.state, PrepCoordinatedAttack):
        prepped_target = agent.state.target
    if agent.update_state_if_is(PrepCoordinatedAttack):
        if not isinstance(agent.state, State):
            return
        was_prepping = True
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

    targets.sort(key=lambda x: (-x.max_ships_to_spawn, x.distance_from_shipyards))

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

    best_sy_to_launch = None
    best_attack_diff = None
    best_t = None

    if was_prepping:
        max_attack_distance = max(max_attack_distance, 20)

    my_shipyard_count = len(agent.shipyards)
    op_shipyard_count = max(len(x.shipyards) for x in agent.opponents)
    op_future_shipyard_count = max(len(x.future_shipyards) for x in agent.opponents)

    # if my_shipyard_count > 1 and my_shipyard_count <= op_shipyard_count \
    #     and op_future_shipyard_count > 0:
    #     op_future_sy = agent.opponents[0].future_shipyards[0]
    #     shipyards = list(filter(
    #         lambda x: x.point.distance_from(op_future_sy.point) <= max_attack_distance,
    #         agent_shipyards
    #     ))
    #     if len(shipyards) >= my_shipyard_count // 2:
    #         logger.info("Prepping coordinated attack")
    #         agent.state = PrepCoordinatedAttack(8)
    #         agent.update_state()
    #         was_prepping = isinstance(agent.state, State)

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
            power_est = t.estimate_shipyard_power(max_sy_dist)
            # logger.info(f"Coordinate {i} shipyards {t.point}, {total_power}, {t.estimate_shipyard_power(max_sy_dist)}")
            if total_power >= power_est or \
                    (my_ship_count > mult_factor * op_ship_count and \
                    total_power * mult_factor >= power_est):
                loaded_attack = True
                break
            attack_diff = total_power - power_est
            if best_attack_diff is None or attack_diff > best_attack_diff and \
                (prepped_target is None or t.point == prepped_target):
                best_attack_diff = attack_diff
                best_sy_to_launch = shipyard_to_launch
                best_t = t

        if loaded_attack:
            logger.info(f"Starting coordinated attack: {t.point}, {shipyard_to_launch.items()}")
            agent.state = CoordinatedAttack(shipyard_to_launch, t.point)
            agent.update_state()
            break

    if was_prepping and best_sy_to_launch is not None:
        logger.info(f"Starting forced coordinated attack: {best_t.point}, {shipyard_to_launch.items()}")
        agent.state = CoordinatedAttack(best_sy_to_launch, best_t.point)
        agent.update_state()


WHITTLE_COOLDOWN = 20
last_whittle_attack = -WHITTLE_COOLDOWN

def should_whittle_attack(agent: Player, step: int, min_overage: int = 50):
    global last_whittle_attack
    board = agent.board
    my_ship_count = agent.ship_count
    my_shipyard_count = len(agent.shipyards)

    op_shipyard_positions = {
        x.point for x in board.all_shipyards if x.player_id != agent.game_id
    }
    attacking_count = sum(
        x.ship_count for x in agent.fleets if x.route.end in op_shipyard_positions
    )

    available_ships = my_ship_count - attacking_count
    op_ship_count = max(x.ship_count for x in agent.opponents)
    op_shipyard_count = max(len(x.shipyards) for x in agent.opponents)

    # if step - last_whittle_attack < WHITTLE_COOLDOWN:
    #     return False

    return available_ships > 100 and \
            available_ships - min_overage > op_ship_count


def whittle_attack(agent: Player, step: int,
    max_attack_distance: int = 10, max_time_to_wait: int = 3, whittle_power: int = 50
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

    targets.sort(key=lambda x: x.distance_from_shipyards)

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
            agent_shipyards, key=lambda x: t.point.distance_from(x.point) + x.calc_time_for_ships_for_action(50)
        )

        for sy in shipyards:
            if sy.action:
                continue

            if not t.can_attack_from(sy.point):
                continue

            distance = sy.distance_from(t.point)
            if distance > max_attack_distance:
                continue

            if sy.available_ship_count < whittle_power:
                if sy.estimate_shipyard_power(max_time_to_wait) >= whittle_power:
                    _spawn(agent, sy)
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
                best_route = min(routes, key=lambda route: (route.expected_kore(board, num_ships_to_launch), len(route)))
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

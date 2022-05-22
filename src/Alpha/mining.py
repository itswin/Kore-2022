import random
import numpy as np
import os
from typing import List, Dict
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import PlanRoute, Point
    from board import Player, BoardRoute, Launch, Shipyard, MiningRoute
    from helpers import is_intercept_route, find_closest_shipyards
    from logger import logger
else:
    from .geometry import PlanRoute, Point
    from .board import Player, BoardRoute, Launch, Shipyard, MiningRoute
    from .helpers import is_intercept_route, find_closest_shipyards
    from .logger import logger

# <--->


def mine(agent: Player):
    board = agent.board
    if not agent.opponents:
        return

    safety = False
    my_ship_count = agent.ship_count

    if my_ship_count < 21 and agent.kore > board.spawn_cost:
        return

    op_ship_count = max(x.ship_count for x in agent.opponents)
    if my_ship_count < 2 * op_ship_count:
        safety = True

    op_ship_count = []
    for op in agent.opponents:
        for fleet in op.fleets:
            op_ship_count.append(fleet.ship_count)

    if not op_ship_count:
        mean_fleet_size = 0
        max_fleet_size = np.inf
    else:
        mean_fleet_size = np.percentile(op_ship_count, 75)
        max_fleet_size = int(max(op_ship_count) * 1.1)

    point_to_time_to_score = estimate_board_risk(agent)

    shipyard_count = len(agent.shipyards)
    if shipyard_count < 10:
        max_distance = 15
    elif shipyard_count < 20:
        max_distance = 12
    else:
        max_distance = 8

    max_distance = min(int(board.steps_left // 2), max_distance)

    for sy in agent.shipyards:
        if sy.action:
            continue

        free_ships = sy.available_ship_count

        if free_ships <= 2:
            continue

        routes = find_shipyard_mining_routes(
            sy, safety=safety, max_distance=max_distance
        )

        route_to_score = {}
        for route in routes:
            route_points = route.points()

            worst_score = min(point_to_time_to_score[p][t] for t, p in enumerate(route_points))
            if worst_score > 0:
                num_ships_to_launch = free_ships
                score_penalty = 1
            else:
                if free_ships < mean_fleet_size:
                    continue
                num_ships_to_launch = min(free_ships, max_fleet_size)
                opp_adv = -(free_ships + worst_score)
                score_penalty = 0 if opp_adv >= 0 else 1

            score = route.expected_kore(board, num_ships_to_launch) / len(route) * score_penalty
            route_to_score[route] = (score, worst_score)

        if not route_to_score:
            continue

        best_route = max(route_to_score, key=lambda x: route_to_score[x][0])
        worst_score = route_to_score[best_route][1]
        if worst_score > 0:
            num_ships_to_launch = free_ships
        else:
            num_ships_to_launch = min(free_ships, 199)
        if best_route.can_execute():
            logger.info(f"Mining Route: {best_route.plan}, {route_to_score[best_route]}")
            sy.action = Launch(num_ships_to_launch, best_route)
        else:
            logger.info(f"Waiting for route: {best_route.plan}, {route_to_score[best_route]}")


# Note: Does not take into account getting a ship next to yours for damage
def estimate_board_risk(player: Player) -> Dict[Point, Dict[int, int]]:
    board = player.board

    point_to_time_to_score = defaultdict(dict)
    for p in board:
        (closest_friendly_sy,
         closest_enemy_sy,
         min_friendly_distance,
         min_enemy_distance) = find_closest_shipyards(player, p)

        for t in range(0, 2 * board.size):
            if min_friendly_distance + t < min_enemy_distance:
                point_to_time_to_score[p][t] = 1
            else:
                dt = min_friendly_distance + t - min_enemy_distance
                point_to_time_to_score[p][t] = -closest_enemy_sy.estimate_shipyard_power(dt)

    return point_to_time_to_score


def find_shipyard_mining_routes(
    sy: Shipyard, safety=True, max_distance: int = 15
) -> List[BoardRoute]:
    if max_distance < 1:
        return []

    departure = sy.point
    player = sy.player

    destinations = set()
    for shipyard in sy.player.shipyards:
        siege = sum(x.ship_count for x in shipyard.incoming_hostile_fleets)
        if siege >= shipyard.ship_count:
            continue
        destinations.add(shipyard.point)

    if not destinations:
        return []

    routes = []
    for c in sy.point.nearby_points(max_distance):
        if c == departure or c in destinations:
            continue

        destination = min(destinations, key=lambda x: c.distance_from(x))
        plans = departure.get_plans_through([c, destination])

        for plan in plans:
            wait_time = sy.calc_time_for_ships(plan.min_fleet_size())
            route = MiningRoute(departure, plan, wait_time)

            if is_intercept_route(route, player, safety):
                continue

            routes.append(MiningRoute(departure, plan, wait_time))

    return routes

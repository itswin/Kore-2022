import random
import numpy as np
import os
from typing import List
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import PlanRoute
    from board import Player, BoardRoute, Launch, Shipyard, MiningRoute
    from helpers import is_intercept_route
    from logger import logger
else:
    from .geometry import PlanRoute
    from .board import Player, BoardRoute, Launch, Shipyard, MiningRoute
    from .helpers import is_intercept_route
    from .logger import logger

# <--->


def mine(agent: Player):
    board = agent.board
    if not agent.opponents:
        return

    safety = False
    my_ship_count = agent.ship_count

    if my_ship_count < 21:
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

    point_to_score = estimate_board_risk(agent)

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

            worst_score = min(point_to_score[p] for p in route_points)
            if worst_score <= 0:
                num_ships_to_launch = free_ships
                score_penalty = 1
            else:
                if free_ships < mean_fleet_size:
                    continue
                num_ships_to_launch = min(free_ships, max_fleet_size)
                opp_adv = -(free_ships + worst_score)
                score_penalty = 0 if opp_adv > 0 else 1

            score = route.expected_kore(board, num_ships_to_launch) / len(route) * score_penalty
            route_to_score[route] = score

        if not route_to_score:
            continue

        best_route = max(route_to_score, key=lambda x: route_to_score[x])
        if all(point_to_score[x] >= 1 for x in best_route):
            num_ships_to_launch = free_ships
        else:
            num_ships_to_launch = min(free_ships, 199)
        if best_route.can_execute():
            logger.debug(f"Mining Route: {best_route.plan}, {route_to_score[best_route]}")
            sy.action = Launch(num_ships_to_launch, best_route)
        else:
            logger.debug(f"Waiting for route: {best_route.plan}, {route_to_score[best_route]}")


def estimate_board_risk(player: Player):
    board = player.board

    shipyard_to_area = defaultdict(list)
    for p in board:
        closest_shipyard = None
        min_distance = board.size
        for sh in board.shipyards:
            distance = sh.point.distance_from(p)
            if distance < min_distance:
                closest_shipyard = sh
                min_distance = distance

        shipyard_to_area[closest_shipyard].append(p)

    point_to_score = {}
    for sy, points in shipyard_to_area.items():
        if sy.player_id == player.game_id:
            for p in points:
                point_to_score[p] = 1
        else:
            for p in points:
                t = p.distance_from(sy.point)
                point_to_score[p] = -sy.estimate_shipyard_power(t + 1)

    return point_to_score


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

        paths = departure.dirs_to(c)
        plans1 = [PlanRoute(path) for path in paths]
        destination = sorted(destinations, key=lambda x: c.distance_from(x))[0]
        plans2 = []
        if destination == departure:
            # @time_save 25%
            # Don't consider rectangle paths in both directions
            # The only difference is intercept timings.
            plans2 = [plan.reverse() for plan in plans1]
        else:
            paths2 = c.dirs_to(destination)
            plans2 = [PlanRoute(path) for path in paths2]

        plans = []
        for plan1 in plans1:
            for plan2 in plans2:
                plans.append(plan1 + plan2)

        for plan in plans:
            wait_time = sy.calc_time_for_ships(plan.min_fleet_size())
            route = MiningRoute(departure, plan, wait_time)

            if is_intercept_route(route, player, safety):
                continue

            routes.append(MiningRoute(departure, plan, wait_time))

    return routes

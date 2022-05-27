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
    from helpers import is_intercept_route, find_closest_shipyards
    from logger import logger
else:
    from .geometry import PlanRoute
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

    shipyard_count = len(agent.shipyards)
    if shipyard_count < 10:
        max_distance = 15
    elif shipyard_count < 20:
        max_distance = 12
    else:
        max_distance = 8

    max_distance = min(int(board.steps_left // 2), max_distance)
    can_deplete_kore_fast = agent.shipyard_production_capacity * board.spawn_cost * 5 > agent.kore

    for sy in agent.shipyards:
        if sy.action:
            continue

        free_ships = sy.available_ship_count

        if free_ships <= 2:
            continue

        routes = find_shipyard_mining_routes(
            sy, safety=safety, max_distance=max_distance
        )

        route_to_info = {}
        for route in routes:
            route_points = route.points()

            board_risk = max(agent.estimate_board_risk(p, t + 1) for t, p in enumerate(route_points))
            num_ships_to_launch = route.plan.min_fleet_size() if can_deplete_kore_fast else free_ships
            if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                num_ships_to_launch = min(free_ships, board_risk + 1)
                if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                    continue

            score = route.expected_kore(board, num_ships_to_launch) / len(route)
            route_to_info[route] = (score, num_ships_to_launch, board_risk)

        if not route_to_info:
            continue

        # items = sorted(route_to_info.items(), key=lambda x: x[1], reverse=True)
        # for i in range(0, 5):
        #     route = items[i][0]
        #     score, num_ships_to_launch, board_risk = route_to_info[route]
        #     logger.info(f"Mining Route: {route.plan}, {score}, {board_risk}")

        best_route = max(route_to_info, key=lambda x: route_to_info[x][0])
        # for t, p in enumerate(best_route.points()):
        #     logger.info(f"{p} {t}, ")
        score, num_ships_to_launch, board_risk = route_to_info[best_route]
        if best_route.can_execute():
            logger.info(f"Mining Route: {best_route.plan}, {score}, {board_risk}")
            sy.action = Launch(num_ships_to_launch, best_route)
        else:
            logger.info(f"Waiting for Route: {best_route.plan}, {score}, {board_risk}")



def find_shipyard_mining_routes(
    sy: Shipyard, safety=True, max_distance: int = 15
) -> List[BoardRoute]:
    if max_distance < 1:
        return []

    departure = sy.point
    player = sy.player

    def get_destinations(shipyards):
        destinations = set()
        for shipyard in shipyards:
            siege = sum(x.ship_count for x in shipyard.incoming_hostile_fleets)
            if siege >= shipyard.ship_count:
                continue
            destinations.add(shipyard)
        return destinations

    destinations = get_destinations(sy.player.shipyards)
    future_destinations = get_destinations(sy.player.future_shipyards)

    if not destinations:
        return []

    routes = []
    for c in sy.point.nearby_points(max_distance):
        if c == departure or c in destinations:
            continue

        dest_sy = min(destinations, key=lambda x: c.distance_from(x.point))
        future_dest_sys = list(filter(
            lambda x: x.time_to_build <= x.point.distance_from(c) + sy.point.distance_from(c),
            future_destinations
        ))

        if future_dest_sys:
            future_dest_sy = min(future_dest_sys, key=lambda x: c.distance_from(x.point))
            dest_sy = min([dest_sy, future_dest_sy], key=lambda x: c.distance_from(x.point))

        destination = dest_sy.point
        plans = departure.get_plans_through([c, destination])

        for plan in plans:
            wait_time = sy.calc_time_for_ships(plan.min_fleet_size())
            route = MiningRoute(departure, plan, wait_time)

            if is_intercept_route(route, player, safety):
                continue

            routes.append(MiningRoute(departure, plan, wait_time))

    return routes

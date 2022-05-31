import random
import itertools
import math
import numpy as np
import os
from typing import List
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import PlanRoute, Point
    from board import Player, BoardRoute, Launch, Shipyard, MiningRoute, Board, AllowMine
    from helpers import is_intercept_route, find_closest_shipyards
    from logger import logger
else:
    from .geometry import PlanRoute, Point
    from .board import Player, BoardRoute, Launch, Shipyard, MiningRoute, Board, AllowMine
    from .helpers import is_intercept_route, find_closest_shipyards
    from .logger import logger

# <--->


def mine(agent: Player, remaining_time: float, step: int):
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

    shipyard_count = len(agent.all_shipyards)
    if shipyard_count < 10:
        max_distance = 15
    elif shipyard_count < 20:
        max_distance = 12
    else:
        max_distance = 8

    max_distance = min(int(board.steps_left // 2), max_distance)
    shipyard_production_capacity = agent.shipyard_production_capacity
    num_turns_to_deplete_kore = agent.kore / (board.spawn_cost * shipyard_production_capacity) if shipyard_production_capacity > 0 else 500
    can_deplete_kore_fast = num_turns_to_deplete_kore < 5
    use_second_points = len(agent.all_shipyards) < 10 and remaining_time > 30

    def score_route(route: BoardRoute, num_ships_to_launch: int) -> float:
        # Don't do short routes if we need to spawn
        if num_turns_to_deplete_kore > 1 and len(route) == 2:
            return 0

        exp_kore = route.expected_kore(board, num_ships_to_launch)
        if my_ship_count < 50:
            if exp_kore < 10:
                return 0

        dist_penalty = sy.get_idle_turns_before(len(route)) / 4.0 if can_deplete_kore_fast else 0
        return exp_kore / len(route) - 0

    logger.info(f"Can deplete kore fast: {can_deplete_kore_fast} in {num_turns_to_deplete_kore}")
    for sy in agent.shipyards:
        sy_max_dist = max_distance

        if sy.action:
            if not isinstance(sy.action, AllowMine):
                continue
            sy_max_dist = sy.action.max_distance

        free_ships = sy.available_ship_count

        if free_ships <= 2:
            continue

        routes = find_shipyard_mining_routes(
            sy, safety=safety, max_distance=sy_max_dist, use_second_points=use_second_points
        )

        route_to_info = {}
        for route in routes:
            route_points = route.points()

            board_risk = max(agent.estimate_board_risk(p, t + 1 + route.time_to_mine) for t, p in enumerate(route_points))
            num_ships_to_launch = route.plan.min_fleet_size() if can_deplete_kore_fast else free_ships
            # Prevent sending huge long routes
            if route.plan.min_fleet_size() > 21 and \
                (num_ships_to_launch > 0.2 * agent.ship_count or \
                 num_ships_to_launch > 0.5 * sy.estimate_shipyard_power(10)):
                continue
            if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                num_ships_to_launch = min(free_ships, board_risk + 1)
                if step < 100 or not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                    continue

            score = score_route(route, num_ships_to_launch)
            route_to_info[route] = (score, num_ships_to_launch, board_risk)

        route_to_info = {k:v for k,v in route_to_info.items() if v[0] >= 0}
        sorted_routes = sorted(
            route_to_info.items(),
            key=lambda x: x[1], reverse=True
        )
        if not route_to_info:
            continue

        # l = min(10, len(sorted_routes))
        # for i in range(0, l):
        #     route = sorted_routes[i][0]
        #     score, num_ships_to_launch, board_risk = route_to_info[route]
        #     logger.info(f"Mining Route: {route.plan}, {len(route)}, {score}, {board_risk}")

        best_route = choose_route(sy, free_ships, sorted_routes)
        # best_route = max(route_to_info, key=lambda x: route_to_info[x][0])
        # for t, p in enumerate(best_route.points()):
        #     logger.info(f"{p} {t}, ")
        score, num_ships_to_launch, board_risk = route_to_info[best_route]
        if best_route.can_execute():
            logger.info(f"Mining Route: {best_route.plan}, {score}, {board_risk}, {best_route.expected_kore(board, num_ships_to_launch)} {num_ships_to_launch}")
            sy.action = Launch(num_ships_to_launch, best_route)
        else:
            logger.info(f"Waiting for Route: {best_route.plan}, {score}, {board_risk}")


# Choose a route out of the top 10.
# If you can launch multiple of them, add their scores together.
def choose_route(sy, num_ships, sorted_routes, max_routes: int = 10):
    best_route = sorted_routes[0][0]
    score, num_ships_to_launch, board_risk = sorted_routes[0][1]
    if best_route.can_execute():
        return best_route

    max_size = min(num_ships, sy.estimate_shipyard_power_before_action(best_route.time_to_mine) - num_ships_to_launch)
    temp_route = find_first_route_of_size(sorted_routes, max_size)
    if temp_route is not None:
        logger.info(f"Waiting for route {best_route.plan}, {score}, {board_risk} but can launch one before it with {max_size}")
        return temp_route

    return best_route


def find_first_route_of_size(routes, size):
    for route, (score, num_ships_to_launch, board_risk) in routes:
        if num_ships_to_launch <= size:
            return route
    return None


def find_shipyard_mining_routes(
    sy: Shipyard, safety=True, max_distance: int = 15, use_second_points: int = False
) -> List[BoardRoute]:
    if max_distance < 1:
        return []

    departure = sy.point
    player = sy.player
    board = player.board

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

    route_set = set()
    routes = []
    if use_second_points:
        for c in sy.point.nearby_points(max_distance):
            if c == departure or any(c == x.point for x in destinations):
                continue

            adjs = c.nine_adjacent_points
            for adj in adjs:
                if adj == departure or any(adj == x.point for x in destinations):
                    continue
                cs = [c] if adj == c else [c, adj]

                dist_through_cs = sum(c.distance_from(d) for c, d in zip(cs, cs[1:])) + sy.distance_from(cs[0])
                dest_sy = min(destinations, key=lambda x: cs[-1].distance_from(x.point))
                future_dest_sys = list(filter(
                    lambda x: x.time_to_build <= dist_through_cs + x.distance_from(cs[-1]),
                    future_destinations
                ))

                if future_dest_sys:
                    future_dest_sy = min(future_dest_sys, key=lambda x: x.distance_from(cs[-1]))
                    if future_dest_sy.point != c and future_dest_sy.point != adj:
                        dest_sy = min([dest_sy, future_dest_sy], key=lambda x: x.distance_from(cs[-1]))

                destination = dest_sy.point
                points = [departure] + cs + [destination]
                best_plan = get_greedy_mining_plan_through(points, board)
                wait_time = sy.calc_time_for_ships(best_plan.min_fleet_size())
                route = MiningRoute(departure, best_plan, wait_time)

                if route.plan.to_str() in route_set:
                    continue

                if is_intercept_route(route, player, safety):
                    continue

                routes.append(route)
                route_set.add(route.plan.to_str())
    else:
        for c in sy.point.nearby_points(max_distance):
            if c == departure or any(c == x.point for x in destinations):
                continue

            dest_sy = min(destinations, key=lambda x: c.distance_from(x.point))
            future_dest_sys = list(filter(
                lambda x: x.time_to_build <= x.distance_from(c) + sy.distance_from(c),
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

                if route.plan.to_str() in route_set:
                    continue

                if is_intercept_route(route, player, safety):
                    continue

                routes.append(MiningRoute(departure, plan, wait_time))
                route_set.add(route.plan.to_str())

    return routes


def get_greedy_mining_plan_through(points: List["Point"], board: Board) -> PlanRoute:
    last = points[0]
    plan = PlanRoute([])
    for p in points[1:]:
        plans = last.dirs_to(p)
        paths = [PlanRoute(plan) for plan in plans]
        best_route = max(
            (BoardRoute(last, path, plan.num_steps) for path in paths),
            key=lambda x: x.expected_kore(board, 20)
        )
        plan += best_route.plan
        last = p

    return plan

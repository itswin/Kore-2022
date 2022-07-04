import random
import itertools
import numpy as np
import os
from typing import List
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import PlanRoute, Point
    from board import Player, BoardRoute, Launch, Shipyard, MiningRoute, Board, AllowMine
    from helpers import is_intercept_route, find_closest_shipyards, _spawn
    from logger import logger
else:
    from .geometry import PlanRoute, Point
    from .board import Player, BoardRoute, Launch, Shipyard, MiningRoute, Board, AllowMine
    from .helpers import is_intercept_route, find_closest_shipyards, _spawn
    from .logger import logger

# <--->


def mine(agent: Player, remaining_time: float):
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

    if board.step < 50 and not op_ship_count and agent.kore > board.spawn_cost:
        return

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
    # use_second_points = len(agent.all_shipyards) < 10 and remaining_time > 30
    use_second_points = False

    fleet_distance = []
    for sy in agent.all_shipyards:
        for f in sy.incoming_allied_fleets:
            fleet_distance.append(len(f.route))

    fleet_distance = fleet_distance or [1]
    mean_fleet_distance = sum(fleet_distance) / len(fleet_distance)
    target_mean_distance = 10

    def score_route(route: BoardRoute, num_ships_to_launch: int) -> float:
        # Don't do short routes if we need to spawn
        if num_turns_to_deplete_kore > 1 and len(route) == 2:
            return 0

        exp_kore = route.expected_kore_mining(board, num_ships_to_launch)
        if my_ship_count < 50:
            if exp_kore < 10:
                return 0

        dist_penalty = sy.get_idle_turns_before(len(route)) / 20.0 if can_deplete_kore_fast else 0
        dist_bonus = 0
        return exp_kore / len(route) - dist_penalty + dist_bonus

    for sy in agent.shipyards:
        sy_max_dist = max_distance
        if sy.action:
            if not isinstance(sy.action, AllowMine):
                continue
            sy_max_dist = sy.action.max_distance

        free_ships = sy.available_ship_count if my_ship_count < 105 else min(sy.available_ship_count, my_ship_count // 5)

        if free_ships <= 2:
            continue

        routes = find_shipyard_mining_routes(
            sy, safety=safety, max_distance=sy_max_dist, use_second_points=use_second_points
        )

        route_to_info = {}
        for route in routes:
            route_points = route.points()

            min_fleet_size = route.plan.min_fleet_size()
            board_risk = max(agent.estimate_board_risk(p, t + 1 + route.time_to_mine) for t, p in enumerate(route_points))
            num_ships_to_launch = min_fleet_size \
                if can_deplete_kore_fast and not len(route) == 2 \
                else free_ships
            # Prevent sending huge long routes
            if min_fleet_size > 21 and \
                (num_ships_to_launch > 0.2 * agent.ship_count or \
                 num_ships_to_launch > 0.5 * sy.estimate_shipyard_power(10)):
                continue
            if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                num_ships_to_launch = min(free_ships, board_risk + 1)
                if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                    continue

            score = score_route(route, num_ships_to_launch)
            route_to_info[route] = (score, num_ships_to_launch, board_risk)

        if not route_to_info:
            continue

        # items = sorted(route_to_info.items(), key=lambda x: x[1], reverse=True)
        # for i in range(0, min(len(items), 10)):
        #     route = items[i][0]
        #     score, num_ships_to_launch, board_risk = route_to_info[route]
        #     logger.info(f"{sy.point} Mining Route: {route.plan}, {score}, {board_risk}")

        best_route = max(route_to_info, key=lambda x: route_to_info[x][0])
        # for t, p in enumerate(best_route.points()):
        #     logger.info(f"{p} {t}, {agent.estimate_board_risk(p, t + 1 + best_route.time_to_mine)}")
        score, num_ships_to_launch, board_risk = route_to_info[best_route]
        if should_not_launch_small_fleet(
            agent, best_route, can_deplete_kore_fast, num_ships_to_launch,
            sy, mean_fleet_distance
        ):
            logger.info(f"{sy.point} should spawn not launch small fleet. {best_route.plan} {num_ships_to_launch}")
            _spawn(agent, sy)
            continue
        if best_route.can_execute():
            logger.info(f"{sy.point} Mining Route: {best_route.plan}, {score:.2f}, {num_ships_to_launch} > {board_risk}")
            sy.action = Launch(num_ships_to_launch, best_route)
        else:
            logger.info(f"{sy.point} Waiting for Route: {best_route.plan}, {score:.2f}, {num_ships_to_launch} > {board_risk} in {best_route.time_to_mine}")



def should_not_launch_small_fleet(
    agent: Player, best_route: BoardRoute, can_deplete_kore_fast: bool,
    num_ships_to_launch: int, sy: Shipyard, mean_fleet_distance: int
):
    if len(best_route) < 4 and agent.kore >= 10:
        return True
    if agent.board.step < 50:
        return False

    if num_ships_to_launch < max(21, agent.ship_count // 20) or len(best_route) < mean_fleet_distance:
        if not can_deplete_kore_fast:
            return True
        if agent.available_kore() > 1000:
            return True

    return False


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

    routes = []
    route_set = set()
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
                wait_time = sy.calc_time_for_ships_for_action(best_plan.min_fleet_size())
                route = MiningRoute(departure, best_plan, wait_time)

                if len(route) > 2 * max_distance:
                    continue

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
                wait_time = sy.calc_time_for_ships_for_action(plan.min_fleet_size())
                route = MiningRoute(departure, plan, wait_time)

                if len(route) > 2 * max_distance:
                    continue

                if route.plan.to_str() in route_set:
                    continue

                if is_intercept_route(route, player, safety):
                    continue

                # if wait_time > 2:
                #     continue

                routes.append(route)
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

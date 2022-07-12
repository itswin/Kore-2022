import random
import itertools
import numpy as np
import os
from typing import List, Dict
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import PlanRoute, Point, ACTION_TO_ORTH_ACTIONS, PlanPath, ACTION_TO_OPPOSITE_ACTION, ALL_DIRECTIONS
    from board import Player, BoardRoute, Launch, Shipyard, MiningRoute, Board, AllowMine, HailMary, DirectAttack
    from helpers import is_intercept_route, find_closest_shipyards, _spawn
    from logger import logger
else:
    from .geometry import PlanRoute, Point, ACTION_TO_ORTH_ACTIONS, PlanPath, ACTION_TO_OPPOSITE_ACTION, ALL_DIRECTIONS
    from .board import Player, BoardRoute, Launch, Shipyard, MiningRoute, Board, AllowMine, HailMary, DirectAttack
    from .helpers import is_intercept_route, find_closest_shipyards, _spawn
    from .logger import logger

# <--->

SHOW_ROUTES = False
NUM_SHOW_ROUTES = 5

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

    if board.step < 50 and not op_ship_count:
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

    def get_best_plan_through_points(p1: Point, p2: Point, cache: Dict[Point, Dict[Point, PlanRoute]]={}):
        if p1 in cache:
            if p2 in cache[p1]:
                return cache[p1][p2]
        else:
            cache[p1] = {}

        plans = p1.dirs_to(p2)
        paths = [PlanRoute(plan) for plan in plans]
        best_route = max(
            (BoardRoute(p1, path) for path in paths),
            key=lambda x: x.expected_kore(board, 20)
        )

        cache[p1][p2] = best_route.plan
        return best_route.plan

    def score_route(route: BoardRoute, num_ships_to_launch: int, board_risk: int, free_ships: int) -> float:
        # Don't do short routes if we need to spawn
        if num_turns_to_deplete_kore > 1 and len(route) == 2:
            return 0

        exp_kore = route.expected_kore_mining(board, num_ships_to_launch)
        if my_ship_count < 50:
            if exp_kore < 10:
                return 0

        dist_penalty = sy.get_idle_turns_before(len(route)) / 20.0 if can_deplete_kore_fast else 0
        dist_bonus = 0
        board_risk_penalty = max((board_risk - route.plan.min_fleet_size()) / 10.0, 0)
        board_risk_penalty = 0
        percentage_bonus = num_ships_to_launch / free_ships
        return exp_kore / len(route) - dist_penalty + dist_bonus - board_risk_penalty + percentage_bonus
    
    def is_short_route(route):
        return len(route) < 6

    for sy in agent.shipyards:
        sy_max_dist = max_distance
        forced_destination = None
        max_time = max_distance * 2
        hail_mary = False
        if sy.action:
            if isinstance(sy.action, HailMary):
                hail_mary = True
            elif isinstance(sy.action, AllowMine):
                sy_max_dist = sy.action.max_distance
                forced_destination = sy.action.target
                max_time = sy.action.max_time
            elif not isinstance(sy.action, DirectAttack):
                continue

        free_ships = sy.available_ship_count
        num_short_routes = sum(1 for f in sy.incoming_allied_fleets if is_short_route(f.route))

        if free_ships <= 2:
            continue

        (closest_friendly_sy,
         closest_enemy_sy,
         min_friendly_distance,
         min_enemy_distance) = find_closest_shipyards(agent, sy.point, board.all_shipyards)

        routes = find_shipyard_mining_routes(
            sy, get_best_plan_through_points, safety=safety, max_distance=sy_max_dist, use_second_points=use_second_points,
            forced_destination=forced_destination, max_time=max_time
        )

        route_to_info = {}
        for route in routes:
            route_points = route.points()
            if len(route_points) > 6:
                route_points = route_points[:-3]
            elif len(route_points) > 2:
                route_points = route_points[:-2]

            if is_short_route(route) and num_short_routes >= 2:
                continue
            min_fleet_size = route.plan.min_fleet_size()
            board_risk = max(agent.estimate_board_risk(p, t + 1 + route.time_to_mine) for t, p in enumerate(route_points))
            optimistic_board_risk = max(agent.estimate_board_risk(p, t + 1 + route.time_to_mine, pessimistic=False) for t, p in enumerate(route_points))
            num_ships_to_launch = min_fleet_size \
                if can_deplete_kore_fast and not len(route) == 2 and not hail_mary \
                else free_ships
            # Clip number of ships to launch
            num_ships_to_launch = max(num_ships_to_launch, free_ships // 4)
            num_ships_to_launch = min(num_ships_to_launch, max(my_ship_count // 5, min_fleet_size))

            # Ignore everything if hail mary lol
            if not hail_mary:
                # Prevent sending huge long routes
                if min_fleet_size > 21 and \
                    (num_ships_to_launch > 0.2 * agent.ship_count or \
                    num_ships_to_launch > 0.5 * sy.estimate_shipyard_power(10)):
                    continue
                if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                    num_ships_to_launch = min(free_ships, board_risk + 1)
                    if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                        continue
                if optimistic_board_risk >= num_ships_to_launch:
                    num_ships_to_launch = min(free_ships, optimistic_board_risk + 1)
                    if optimistic_board_risk >= num_ships_to_launch:
                        continue
                if my_ship_count > 105 and num_ships_to_launch > my_ship_count // 5:
                    continue
                dec = 0 if len(route) < min_enemy_distance and route.end == sy.point else num_ships_to_launch
                if min_enemy_distance < 10 and sy.estimate_shipyard_power(min_enemy_distance) - dec < closest_enemy_sy.ship_count:
                    continue

            score = score_route(route, num_ships_to_launch, board_risk, free_ships)
            route_to_info[route] = (score, num_ships_to_launch, board_risk, optimistic_board_risk)

        if not route_to_info:
            logger.info(f"No mining routes for {sy.point}")
            continue

        items = sorted(route_to_info.items(), key=lambda x: x[1], reverse=True)
        if SHOW_ROUTES:
            for i in range(0, min(len(items), NUM_SHOW_ROUTES)):
                route = items[i][0]
                score, num_ships_to_launch, board_risk, optimistic_board_risk = route_to_info[route]
                logger.info(f"{sy.point} Mining Route: {route.plan}, {score}, {board_risk}")

        for i in range(min(1, len(items))):
            # best_route = max(route_to_info, key=lambda x: route_to_info[x][0])
            best_route = items[i][0]
            # for t, p in enumerate(best_route.points()):
            #     logger.info(f"{p} {t}, {agent.estimate_board_risk(p, t + 1 + best_route.time_to_mine)}")
            score, num_ships_to_launch, board_risk, optimistic_board_risk = route_to_info[best_route]
            if should_not_launch_small_fleet(
                agent, best_route, can_deplete_kore_fast, num_ships_to_launch,
                sy, mean_fleet_distance
            ):
                logger.info(f"{sy.point} should spawn not launch small fleet. {best_route.plan} {num_ships_to_launch}")
                _spawn(agent, sy)
                continue
            if best_route.can_execute():
                logger.info(f"{sy.point} Mining Route: {best_route.plan}, {score:.2f}, {num_ships_to_launch} > {board_risk}. {optimistic_board_risk}")
                if isinstance(sy.action, DirectAttack):
                    attack_score = sy.action.score
                    if score < attack_score:
                        logger.info(f"Mining route worse than current attack. {score} < {attack_score}")
                        continue
                    else:
                        logger.info(f"Overriding attack with mining route {best_route.plan}")
                sy.action = Launch(num_ships_to_launch, best_route)
                break
            else:
                logger.info(f"{sy.point} Waiting for Route: {best_route.plan}, {score:.2f}, {num_ships_to_launch} > {board_risk}. {optimistic_board_risk} in {best_route.time_to_mine}")
                break



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
            logger.info(f"Can not deplete kore fast")
            return True
        if agent.available_kore() > 1000:
            logger.info(f"Lots of kore")
            return True
        if sy.max_ships_to_spawn > agent.avg_shipyard_production_capacity and agent.kore >= sy.max_ships_to_spawn * 10 * 5:
            if len(sy.incoming_allied_fleets) <= 2 and sy.ship_count >= 21:
                return False
            logger.info(f"Better than average production")
            return True

    return False


def find_shipyard_mining_routes(
    sy: Shipyard, get_best_plan_through_points, safety=True, max_distance: int = 15, use_second_points: int = False,
    forced_destination: Point = None, max_time: int = 30
) -> List[BoardRoute]:
    if max_distance < 1:
        return []

    departure = sy.point
    player = sy.player
    board = player.board
    max_time = min(max_time, max_distance * 2)

    def force_destination_to(choices: List[Shipyard], closest_n: int):
        nonlocal forced_destination
        if forced_destination or not choices:
            return

        # Don't send to one of these if it is getting sieged or has enough ships
        choice_sy = min(choices, key=lambda x: x.distance_from(sy))
        incoming_hostile_power = sum(x.ship_count for x in choice_sy.incoming_hostile_fleets)
        incoming_allied_power = sum(x.ship_count for x in choice_sy.incoming_allied_fleets)
        future_power = choice_sy.ship_count + incoming_allied_power - incoming_hostile_power
        if future_power <= 0:
            logger.debug(f"Forced dest is sieged {choice_sy.point}")
            return

        avg_ships = player.ship_count / max(len(player.all_shipyards), 1)
        if future_power > avg_ships:
            logger.debug(f"Forced dest has enough ships {choice_sy.point}")
            return

        sorted_sys = sorted(sy.player.shipyards, key=lambda x: x.distance_from(choice_sy))
        num_closest_sys_to_help = min(closest_n, len(sorted_sys))
        for i in range(0, num_closest_sys_to_help):
            shipyard = sorted_sys[i]
            if shipyard.point == sy.point:
                forced_destination = choice_sy.point

    # recently_attacked_sys = sy.player.memory.recently_attacked_sys(sy.board.step)
    # force_destination_to(recently_attacked_sys, 2)

    young_sys = sy.player.future_shipyards + list(sy for sy in sy.player.shipyards if sy.turns_controlled < 5)
    force_destination_to(young_sys, 2)

    if forced_destination:
        logger.info(f"{sy.point} Forcing mining to {forced_destination}")

    def get_destinations(shipyards):
        destinations = set()
        for shipyard in shipyards:
            if forced_destination:
                if shipyard.point == forced_destination:
                    destinations.add(shipyard)
                continue
            siege = sum(x.ship_count for x in shipyard.incoming_hostile_fleets)
            help = sum(x.ship_count for x in shipyard.incoming_allied_fleets)
            if siege >= shipyard.ship_count + help:
                continue
            destinations.add(shipyard)
        return destinations

    destinations = get_destinations(sy.player.shipyards)
    future_destinations = get_destinations(sy.player.future_shipyards)

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
                css = [[c]] if adj == c else [[c, adj], [c, adj, c]]
                # css = [[c]] if adj == c else [[c, adj]]
                for cs in css:
                    dist_through_cs = sum(c.distance_from(d) for c, d in zip(cs, cs[1:])) + sy.distance_from(cs[0])
                    future_dest_sys = set(filter(
                        lambda x: (x.time_to_build <= dist_through_cs + x.distance_from(cs[-1])) and (x.point != c and x.point != adj),
                        future_destinations
                    ))
                    temp_dests = destinations | future_dest_sys
                    if not temp_dests:
                        continue
                    dest_sy = min(temp_dests, key=lambda x: cs[-1].distance_from(x.point))

                    destination = dest_sy.point
                    points = [departure] + cs + [destination]
                    best_plan = get_greedy_mining_plan_through(points, board, get_best_plan_through_points)
                    wait_time = sy.calc_time_for_ships_for_action(best_plan.min_fleet_size())
                    route = MiningRoute(departure, best_plan, wait_time)

                    if len(route) > max_time:
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

            future_dest_sys = set(filter(
                lambda x: x.time_to_build <= x.distance_from(c) + sy.distance_from(c),
                future_destinations
            ))
            temp_dests = destinations | future_dest_sys
            if not temp_dests:
                continue
            dest_sy = min(temp_dests, key=lambda x: c.distance_from(x.point))

            destination = dest_sy.point
            plans = departure.get_plans_through([c, destination])

            for plan in plans:
                wait_time = sy.calc_time_for_ships_for_action(plan.min_fleet_size())
                route = MiningRoute(departure, plan, wait_time)

                if len(route) > max_time:
                    continue

                if route.plan.to_str() in route_set:
                    continue

                if is_intercept_route(route, player, safety):
                    continue

                # if wait_time > 2:
                #     continue

                routes.append(route)
                route_set.add(route.plan.to_str())

    def is_yoyo(route):
        return len(route.plan.paths) == 2 and route.plan.paths[0].num_steps > 1 and route.start == route.end
    
    def is_flat_rectangle(route):
        return len(route.plan.paths) == 4 and \
            route.plan.paths[0].num_steps > 1 and \
            route.plan.paths[1].num_steps == 1 and \
            route.plan.paths[2].num_steps > 1 and route.start == route.end

    orig_len = len(routes)
    new_plans = []
    for i in range(orig_len):
        route = routes[i]
        if is_yoyo(route):
            first_action = route.first_action()
            last_action = route.last_action()
            for orth in ACTION_TO_ORTH_ACTIONS[first_action]:
                opp_orth = ACTION_TO_OPPOSITE_ACTION[orth]
                # E8W -> NE8W(X)SW
                for n in range(1, route.plan.paths[0].num_steps):
                    plan = PlanRoute([PlanPath(orth, 1), route.plan.paths[0], PlanPath(route.plan.paths[1].direction, n), PlanPath(opp_orth, 1), PlanPath(last_action, route.plan.paths[0].num_steps - n)])
                    new_plans.append(plan)
                # E8W -> E8WSNW
                plan = PlanRoute([route.plan.paths[0], PlanPath(route.plan.paths[1].direction, 1), PlanPath(orth, 1), PlanPath(opp_orth, 1), PlanPath(last_action, route.plan.paths[0].num_steps - 1)])
                new_plans.append(plan)
        elif is_flat_rectangle(route):
            # E4NW4S -> E4NW2SW
            for n in range(1, route.plan.paths[0].num_steps):
                opp_dir = ACTION_TO_OPPOSITE_ACTION[route.plan.paths[1].direction]
                plan = PlanRoute([route.plan.paths[0], route.plan.paths[1], PlanPath(route.plan.paths[2].direction, n), PlanPath(opp_dir, 1), PlanPath(route.plan.paths[2].direction, route.plan.paths[0].num_steps - n)])
                new_plans.append(plan)

            # E4NW4S -> NE4NW4S
            last_action = route.last_action()
            opp_orth = ACTION_TO_OPPOSITE_ACTION[last_action]
            plan = PlanRoute([PlanPath(opp_orth, 1)] + route.plan.paths[:-1] + [PlanPath(last_action, route.plan.paths[-1].num_steps + 1)])
            new_plans.append(plan)

            # E4NW4S -> E4NWESW
            opp_dir = ACTION_TO_OPPOSITE_ACTION[route.plan.paths[1].direction]
            plan = PlanRoute([route.plan.paths[0], route.plan.paths[1], PlanPath(route.plan.paths[2].direction, 1), PlanPath(route.plan.paths[0].direction, 1), PlanPath(opp_dir, 1), PlanPath(route.plan.paths[2].direction, route.plan.paths[0].num_steps)])
            new_plans.append(plan)
    
    # N2W20S
    for dir in ALL_DIRECTIONS:
        opp_dir = ACTION_TO_OPPOSITE_ACTION[dir]
        for orth_dir in ACTION_TO_ORTH_ACTIONS[dir]:
            for n in range(1,6):
                plan = PlanRoute([PlanPath(dir, n), PlanPath(orth_dir, 21), PlanPath(opp_dir, n)])
                new_plans.append(plan)

    # N4WEN
    for dir in ALL_DIRECTIONS:
        for orth_dir in ACTION_TO_ORTH_ACTIONS[dir]:
            opp_orth = ACTION_TO_OPPOSITE_ACTION[orth_dir]
            for n in range(1,10):
                plan = PlanRoute([PlanPath(dir, n), PlanPath(orth_dir, 1), PlanPath(opp_orth, 1), PlanPath(dir, n)])
                new_plans.append(plan)

    for plan in new_plans:
        wait_time = sy.calc_time_for_ships_for_action(plan.min_fleet_size())
        new_route = MiningRoute(departure, plan, wait_time)

        if new_route.plan.to_str() in route_set:
            continue

        if is_intercept_route(new_route, player, safety):
            continue

        if new_route.start != new_route.end:
            continue

        routes.append(new_route)
        route_set.add(new_route.plan.to_str())

    return routes


def get_greedy_mining_plan_through(points: List["Point"], board: Board, get_best_plan_through_points) -> PlanRoute:
    last = points[0]
    plan = PlanRoute([])
    for p in points[1:]:
        plan += get_best_plan_through_points(last, p)
        last = p

    return plan

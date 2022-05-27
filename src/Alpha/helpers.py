import os
import random
from typing import List, Tuple
from math import pi, exp

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import Point
    from board import Board, Player, BoardRoute, PlanRoute, Shipyard
    from logger import logger
else:
    from .geometry import Point
    from .board import Board, Player, BoardRoute, PlanRoute, Shipyard
    from .logger import logger

# <--->


def is_intercept_route(
    route: BoardRoute, player: Player, safety=True, allow_shipyard_intercept=False, allowed_join_point=None
):
    board = player.board

    if not allow_shipyard_intercept:
        shipyard_points = {x.point for x in board.shipyards}
    else:
        shipyard_points = {}

    for time, point in enumerate(route.points()[:-1]):
        if point in shipyard_points:
            return True

        for pl in board.players:
            is_enemy = pl != player

            if point in pl.expected_fleets_positions[time]:
                if allowed_join_point is None or pl.expected_fleets_positions[time][point].route.end != allowed_join_point:
                    return True

            if safety and is_enemy:
                if point in pl.expected_dmg_positions[time]:
                    return True

    return False


def find_shortcut_routes(
    board: Board,
    start: Point,
    end: Point,
    player: Player,
    num_ships: int,
    safety: bool = True,
    allow_shipyard_intercept=False,
    route_distance=None,
    allow_join=False,
) -> List[BoardRoute]:
    if route_distance is None:
        route_distance = start.distance_from(end)
    routes = []
    for p in board:
        distance = start.distance_from(p) + p.distance_from(end)
        if distance != route_distance:
            continue

        plans = start.get_plans_through([p, end])

        for plan in plans:
            if num_ships < plan.min_fleet_size():
                continue

            route = BoardRoute(start, plan)

            if is_intercept_route(
                route,
                player,
                safety=safety,
                allow_shipyard_intercept=allow_shipyard_intercept,
                allowed_join_point=end if allow_join else None,
            ):
                continue

            routes.append(route)

    return routes


def is_inevitable_victory(player: Player):
    if not player.opponents:
        return True

    board = player.board
    if board.steps_left > 100:
        return False

    board_kore = board.total_kore * (1 + board.regen_rate) ** board.steps_left

    player_kore = player.kore + player.fleet_expected_kore()
    opponent_kore = max(x.kore + x.fleet_expected_kore() for x in player.opponents)
    return player_kore > opponent_kore + board_kore


def find_closest_shipyards(player: Player, p: Point, shipyards=None) -> Tuple[Shipyard, Shipyard, int, int]:
    board = player.board
    closest_friendly_sy = None
    closest_enemy_sy = None
    min_friendly_distance = 100000
    min_enemy_distance = 100000

    if shipyards is None:
        shipyards = board.shipyards
    for shipyard in shipyards:
        distance = shipyard.distance_from(p)
        if shipyard.player_id != player.game_id:
            if distance < min_enemy_distance:
                closest_enemy_sy = shipyard
                min_enemy_distance = distance
        else:
            if distance < min_friendly_distance:
                closest_friendly_sy = shipyard
                min_friendly_distance = distance

    return closest_friendly_sy, closest_enemy_sy, min_friendly_distance, min_enemy_distance


def is_safety_route_to_convert(route_points: List[Point], player: Player):
    board = player.board

    target_point = route_points[-1]
    target_time = len(route_points)
    for pl in board.players:
        if pl != player:
            for t, positions in pl.expected_fleets_positions.items():
                if t >= target_time and target_point in positions:
                    return False

    shipyard_positions = {x.point for x in board.shipyards}

    for time, point in enumerate(route_points):
        for pl in board.players:
            if point in shipyard_positions:
                return False

            is_enemy = pl != player

            if point in pl.expected_fleets_positions[time]:
                return False

            if is_enemy:
                if point in pl.expected_dmg_positions[time]:
                    return False

    return True


def gaussian(x, mu, sigma):
    return 1 / (sigma * (2 * pi) ** 0.5) * exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

import random
import os
from typing import List
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import Convert
    from board import Player
    from logger import logger
    from helpers import find_closest_shipyards
    from state import Expansion
else:
    from .geometry import Convert
    from .board import Player
    from .logger import logger
    from .helpers import find_closest_shipyards
    from .state import Expansion

# <--->


def expand(player: Player, max_time_to_wait: int = 10):
    if player.update_state_if_is(Expansion):
        return
    board = player.board
    num_shipyards_to_create = need_more_shipyards(player)
    if not num_shipyards_to_create:
        return

    logger.info("---- Need to build shipyard ----")

    shipyard_to_point = find_best_position_for_shipyards(player)
    poses = sorted(shipyard_to_point.items(), key=lambda x: x[1]["score"], reverse=True)

    shipyard_count = 0
    shipyard_to_target = {}
    for shipyard, pose in poses:
        if shipyard_count >= num_shipyards_to_create:
            break

        incoming_hostile_fleets = shipyard.incoming_hostile_fleets
        if incoming_hostile_fleets:
            continue

        if shipyard.estimate_shipyard_power(max_time_to_wait) < board.shipyard_cost:
            continue
        
        target = pose["point"]
        shipyard_to_target[shipyard] = target
        shipyard_count += 1

    if shipyard_to_target:
        logger.info(f"Starting expansion: {shipyard_to_target}")
        player.state = Expansion(shipyard_to_target)
        player.update_state()


def find_best_position_for_shipyards(player: Player):
    board = player.board
    shipyards = board.shipyards

    shipyard_to_scores = defaultdict(list)
    for p in board:
        if p.kore > 100 or p.kore > board.total_kore * 0.01:
            continue

        (closest_friendly_sy,
         closest_enemy_sy,
         min_friendly_distance,
         min_enemy_distance) = find_closest_shipyards(player, p)

        closest_sy = closest_friendly_sy if min_friendly_distance < min_enemy_distance else closest_enemy_sy
        min_distance = min(min_friendly_distance, min_enemy_distance)

        if (
            not closest_sy
            or closest_sy.player_id != player.game_id
            or min_distance < 3
            or min_distance > 5
        ):
            continue

        nearby_kore = sum(x.kore / p.distance_from(x) for x in p.nearby_points(10))
        nearby_shipyards = sum(1 for x in board.shipyards if x.distance_from(p) < 5)
        shipyard_penalty = 100 * nearby_shipyards
        distance_penalty = 100 * min_distance
        enemy_penalty = 0 if min_enemy_distance >= 9 else \
            3 * closest_enemy_sy.estimate_shipyard_power(min_friendly_distance + 3) * (9 - min_enemy_distance)

        score = nearby_kore - shipyard_penalty - distance_penalty - enemy_penalty
        shipyard_to_scores[closest_sy].append({"score": score, "point": p})

    shipyard_to_point = {}
    for shipyard, scores in shipyard_to_scores.items():
        if scores:
            shipyard_to_point[shipyard] = max(scores, key=lambda x: x["score"])

    return shipyard_to_point


def need_more_shipyards(player: Player) -> int:
    board = player.board

    if player.ship_count < 100:
        return 0

    fleet_distance = []
    for sy in player.shipyards:
        for f in sy.incoming_allied_fleets:
            fleet_distance.append(len(f.route))

    if not fleet_distance:
        return 0

    mean_fleet_distance = sum(fleet_distance) / len(fleet_distance)

    shipyard_production_capacity = player.shipyard_production_capacity

    steps_left = board.steps_left
    if steps_left > 100:
        scale = 3
    elif steps_left > 50:
        scale = 4
    elif steps_left > 10:
        scale = 100
    else:
        scale = 1000

    logger.info(f"Need more shipyards {player.kore:.2f}, {scale * shipyard_production_capacity * mean_fleet_distance:.2f}, {shipyard_production_capacity:.2f}, {mean_fleet_distance:.2f}, {scale}")
    needed = player.kore > scale * shipyard_production_capacity * mean_fleet_distance
    if not needed:
        return 0

    current_shipyard_count = len(player.shipyards)

    op_shipyard_positions = {
        x.point for x in board.shipyards if x.player_id != player.game_id
    }
    expected_shipyard_count = current_shipyard_count + sum(
        1
        for x in player.fleets
        if x.route.last_action() == Convert or x.route.end in op_shipyard_positions
    )

    opponent_shipyard_count = max(len(x.shipyards) for x in player.opponents)
    opponent_ship_count = max(x.ship_count for x in player.opponents)
    if (
        expected_shipyard_count > opponent_shipyard_count
        and player.ship_count < opponent_ship_count
    ):
        return 0

    if current_shipyard_count < 10:
        if expected_shipyard_count > current_shipyard_count:
            return 0
        else:
            return 1

    return max(0, 5 - (expected_shipyard_count - current_shipyard_count))


import random
import math
import os
from typing import Set
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import Convert
    from board import Player, Shipyard
    from logger import logger
    from helpers import find_closest_shipyards, gaussian
    from state import Expansion
else:
    from .geometry import Convert
    from .board import Player, Shipyard
    from .logger import logger
    from .helpers import find_closest_shipyards, gaussian
    from .state import Expansion

# <--->
first_expansion = None
MIN_TIME_AFTER_FIRST_EXPANSION = 25

def expand(player: Player, step: int, self_built_sys: Set[Shipyard], max_time_to_wait: int = 10):
    global first_expansion
    if first_expansion is None and len(player.shipyards) == 2:
        first_expansion = step

    board = player.board
    num_shipyards_to_create = need_more_shipyards(player)
    if not num_shipyards_to_create:
        return
    if first_expansion is not None and step - first_expansion < MIN_TIME_AFTER_FIRST_EXPANSION:
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
        player.state = Expansion(shipyard_to_target, self_built_sys, step)
        player.update_state()


def find_best_position_for_shipyards(player: Player):
    board = player.board

    max_distance = 6
    my_shipyard_count = len(player.all_shipyards)
    op_shipard_count = max(len(x.all_shipyards) for x in player.opponents)
    if my_shipyard_count == 1 and op_shipard_count > 1:
        max_distance = 8

    shipyard_to_scores = defaultdict(list)
    for p in board:
        if p.kore > 100 or p.kore > board.total_kore * 0.01:
            continue
        # Dont form 3 shipyards in a line.
        # We can't send reinforcements in this case because
        # find_shortcut_routes chooses only routes along the line.
        if len(player.all_shipyards) == 2 and \
            (all(sy.point.x == p.x for sy in player.all_shipyards) or all(sy.point.y == p.y for sy in player.all_shipyards)):
            continue

        (closest_friendly_sy,
         closest_enemy_sy,
         min_friendly_distance,
         min_enemy_distance) = find_closest_shipyards(player, p, board.all_shipyards)

        closest_sy = closest_friendly_sy if min_friendly_distance < min_enemy_distance else closest_enemy_sy
        min_distance = min(min_friendly_distance, min_enemy_distance)
        dist_diff = min_enemy_distance - min_friendly_distance

        if (
            not closest_sy
            or closest_sy.player_id != player.game_id
            or min_distance < 4
            or min_distance > max_distance
        ):
            continue

        kore_mu = 0
        kore_sigma = 5
        gaussian_mid = gaussian(0, kore_mu, kore_sigma)
        nearby_kore = sum(
            (x.kore ** 1.1) * gaussian(p.distance_from(x), kore_mu, kore_sigma) / gaussian_mid
            for x in p.nearby_points(10)
        )
        nearby_shipyards = sum(1 for x in board.all_shipyards if x.distance_from(p) < 5)
        shipyard_penalty = 100 * nearby_shipyards
        distance_penalty = 100 * min_distance
        enemy_penalty = 0 if dist_diff >= 9 else \
            3 * player.estimate_board_risk(p, min_friendly_distance + 3 + min_enemy_distance // 2) * (9 - dist_diff)

        score = nearby_kore - shipyard_penalty - distance_penalty - enemy_penalty
        shipyard_to_scores[closest_sy].append({
            "score": score,
            "point": p,
            "shipyard_penalty": shipyard_penalty,
            "distance_penalty": distance_penalty,
            "enemy_penalty": enemy_penalty,
        })

    shipyard_to_point = {}
    for shipyard, scores in shipyard_to_scores.items():
        if scores:
            shipyard_to_point[shipyard] = max(scores, key=lambda x: x["score"])
            # scores.sort(key=lambda x: x["score"], reverse=True)
            # for i in range(0, min(3, len(scores))):
            #     pose = scores[i]
            #     logger.info(f"Expansion {shipyard.point}->{pose['point']} Score: {pose['score']:.2f} Shipyard penalty: {pose['shipyard_penalty']}, Distance penalty: {pose['distance_penalty']}, Enemy penalty: {pose['enemy_penalty']:.2f}")

    return shipyard_to_point


def need_more_shipyards(player: Player) -> int:
    board = player.board
    if len(player.future_shipyards) > 0:
        return False
    if isinstance(player.state, Expansion):
        return True

    if player.ship_count < 100:
        return 0

    fleet_distance = []
    for sy in player.all_shipyards:
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

    # my_ship_count = player.ship_count
    # op_ship_count = max(x.ship_count for x in player.opponents)
    # return len(player.all_shipyards) * 100 < player.ship_count and \
    #     (player.kore > shipyard_production_capacity * scale * 5 or my_ship_count > op_ship_count + 25)

    logger.info(f"Need more shipyards {player.kore:.2f}, {scale * shipyard_production_capacity * mean_fleet_distance:.2f}, {shipyard_production_capacity:.2f}, {mean_fleet_distance:.2f}, {scale}")
    needed = player.kore > scale * shipyard_production_capacity * mean_fleet_distance
    if not needed:
        return 0

    current_shipyard_count = len(player.shipyards)

    op_shipyard_positions = {
        x.point for x in board.all_shipyards if x.player_id != player.game_id
    }
    expected_shipyard_count = current_shipyard_count + sum(
        1
        for x in player.fleets
        if x.route.last_action() == Convert or x.route.end in op_shipyard_positions
    )

    opponent_shipyard_count = max(len(x.all_shipyards) for x in player.opponents)
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


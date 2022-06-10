import random
import math
import os
from typing import Dict, Set
from collections import defaultdict

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import Convert, Point
    from board import Player, Shipyard
    from logger import logger
    from helpers import find_closest_shipyards, gaussian, create_scorer
    from state import Expansion
else:
    from .geometry import Convert, Point
    from .board import Player, Shipyard
    from .logger import logger
    from .helpers import find_closest_shipyards, gaussian, create_scorer
    from .state import Expansion

# <--->
def expand(player: Player, step: int, self_built_sys: Set[Shipyard], max_time_to_wait: int = 10):
    board = player.board
    num_shipyards_to_create = need_more_shipyards(player)
    if not num_shipyards_to_create:
        return
    logger.info("---- Need to build shipyard ----")

    shipyard_to_point = find_best_position_for_shipyards(player)
    poses = sorted(shipyard_to_point.items(), key=lambda x: x[1]["score"], reverse=True)

    shipyard_count = 0
    shipyard_to_target = {}
    available_sys = set(sy for sy in player.shipyards if not sy.incoming_hostile_fleets)
    for _, pose in poses:
        if shipyard_count >= num_shipyards_to_create or not available_sys:
            break

        target = pose["point"]
        best_sy = find_best_shipyard(available_sys, target)
        if not best_sy:
            break

        if best_sy.estimate_shipyard_power(max_time_to_wait) < board.shipyard_cost:
            continue

        shipyard_to_target[best_sy] = target
        shipyard_count += 1

    if shipyard_to_target:
        logger.info(f"Starting expansion: {shipyard_to_target}")
        extra_distance = player.state.extra_distance if isinstance(player.state, Expansion) else 0
        player.state = Expansion(shipyard_to_target, self_built_sys, extra_distance)
        player.update_state()


def find_best_shipyard(available_sys: Set[Shipyard], p: Point) -> Shipyard:
    best_sy = None
    best_time = 999999
    for sy in available_sys:
        time = sy.distance_from(p) + sy.calc_time_for_ships_for_action(63)
        if time < best_time:
            best_time = time
            best_sy = sy
    available_sys.remove(best_sy)
    return best_sy


def find_best_position_for_shipyards(player: Player) -> Dict[Shipyard, Point]:
    board = player.board

    kore_sigma = 5
    g = create_scorer(kore_sigma)

    # Penalize kore based on how close it is to another shipyard
    point_to_kore = {}
    for p in board:
        (closest_friendly_sy,
         closest_enemy_sy,
         min_friendly_distance,
         min_enemy_distance) = find_closest_shipyards(player, p, board.all_shipyards)

        # closest_sy = closest_friendly_sy if min_friendly_distance < min_enemy_distance else closest_enemy_sy
        # min_distance = min(min_friendly_distance, min_enemy_distance)
        if closest_friendly_sy is None:
            point_to_kore[p] = p.kore
        else:
            point_to_kore[p] = p.kore * min((0.1 + 0.1 * min_friendly_distance, 1))

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
            or min_distance > 8
        ):
            continue

        nearby_kore = sum(
            (point_to_kore[x] ** 1.1) * g(p, x)
            for x in p.nearby_points(10)
        )
        nearby_shipyards = sum(1 for x in board.all_shipyards if x.distance_from(p) < 5)
        shipyard_penalty = 100 * nearby_shipyards
        distance_penalty = 50 * min_distance
        enemy_penalty = 0 if dist_diff >= 9 else \
            3 * player.estimate_board_risk(p, min_friendly_distance + 3 + min_enemy_distance // 2) * (9 - dist_diff)

        avg_dist_penalty = 10 * sum(x.distance_from(p) ** 1.5 for x in board.shipyards) / len(board.shipyards)
        # risk = player.estimate_board_risk(p, min_friendly_distance + min_enemy_distance + 3)
        # help = closest_sy.estimate_shipyard_power(dist_diff) - 50
        # enemy_penalty = max(3 * (risk - help // 2) * 16 / math.sqrt(dist_diff + 1), 0)
        # logger.error(f"{p}, {risk}, {help}, {enemy_penalty}")

        score = nearby_kore - shipyard_penalty - distance_penalty - enemy_penalty - avg_dist_penalty
        # score = nearby_kore - shipyard_penalty - distance_penalty - enemy_penalty
        shipyard_to_scores[closest_sy].append({
            "score": score,
            "point": p,
            "nearby_kore": nearby_kore,
            "shipyard_penalty": shipyard_penalty,
            "distance_penalty": distance_penalty,
            "enemy_penalty": enemy_penalty,
            "avg_dist_penalty": avg_dist_penalty,
        })

    max_kore_score = 0
    for shipyard, scores in shipyard_to_scores.items():
        max_kore_score = max(max_kore_score, max((x["nearby_kore"] for x in scores), default=0))

    BASELINE_KORE_SCORE = 5000
    for shipyard, scores in shipyard_to_scores.items():
        for score in scores:
            score["nearby_kore"] = score["nearby_kore"] * BASELINE_KORE_SCORE / max_kore_score
            score["score"] = score["nearby_kore"] - score["shipyard_penalty"] - score["distance_penalty"] - score["enemy_penalty"] - score["avg_dist_penalty"]

    shipyard_to_point = {}
    for shipyard, scores in shipyard_to_scores.items():
        if scores:
            shipyard_to_point[shipyard] = max(scores, key=lambda x: x["score"])
            # scores.sort(key=lambda x: x["score"], reverse=True)
            # for i in range(0, min(5, len(scores))):
            #     pose = scores[i]
            #     logger.info(f"Expansion {shipyard.point}->{pose['point']} Score: {pose['score']:.2f} Nearby kore: {pose['nearby_kore']:.2f} Shipyard: {pose['shipyard_penalty']}, Distance: {pose['distance_penalty']}, Enemy: {pose['enemy_penalty']:.2f}, Avg dist: {pose['avg_dist_penalty']:.2f}")

    return shipyard_to_point


def need_more_shipyards(player: Player) -> int:
    board = player.board
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


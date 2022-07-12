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
    from helpers import find_closest_shipyards, create_scorer
    from state import Expansion, PrepCoordinatedAttack, State
else:
    from .geometry import Convert, Point
    from .board import Player, Shipyard
    from .logger import logger
    from .helpers import find_closest_shipyards, create_scorer
    from .state import Expansion, PrepCoordinatedAttack, State

# <--->
SHOW_EXPANSIONS = False
NUM_SHOW_EXPANSIONS = 8

def expand(
    player: Player, step: int, self_built_sys: Set[Shipyard], 
    lost_sys: Set[Shipyard], max_time_to_wait: int = 10
):
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

    # lost_sy = {sy: sum(sy.distance_from(sy.point) for sy in player.shipyards) for sy in lost_sys}
    # lost_sy = sorted(lost_sy, key=lost_sy.get)
    # if lost_sy:
    #     target = lost_sy[0]
    #     logger.info(f"Starting expansion to lost shipyard {target}")
    #     extra_distance = player.state.extra_distance if isinstance(player.state, Expansion) else 0
    #     player.state = PrepCoordinatedAttack(10, target)
    #     player.update_state()
    #     return

    for _, pose in poses:
        if shipyard_count >= num_shipyards_to_create or not available_sys:
            break

        target = pose["point"]
        score = pose["score"]
        # if score < 2500:
            # player.state = State()
            # return
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

    kore_sigma = 4
    g = create_scorer(kore_sigma)

    def closer_bonus(point_to_closest_sy, x, p):
        f_dist = point_to_closest_sy[x][2]
        e_dist = point_to_closest_sy[x][3]
        new_dist = min(x.distance_from(p), f_dist)

        old_diff = max(e_dist - f_dist, 1)
        new_diff = max(e_dist - new_dist, 1)
        if new_diff <= old_diff:
            return 1
        div = new_diff / old_diff
        return 1 + 2 * div / kore_sigma

    # Penalize kore based on how close it is to another shipyard
    point_to_kore = {}
    for p in board:
        (closest_friendly_sy,
         closest_enemy_sy,
         min_friendly_distance,
         min_enemy_distance) = find_closest_shipyards(player, p, board.all_shipyards)

        closest_sy = closest_friendly_sy if min_friendly_distance < min_enemy_distance else closest_enemy_sy
        min_distance = min(min_friendly_distance, min_enemy_distance)
        if closest_friendly_sy is None:
            point_to_kore[p] = p.kore
        else:
            point_to_kore[p] = p.kore * min((0.1 + 0.2 * min_friendly_distance, 1))

    point_to_closest_sy = {}
    for p in board:
        point_to_closest_sy[p] = find_closest_shipyards(player, p, board.all_shipyards)

    # op_shipyard_positions = {
    #     x.point for x in board.all_shipyards if x.player_id != player.game_id
    # }
    # attacking_count = sum(
    #     x.ship_count for x in player.fleets if x.route.end in op_shipyard_positions
    # )
    # my_ship_count = player.ship_count - attacking_count
    # op_ship_count = max(x.ship_count for x in player.opponents)
    # my_sy_count = len(player.all_shipyards)
    # op_sy_count = max(len(x.all_shipyards) for x in player.opponents)
    # max_exp = 8 if op_sy_count > my_sy_count and my_ship_count >= op_ship_count else 6
    max_exp = 6
    # logger.info(f"Max exp {max_exp}")

    num_sys = len(player.all_shipyards)
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
         min_enemy_distance) = point_to_closest_sy[p]

        closest_sy = closest_friendly_sy if min_friendly_distance < min_enemy_distance else closest_enemy_sy
        min_distance = min(min_friendly_distance, min_enemy_distance)
        dist_diff = min_enemy_distance - min_friendly_distance

        if (
            not closest_sy
            or closest_sy.player_id != player.game_id
            or min_distance < 4
            or min_distance > max_exp
        ):
            continue

        nearby_kore = sum(
            (point_to_kore[x] ** 1.1) * g(p, x) * closer_bonus(point_to_closest_sy, x, p)
            for x in p.nearby_points(10)
        )
        nearby_shipyards = sum(1 for x in board.all_shipyards if x.distance_from(p) < 5)
        shipyard_penalty = 100 * nearby_shipyards
        distance_penalty = 50 * min_distance
        enemy_penalty = 0 if dist_diff >= 9 else \
            3 * player.estimate_board_risk(p, min_friendly_distance + 3 + min_enemy_distance // 2) * (9 - dist_diff)

        avg_dist_penalty = 10 * sum(x.distance_from(p) ** 1.5 for x in player.all_shipyards) / num_sys if num_sys else 0
        risk = player.estimate_board_risk(p, min_friendly_distance + min_enemy_distance + 3)
        help = player.opponents[0].estimate_board_risk(p, min_friendly_distance + min_enemy_distance // 2) - 50
        # enemy_penalty = max(3 * (risk - help // 2) * 16 / math.sqrt(dist_diff + 1), 0)
        if risk > help * 1.5:
            enemy_penalty += max(1000, enemy_penalty)
        # logger.error(f"{p}, {risk}, {help}, {enemy_penalty}")

        score = nearby_kore - shipyard_penalty - distance_penalty - enemy_penalty - avg_dist_penalty
        shipyard_to_scores[closest_sy].append({
            "score": score,
            "point": p,
            "nearby_kore": nearby_kore,
            "shipyard_penalty": shipyard_penalty,
            "distance_penalty": distance_penalty,
            "enemy_penalty": enemy_penalty,
            "avg_dist_penalty": avg_dist_penalty,
        })

    max_kore_score = 1
    max_enemy_penalty = 1
    for shipyard, scores in shipyard_to_scores.items():
        max_kore_score = max(max_kore_score, max((x["nearby_kore"] for x in scores), default=0))
        max_enemy_penalty = max(max_enemy_penalty, max((x["enemy_penalty"] for x in scores), default=0))

    BASELINE_KORE_SCORE = 5000
    # BASELINE_ENEMY_PENALTY = 2500
    for shipyard, scores in shipyard_to_scores.items():
        for score in scores:
            score["nearby_kore"] = score["nearby_kore"] * BASELINE_KORE_SCORE / max_kore_score
            # score["enemy_penalty"] = score["enemy_penalty"] * BASELINE_ENEMY_PENALTY / max_enemy_penalty
            score["score"] = score["nearby_kore"] - score["shipyard_penalty"] - score["distance_penalty"] - score["enemy_penalty"] - score["avg_dist_penalty"]

    shipyard_to_point = {}
    for shipyard, scores in shipyard_to_scores.items():
        if scores:
            shipyard_to_point[shipyard] = max(scores, key=lambda x: x["score"])
            if SHOW_EXPANSIONS:
                scores.sort(key=lambda x: x["score"], reverse=True)
                for i in range(0, min(NUM_SHOW_EXPANSIONS, len(scores))):
                    pose = scores[i]
                    logger.info(f"Expansion {shipyard.point}->{pose['point']} Score: {pose['score']:.2f} Nearby kore: {pose['nearby_kore']:.2f} Shipyard: {pose['shipyard_penalty']}, Distance: {pose['distance_penalty']}, Enemy: {pose['enemy_penalty']:.2f}, Avg dist: {pose['avg_dist_penalty']:.2f}")

    return shipyard_to_point


def need_more_shipyards(player: Player) -> int:
    board = player.board

    op_shipyard_positions = {
        x.point for x in board.all_shipyards if x.player_id != player.game_id
    }
    attacking_count = sum(
        x.ship_count for x in player.fleets if x.route.end in op_shipyard_positions
    )
    avail_sy_count = player.ship_count - attacking_count
    if avail_sy_count < 100:
        return 0

    my_sy_count = len(player.all_shipyards)
    op_sy_count = max(len(x.all_shipyards) for x in player.opponents)

    my_ship_count = avail_sy_count
    op_ship_count = max(x.ship_count for x in player.opponents)

    op_stockpile = sum(x.ship_count for x in player.opponents[0].shipyards)
    if op_stockpile > op_ship_count * 0.5:
        logger.info(f"Enemy stockpiling. Do not expand")
        if isinstance(player.state, Expansion):
            logger.info(f"Exiting expansion")
            player.state = State()
        return 0

    if my_sy_count * 75 > my_ship_count:
        return 0

    if isinstance(player.state, Expansion):
        return True
    if player.state.__repr__() != "State":
        return False

    fleet_distance = []
    for sy in player.all_shipyards:
        for f in sy.incoming_allied_fleets:
            fleet_distance.append(len(f.route))

    if not fleet_distance:
        return 0

    mean_fleet_distance = sum(fleet_distance) / len(fleet_distance)

    shipyard_production_capacity = player.adj_shipyard_production_capacity

    steps_left = board.steps_left
    if steps_left > 100:
        scale = 3
    elif steps_left > 50:
        scale = 4
    elif steps_left > 10:
        scale = 100
    else:
        scale = 1000

    # if my_sy_count != 1 and my_sy_count >= op_sy_count and my_ship_count < op_ship_count - 25:
    #     return 0

    # needed = player.kore > scale * shipyard_production_capacity * mean_fleet_distance
    # if needed:
    #     return 1

    # if my_sy_count < op_sy_count and op_sy_count <= 5:
    #     return 1

    # mean_fleet_distance = max(mean_fleet_distance, 8)
    logger.info(f"Need more shipyards {player.kore:.2f}, {scale * shipyard_production_capacity * mean_fleet_distance:.2f}, {shipyard_production_capacity:.2f}, {mean_fleet_distance:.2f}, {scale}")
    needed = player.kore > scale * shipyard_production_capacity * mean_fleet_distance
    # ship_count_needed = my_ship_count > my_sy_count * 150
    # kore_needed = 5 * shipyard_production_capacity * 10
    # logger.info(f"Needed: {ship_count_needed}, {150 * shipyard_production_capacity}")
    # needed = player.kore > kore_needed
    # needed = needed or my_sy_count * 150 > my_ship_count

    if my_sy_count == 1 and my_ship_count >= 150:
        needed = True
        logger.info(f"Lots of ships for first expansion")

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


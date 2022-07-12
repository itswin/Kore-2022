import os

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from geometry import PlanRoute
    from board import Player, Launch, Spawn, Fleet, FleetPointer, BoardRoute, DontLaunch, Shipyard
    from helpers import is_inevitable_victory, find_shortcut_routes, find_closest_shipyards, _spawn
    from logger import logger
else:
    from .geometry import PlanRoute
    from .board import Player, Launch, Spawn, Fleet, FleetPointer, BoardRoute, DontLaunch, Shipyard
    from .helpers import is_inevitable_victory, find_shortcut_routes, find_closest_shipyards, _spawn
    from .logger import logger

# <--->


def direct_attack(agent: Player, max_distance: int = 10, max_time_to_wait: int = 5):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    targets = []
    for x in agent.opponents:
        for sy in x.all_shipyards:
            for fleet in sy.incoming_allied_fleets:
                if fleet.expected_value() > 0.5:
                    targets.append(fleet)

    if not targets:
        return

    targets.sort(key=lambda x: x.expected_value(), reverse=True)

    shipyards = [
        x for x in agent.shipyards if x.available_ship_count > 0 and not x.action
    ]
    if not shipyards:
        return

    point_to_closest_shipyard = {}
    for p in board:
        (closest_friendly_sy, _, _, _) = find_closest_shipyards(agent, p)
        point_to_closest_shipyard[p] = closest_friendly_sy.point

    opponent_shipyard_points = {x.point for x in board.all_shipyards if x.player_id != agent.game_id}
    adjacent_attacks = []
    for t in targets:
        min_ships_to_send = int(t.ship_count + 1)
        attacked = False
        adjacent_action = None
        adjacent_sy = None
        adjacent_target_point = None
        best_candidate_sy = None
        best_candidate_time = max_time_to_wait
        best_target_point = None

        shipyards.sort(key=lambda x: x.distance_from(t))
        for sy in shipyards:
            if sy.action or sy.estimate_shipyard_power(max_time_to_wait) < min_ships_to_send:
                continue

            num_ships_to_launch = sy.available_ship_count

            for target_time, target_point in enumerate(t.route, 1):
                if target_point in opponent_shipyard_points:
                    continue
                if target_time > max_distance:
                    continue

                time_diff = target_time - sy.point.distance_from(target_point)
                is_adjacent_attack = time_diff == 1
                if time_diff != 0 and time_diff != 1:
                    if time_diff > 0 and time_diff < max_time_to_wait and \
                        time_diff < best_candidate_time and \
                        sy.estimate_shipyard_power(time_diff) >= min_ships_to_send:
                        best_candidate_sy = sy
                        best_candidate_time = time_diff
                        best_target_point = target_point
                    continue

                if is_adjacent_attack:
                    for p in target_point.adjacent_points:
                        time = sy.distance_from(p)
                        if time == target_time:
                            target_point = p
                            break

                if sy.available_ship_count < min_ships_to_send:
                    continue

                destination = point_to_closest_shipyard[target_point]
                plans = sy.point.get_plans_through([target_point, destination])
                routes = [BoardRoute(sy.point, plan) for plan in plans]
                routes.sort(key=lambda route: route.expected_kore(board, num_ships_to_launch))
                for route in routes:
                    route_points = route.points()
                    route_points = route_points[:-2] if len(route_points) > 2 else route_points
                    if num_ships_to_launch < route.plan.min_fleet_size():
                        continue

                    if any(x in opponent_shipyard_points for x in route_points):
                        continue

                    board_risk = max(
                        agent.estimate_board_risk(p, time + 1) + 
                        (t.ship_count if (time + 1) >= target_time else 0)
                        for time, p in enumerate(route_points)
                    )

                    num_ships_to_launch = min(board_risk + 1, sy.available_ship_count)
                    if not agent.is_board_risk_worth(board_risk, num_ships_to_launch, sy):
                        continue

                    if is_intercept_direct_attack_route(route, agent, direct_attack_fleet=t):
                        continue

                    if is_adjacent_attack:
                        adjacent_sy = sy
                        adjacent_action = Launch(num_ships_to_launch, route)
                        adjacent_target_point = target_point
                    else:
                        logger.info(
                            f"Direct attack {sy.point}->{target_point}, distance={target_time}"
                        )
                        sy.action = Launch(num_ships_to_launch, route)
                        attacked = True
                        break

                if attacked:
                    break

            if attacked:
                break
        if not attacked:
            if best_candidate_sy is not None:
                _spawn(agent, best_candidate_sy)
                logger.info(f"Saving for direct attack {t.point}, {best_candidate_sy.point}->{best_target_point}, time={best_candidate_time}")
            elif adjacent_action is not None:
                adjacent_attacks.append((adjacent_sy, adjacent_action, adjacent_target_point))

    for sy, action, target_point in adjacent_attacks:
        if sy.action is None:
            logger.info(
                f"Adjacent direct attack {sy.point}->{target_point}, distance={sy.distance_from(target_point)}"
            )
            sy.action = action

def is_intercept_direct_attack_route(
    route: BoardRoute, player: Player, direct_attack_fleet: Fleet
):
    board = player.board

    fleets = [FleetPointer(f) for f in board.fleets if f != direct_attack_fleet]

    for point in route.points()[:-1]:
        for fleet in fleets:
            fleet.update()

            if fleet.point is None:
                continue

            if fleet.point == point:
                return True

            if fleet.obj.player_id != player.game_id:
                for p in fleet.point.adjacent_points:
                    if p == point:
                        return True

    return False


def adjacent_attack(agent: Player, max_distance: int = 10):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    targets = _find_adjacent_targets(agent, max_distance)
    if not targets:
        return

    shipyards = [
        x for x in agent.shipyards if x.available_ship_count > 0 and not x.action
    ]
    if not shipyards:
        return

    fleets_to_be_attacked = set()
    for t in sorted(targets, key=lambda x: (-len(x["fleets"]), x["time"])):
        target_point = t["point"]
        target_time = t["time"]
        target_fleets = t["fleets"]
        if any(x in fleets_to_be_attacked for x in target_fleets):
            continue

        for sy in shipyards:
            if sy.action:
                continue

            distance = sy.distance_from(target_point)
            if distance > target_time:
                continue
            min_ship_count = min(x.ship_count for x in target_fleets)
            num_ships_to_send = min(sy.available_ship_count, min_ship_count)

            routes = find_shortcut_routes(
                board,
                sy.point,
                target_point,
                agent,
                num_ships_to_send,
                route_distance=target_time,
            )
            if not routes:
                continue

            logger.info(
                f"Adjacent attack {sy.point}->{target_point}, distance={distance}, target_time={target_time}"
            )
            best_route = max(routes, key=lambda route: route.expected_kore(board, num_ships_to_send))
            sy.action = Launch(num_ships_to_send, best_route)

            for fleet in target_fleets:
                fleets_to_be_attacked.add(fleet)
            break


def _find_adjacent_targets(agent: Player, max_distance: int = 5):
    board = agent.board
    shipyards_points = {x.point for x in board.shipyards}
    fleets = [FleetPointer(f) for f in board.fleets]
    if len(fleets) < 2:
        return []

    time = 0
    targets = []
    while any(x.is_active for x in fleets) and time <= max_distance:
        time += 1

        for f in fleets:
            f.update()
            if f.build_shipyard:
                shipyards_points.add(f.build_shipyard)

        point_to_fleet = {
            x.point: x.obj
            for x in fleets
            if x.is_active and x.point not in shipyards_points
        }

        for point in board:
            if point in point_to_fleet or point in shipyards_points:
                continue

            adjacent_fleets = [
                point_to_fleet[x] for x in point.adjacent_points if x in point_to_fleet
            ]
            if len(adjacent_fleets) < 2:
                continue

            if any(x.player_id == agent.game_id for x in adjacent_fleets):
                continue

            targets.append({"point": point, "time": time, "fleets": adjacent_fleets})

    return targets


def _need_more_ships(agent: Player, ship_count: int):
    board = agent.board
    if board.steps_left < 10:
        return False
    if ship_count > _max_ships_to_control(agent):
        return False
    if board.steps_left < 50 and is_inevitable_victory(agent):
        return False
    if board.steps_left < 100 and agent.ship_count > 1.5 * sum(x.ship_count for x in agent.opponents):
        return False
    return True


def _max_ships_to_control(agent: Player):
    return max(100, 3 * sum(x.ship_count for x in agent.opponents))


def should_greedy_spawn(agent: Player, kore_ship_mult: float = 1.2):
    if agent.kore < 300:
        return False
    board = agent.board
    op_ship_count = max(x.ship_count for x in agent.opponents)
    op_kore = max(x.kore for x in agent.opponents)
    kore_surplus = agent.available_kore() - op_kore
    ship_surplus_potential = kore_surplus // board.spawn_cost
    ship_surplus = agent.ship_count - op_ship_count * kore_ship_mult
    if ship_surplus < 0 and ship_surplus_potential > -ship_surplus:
        logger.info(f"More kore but behind in ships, greedy spawn")
        return True

    return False


def greedy_spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)
    can_greedy_spawn = should_greedy_spawn(agent)
    for shipyard in agent.shipyards:
        if shipyard.action and not isinstance(shipyard.action, DontLaunch):
            continue

        if len(shipyard.incoming_allied_fleets) <= 1 and shipyard.ship_count >= 21:
            continue

        if not can_greedy_spawn and \
            shipyard.ship_count > agent.ship_count * 0.2 / len(agent.all_shipyards):
            continue

        num_ships_to_spawn = _spawn(agent, shipyard, False)
        if not num_ships_to_spawn:
            continue

        ship_count += num_ships_to_spawn
        logger.info(f"Greedy spawn {shipyard.point} {num_ships_to_spawn}")
        if ship_count > max_ship_count:
            return


def spawn(agent: Player):
    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)
    for shipyard in agent.shipyards:
        if shipyard.action and not isinstance(shipyard.action, DontLaunch):
            continue

        ship_count += _spawn(agent, shipyard)
        if ship_count > max_ship_count:
            return


def conservative_save_kore(agent: Player):
    if agent.ship_count > 1.1 * sum(x.ship_count for x in agent.opponents):
        save_kore(agent)
    if agent.board.steps_left < 10:
        save_kore(agent)


def save_kore(agent: Player):
    board = agent.board

    if board.steps_left < 25:
        agent.inc_kore_reserve(min(agent.available_kore(), 1.25 * sum(x.kore for x in agent.opponents)))
        logger.info(f"Saved kore: {agent.kore_reserve}")

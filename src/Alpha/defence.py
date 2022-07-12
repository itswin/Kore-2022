from collections import defaultdict
import os
from typing import Set

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from basic import max_ships_to_spawn
    from board import Player, Launch, Shipyard, Spawn, AllowMine, HailMary
    from helpers import find_shortcut_routes, _spawn
    from logger import logger
    from state import Expansion, State
else:
    from .basic import max_ships_to_spawn
    from .board import Player, Launch, Shipyard, Spawn, AllowMine, HailMary
    from .helpers import find_shortcut_routes, _spawn
    from .logger import logger
    from .state import Expansion, State

# <--->


def defend_shipyards(agent: Player, self_built_sys: Set[Shipyard]):
    board = agent.board

    need_help_shipyards = []
    for sy in agent.shipyards:
        if sy.action:
            continue

        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_allied_fleets = sy.incoming_allied_fleets

        if not incoming_hostile_fleets:
            continue

        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)

        shipyard_reinforcements = defaultdict(int)
        for f in sy.incoming_allied_fleets:
            shipyard_reinforcements[f.eta] += f.ship_count
        for f in sy.incoming_hostile_fleets:
            shipyard_reinforcements[f.eta] -= f.ship_count

        ship_count = sy.ship_count
        reinforcement_diff = 0
        ship_deficit = 9e9
        for t in range(0, board.size + 1):
            reinforcement_diff += shipyard_reinforcements[t]
            ship_deficit = min(ship_deficit, reinforcement_diff)

        if ship_deficit >= 0:
            sy.set_guard_ship_count(min(sy.ship_count, int(ship_deficit * 1.1)))
            logger.info(f"{sy.point} is under attack, but has enough ships")
            continue

        # spawn as much as possible
        num_ships_to_spawn = _spawn(agent, sy)
        logger.info(f"Spawned {num_ships_to_spawn} ships to protect shipyard {sy.point}")

        # Hail Mary if about to die and no incoming fleets
        immediate_hostile_power = sum(x.ship_count for x in incoming_hostile_fleets if x.eta == 1)
        if incoming_hostile_time == 1 and not incoming_allied_fleets and immediate_hostile_power > sy.ship_count + num_ships_to_spawn:
            logger.info(f"{sy.point} is getting overtaken. Sending hail mary mining fleet")
            sy.action = HailMary()
            continue

        if not isinstance(sy.action, Spawn):
            sy.action = AllowMine(incoming_hostile_time // 2, sy.point)

        need_help_shipyards.append((sy, -ship_deficit))

    for sy in agent.future_shipyards:
        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_allied_fleets = sy.incoming_allied_fleets

        if not incoming_hostile_fleets:
            continue

        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)

        shipyard_reinforcements = defaultdict(int)
        for f in sy.incoming_allied_fleets:
            shipyard_reinforcements[f.eta] += f.ship_count
        for f in sy.incoming_hostile_fleets:
            shipyard_reinforcements[f.eta] -= f.ship_count

        ship_count = sy.ship_count
        reinforcement_diff = 0
        ship_deficit = 9e9
        for t in range(0, board.size + 1):
            reinforcement_diff += shipyard_reinforcements[t]
            ship_deficit = min(ship_deficit, reinforcement_diff)

        if ship_deficit < 0:
            need_help_shipyards.append((sy, -ship_deficit))

    for help_sy, ships_needed in need_help_shipyards:
        incoming_hostile_fleets = help_sy.incoming_hostile_fleets
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)

        shipyards = sorted(agent.shipyards, key=lambda x: x.distance_from(help_sy.point))
        # help_count = 0
        # for sy in shipyards:
        #     if sy == help_sy or sy.action or not sy.available_ship_count:
        #         continue
        #     distance = sy.distance_from(help_sy)
        #     if distance < incoming_hostile_time - 1:
        #         help_count += sy.available_ship_count
        # should_help_now = help_count >= ships_needed
        # help_received = 0
        is_done_helping = False

        for sy in shipyards:
            if sy == help_sy or sy.action or not sy.available_ship_count or is_done_helping:
                continue

            distance = sy.distance_from(help_sy)
            # if should_help_now:
            #     routes = find_shortcut_routes(
            #         board, sy.point, help_sy.point, agent, sy.ship_count
            #     )
            #     if not routes:
            #         logger.error(f"No routes to send reinforcements {sy.point}->{help_sy.point}")
            #         _spawn(agent, sy)
            #         continue

            #     num_ships_to_launch = sy.available_ship_count
            #     best_route = max(routes, key=lambda route: route.expected_kore(board, num_ships_to_launch))
            #     logger.info(f"Send reinforcements {sy.point}->{help_sy.point}. Size: {num_ships_to_launch}")
            #     sy.action = Launch(num_ships_to_launch, best_route)
            #     help_received += sy.available_ship_count
            #     is_done_helping = help_received >= ships_needed
            if distance < incoming_hostile_time - 1:
                num_ships_to_spawn = _spawn(agent, sy)
                logger.info(f"Saving reinforcements for {sy.point}->{help_sy.point}. Spawned {num_ships_to_spawn} ships")
                if not isinstance(sy.action, Spawn):
                    sy.action = AllowMine(incoming_hostile_time // 2, help_sy.point, incoming_hostile_time)
            elif distance == incoming_hostile_time - 1 or \
                (len(agent.all_shipyards) < 5 and help_sy.point in self_built_sys):
                if len(agent.all_shipyards) < 5:
                    logger.info(f"Not many shipyards. Save shipyard at all costs")
                routes = find_shortcut_routes(
                    board, sy.point, help_sy.point, agent, sy.ship_count, allow_join=True
                )

                if not routes:
                    logger.error(f"No routes to send reinforcements {sy.point}->{help_sy.point}")
                    _spawn(agent, sy)
                    continue

                num_ships_to_launch = sy.available_ship_count
                if distance == incoming_hostile_time - 1:
                    best_route = max(routes, key=lambda route: route.expected_kore(board, num_ships_to_launch))
                else:
                    best_route = min(routes, key=lambda route: route.expected_kore(board, num_ships_to_launch))
                logger.info(f"Send reinforcements {sy.point}->{help_sy.point}. Size: {num_ships_to_launch}")
                sy.action = Launch(num_ships_to_launch, best_route)
            else:
                logger.info(f"Not in time to save shipyard {sy.point}->{help_sy.point}")

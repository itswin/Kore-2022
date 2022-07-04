import random
import os
from typing import Set

IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from board import Player, Launch, Shipyard, Spawn, AllowMine
    from helpers import find_shortcut_routes, _spawn
    from logger import logger
    from state import Expansion, State
else:
    from .board import Player, Launch, Shipyard, Spawn, AllowMine
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

        incoming_hostile_power = sum(x.ship_count for x in incoming_hostile_fleets)
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)
        incoming_allied_power = sum(
            x.ship_count
            for x in incoming_allied_fleets
            if x.eta < incoming_hostile_time
        )

        ships_needed = incoming_hostile_power - incoming_allied_power
        if sy.ship_count > ships_needed:
            if ships_needed > 0:
                sy.set_guard_ship_count(min(sy.ship_count, int(ships_needed * 1.1)))
            logger.info(f"{sy.point} is under attack, but has enough ships")
            continue

        # spawn as much as possible
        num_ships_to_spawn = _spawn(agent, sy)
        logger.info(f"Spawned {num_ships_to_spawn} ships to protect shipyard {sy.point}")

        if not isinstance(sy.action, Spawn):
            sy.action = AllowMine(incoming_hostile_time // 2)

        need_help_shipyards.append(sy)

    for sy in agent.future_shipyards:
        incoming_hostile_fleets = sy.incoming_hostile_fleets
        incoming_allied_fleets = sy.incoming_allied_fleets

        if not incoming_hostile_fleets:
            continue

        incoming_hostile_power = sum(x.ship_count for x in incoming_hostile_fleets)
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)
        incoming_allied_power = sum(
            x.ship_count
            for x in incoming_allied_fleets
            if x.eta < incoming_hostile_time
        )

        ships_needed = incoming_hostile_power - incoming_allied_power
        if ships_needed > 0:
            need_help_shipyards.append(sy)

    for help_sy in need_help_shipyards:
        incoming_hostile_fleets = help_sy.incoming_hostile_fleets
        incoming_hostile_time = min(x.eta for x in incoming_hostile_fleets)

        for sy in agent.shipyards:
            if sy == help_sy or sy.action or not sy.available_ship_count:
                continue

            distance = sy.distance_from(help_sy)
            if distance < incoming_hostile_time - 1:
                num_ships_to_spawn = _spawn(agent, sy)
                logger.info(f"Saving reinforcements for {sy.point}->{help_sy.point}. Spawned {num_ships_to_spawn} ships")
                if not isinstance(sy.action, Spawn):
                    sy.action = AllowMine(incoming_hostile_time // 2)
            elif distance == incoming_hostile_time - 1 or \
                (len(agent.all_shipyards) < 5 and help_sy.point in self_built_sys):
                if len(agent.all_shipyards) < 5:
                    logger.info(f"Not many shipyards. Save shipyard at all costs")
                routes = find_shortcut_routes(
                    board, sy.point, help_sy.point, agent, sy.ship_count
                )
                if routes:
                    logger.info(f"Send reinforcements {sy.point}->{help_sy.point}. Size: {sy.available_ship_count}")
                    sy.action = Launch(
                        sy.available_ship_count, random.choice(routes)
                    )
                else:
                    logger.error(f"No routes to send reinforcements {sy.point}->{help_sy.point}")
                    _spawn(agent, sy)
            else:
                logger.info(f"Not in time to save shipyard {sy.point}->{help_sy.point}")

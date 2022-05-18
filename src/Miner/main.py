
from random import random, sample, randint

from kaggle_environments import utils
from kaggle_environments.helpers import Point, Direction
from kaggle_environments.envs.kore_fleets.helpers import Board, ShipyardAction

def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player
    remaining_kore = me.kore
    shipyards = me.shipyards
    convert_cost = board.configuration.convert_cost
    spawn_cost = board.configuration.spawn_cost
    # randomize shipyard order
    shipyards = sample(shipyards, len(shipyards))
    for shipyard in shipyards:
        # if we have over 1k kore and our max spawn is > 5 (we've held this shipyard for a while)
        # create a fleet to build a new shipyard!
        if remaining_kore > 1000 and shipyard.max_spawn > 5:
            if shipyard.ship_count >= convert_cost + 10:
                gap1 = str(randint(3, 9))
                gap2 = str(randint(3, 9))
                start_dir = randint(0, 3)
                flight_plan = Direction.list_directions()[start_dir].to_char() + gap1
                next_dir = (start_dir + 1) % 4
                flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
                next_dir = (next_dir + 1) % 4
                flight_plan += "C"
                shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(max(convert_cost + 10, int(shipyard.ship_count/2)), flight_plan)
            elif remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))

        # launch a large fleet if able
        elif shipyard.ship_count >= 21:
            gap1 = str(randint(3, 9))
            gap2 = str(randint(3, 9))
            start_dir = randint(0, 3)
            flight_plan = Direction.list_directions()[start_dir].to_char() + gap1
            next_dir = (start_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char() + gap2
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char() + gap1
            next_dir = (next_dir + 1) % 4
            flight_plan += Direction.list_directions()[next_dir].to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(21, flight_plan)
    
        # else spawn if possible
        elif remaining_kore > board.configuration.spawn_cost * shipyard.max_spawn:
            remaining_kore -= board.configuration.spawn_cost
            if remaining_kore >= spawn_cost:
                shipyard.next_action = ShipyardAction.spawn_ships(min(shipyard.max_spawn, int(remaining_kore/spawn_cost)))
        # else launch a small fleet
        elif shipyard.ship_count >= 2:
            dir_str = Direction.random_direction().to_char()
            shipyard.next_action = ShipyardAction.launch_fleet_with_flight_plan(2, dir_str)
            
    return me.next_actions

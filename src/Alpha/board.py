import itertools
import numpy as np
import os
import time
from typing import Dict, List, Union, Optional, Generator
from collections import defaultdict
from kaggle_environments.envs.kore_fleets.helpers import Configuration


IS_KAGGLE = os.path.exists("/kaggle_simulations")

# <--->
if IS_KAGGLE:
    from basic import (
        Obj,
        collection_rate_for_ship_count,
        max_ships_to_spawn,
        cached_call,
        cached_property,
        create_spawn_ships_command,
        create_launch_fleet_command,
    )
    from geometry import (
        Field,
        Action,
        Point,
        North,
        South,
        Convert,
        PlanPath,
        PlanRoute,
        GAME_ID_TO_ACTION,
        get_opposite_action,
    )
    from logger import logger
else:
    from .basic import (
        Obj,
        collection_rate_for_ship_count,
        max_ships_to_spawn,
        cached_call,
        cached_property,
        create_spawn_ships_command,
        create_launch_fleet_command,
    )
    from .geometry import (
        Field,
        Action,
        Point,
        North,
        South,
        Convert,
        PlanPath,
        PlanRoute,
        GAME_ID_TO_ACTION,
        get_opposite_action,
    )
    from .logger import logger

# <--->


class _ShipyardAction:
    def to_str(self):
        raise NotImplementedError

    def __repr__(self):
        return self.to_str()


class Spawn(_ShipyardAction):
    def __init__(self, ship_count: int):
        self.ship_count = ship_count

    def to_str(self):
        return create_spawn_ships_command(self.ship_count)


class Launch(_ShipyardAction):
    def __init__(self, ship_count: int, route: "BoardRoute"):
        self.ship_count = ship_count
        self.route = route

    def to_str(self):
        return create_launch_fleet_command(self.ship_count, self.route.plan.to_str())


class DoNothing(_ShipyardAction):
    def __repr__(self):
        return "Do nothing"

    def to_str(self):
        raise NotImplementedError


class DontLaunch(DoNothing):
    def __repr__(self):
        return "Don't launch"

    def to_str(self):
        raise NotImplementedError


class AllowMine(DoNothing):
    def __init__(self, max_distance: int = 15, target: Point = None, max_time: int = 30):
        self.max_distance = max_distance
        self.max_time = max_time
        self.target = target

    def __repr__(self):
        return f"AllowMine {self.max_distance} {self.target} {self.max_time}"

    def to_str(self):
        return NotImplementedError

class HailMary(DoNothing):
    def __init__(self):
        pass

    def __repr__(self):
        return f"HailMary"

    def to_str(self):
        return NotImplementedError

    def __nonzero__(self):
        return False

class BoardPath:
    max_length = 32

    def __init__(self, start: "Point", plan: PlanPath):
        assert plan.num_steps > 0 or plan.direction == Convert

        self._plan = plan

        field = start.field
        x, y = start.x, start.y
        if np.isfinite(plan.num_steps):
            n = plan.num_steps + 1
        else:
            n = self.max_length
        action = plan.direction

        if plan.direction == Convert:
            self._track = []
            self._start = start
            self._end = start
            self._build_shipyard = True
            return

        if action in (North, South):
            track = field.get_column(x, start=y, size=n * action.dy)
        else:
            track = field.get_row(y, start=x, size=n * action.dx)

        self._track = track[1:]
        self._start = start
        self._end = track[-1]
        self._build_shipyard = False

    def __repr__(self):
        start, end = self.start, self.end
        return f"({start.x}, {start.y}) -> ({end.x}, {end.y})"

    def __len__(self):
        return len(self._track)

    @property
    def plan(self):
        return self._plan

    @property
    def points(self):
        return self._track

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end


class BoardRoute:
    def __init__(self, start: "Point", plan: "PlanRoute", start_time: int = 0):
        paths = []
        for p in plan.paths:
            path = BoardPath(start, p)
            start = path.end
            paths.append(path)

        self._plan = plan
        self._paths = paths
        self._start = paths[0].start
        self._end = paths[-1].end
        self._start_time = start_time

    def __repr__(self):
        points = []
        for p in self._paths:
            points.append(p.start)
        points.append(self.end)
        return " -> ".join([f"({p.x}, {p.y})" for p in points])

    def __iter__(self) -> Generator["Point", None, None]:
        for p in self._paths:
            yield from p.points

    def __len__(self):
        return sum(len(x) for x in self._paths)

    def points(self) -> List["Point"]:
        points = []
        for p in self._paths:
            points += p.points
        return points

    @property
    def plan(self) -> PlanRoute:
        return self._plan

    def command(self) -> str:
        return self.plan.to_str()

    @property
    def paths(self) -> List[BoardPath]:
        return self._paths

    @property
    def start(self) -> "Point":
        return self._start

    @property
    def end(self) -> "Point":
        return self._end

    @property
    def start_time(self) -> int:
        return self._start_time

    def command_length(self) -> int:
        return len(self.command())

    def last_action(self):
        return self.paths[-1].plan.direction

    def expected_kore(self, board: "Board", ship_count: int):
        rate = collection_rate_for_ship_count(ship_count)
        if rate <= 0:
            return 0

        point_to_time = {}
        point_to_kore = {}
        for t, p in enumerate(self):
            point_to_time[p] = t + self._start_time
            point_to_kore[p] = p.kore

        for f in board.fleets:
            for t, p in enumerate(f.route):
                if p in point_to_time and t < point_to_time[p]:
                    point_to_kore[p] *= (1 - f.collection_rate)

        res = 0
        for p in self:
            res += point_to_kore[p] * rate
            point_to_kore[p] *= (1 - rate)
        return res

    def expected_kore_mining(self, board: "Board", ship_count: int):
        rate = collection_rate_for_ship_count(ship_count)
        if rate <= 0:
            return 0

        point_to_time = {}
        point_to_kore = {}
        for t, p in enumerate(self):
            point_to_time[p] = t + self._start_time
            point_to_kore[p] = p.kore

        for f in board.fleets:
            for t, p in enumerate(f.route):
                if p in point_to_time and t < point_to_time[p]:
                    point_to_kore[p] *= (1 - f.collection_rate)

        res = 0
        for p in self:
            res += point_to_kore[p]
            point_to_kore[p] *= (1 - rate)
        return res


class MiningRoute(BoardRoute):
    def __init__(self, start: "Point", plan: "PlanRoute", wait_time: int):
        super().__init__(start, plan)
        self._time_to_mine = wait_time

    @property
    def time_to_mine(self):
        return self._time_to_mine

    def can_execute(self):
        return self._time_to_mine == 0

    def __len__(self):
        return super().__len__() + self.time_to_mine


class PositionObj(Obj):
    def __init__(self, *args, point: Point, player_id: int, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self._point = point
        self._player_id = player_id
        self._board = board

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id}, position={self._point}, player={self._player_id})"

    def dirs_to_h(self, obj: Union["PositionObj", Point]):
        if isinstance(obj, Point):
            return self._point.dirs_to_h(obj)
        return self._point.dirs_to_h(obj.point)

    def get_plans_through(self, obj: List[Union["PositionObj", Point]]):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], Point):
            return self._point.get_plans_through(obj)
        return self._point.get_plans_through([o.point for o in obj])

    def distance_from(self, obj: Union["PositionObj", Point]) -> int:
        if isinstance(obj, Point):
            return self._point.distance_from(obj)
        return self._point.distance_from(obj.point)

    @property
    def board(self) -> "Board":
        return self._board

    @property
    def point(self) -> Point:
        return self._point

    @property
    def player_id(self):
        return self._player_id

    @property
    def player(self) -> "Player":
        return self.board.get_player(self.player_id)


class Shipyard(PositionObj):
    def __init__(self, *args, ship_count: int, turns_controlled: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._ship_count = ship_count
        self._turns_controlled = turns_controlled
        self._guard_ship_count = 0
        self.action: Optional[_ShipyardAction] = None
        self._blocked_dirs_at_time = None
        self._reserved_ship_count = 0

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id}, position={self._point} ship_count={self._ship_count})"

    @property
    def turns_controlled(self):
        return self._turns_controlled

    @property
    def max_ships_to_spawn(self) -> int:
        return max_ships_to_spawn(self._turns_controlled)

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def available_ship_count(self):
        return self._ship_count - self._guard_ship_count - self._reserved_ship_count

    @property
    def guard_ship_count(self):
        return self._guard_ship_count

    def set_guard_ship_count(self, ship_count):
        assert 0 <= ship_count <= self._ship_count
        self._guard_ship_count = ship_count

    def increase_reserved_ship_count(self, count):
        self._reserved_ship_count += count
        assert self._reserved_ship_count <= self._ship_count

    @cached_property
    def incoming_allied_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id == self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets

    @cached_property
    def incoming_hostile_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id != self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets

    @cached_property
    def future_ship_count(self):
        player = self.player
        board = self.board

        time_to_fleet_kore = defaultdict(int)
        for sh in player.all_shipyards:
            for f in sh.incoming_allied_fleets:
                time_to_fleet_kore[f.eta] += f.expected_kore()

        shipyard_reinforcements = defaultdict(int)
        for f in self.incoming_allied_fleets:
            shipyard_reinforcements[f.eta] += f.ship_count
        for f in self.incoming_hostile_fleets:
            shipyard_reinforcements[f.eta] -= f.ship_count

        spawn_cost = board.spawn_cost
        player_kore = player.kore
        ship_count = self.ship_count
        ship_counts = []
        for t in range(0, board.size + 1):
            ship_count += shipyard_reinforcements[t]
            ship_counts.append(ship_count)
            player_kore += time_to_fleet_kore[t]

            can_spawn = max_ships_to_spawn(self.turns_controlled + t)
            spawn_count = min(int(player_kore // spawn_cost), can_spawn)
            player_kore -= spawn_count * spawn_cost
            ship_count += spawn_count

        return ship_counts

    def estimate_shipyard_power(self, time):
        if time < 0:
            return 0
        if len(self.future_ship_count) <= time:
            return self.future_ship_count[-1]
        return self.future_ship_count[time] - self._guard_ship_count

    @cached_call
    def calc_time_for_ships_for_action(self, num_ships: int) -> int:
        for t in range(self.board.size + 1):
            if self.estimate_shipyard_power(t) >= num_ships:
                return t
        return 10000

    def can_launch_to_at_time(self, point: Point, time: int) -> bool:
        if self._blocked_dirs_at_time is None:
            self._blocked_dirs_at_time = self._get_blocked_dirs_at_time()
        plans = self.point.get_plans_through([point])
        for p in plans:
            if len(p.paths) == 0:
                continue
            if p.paths[0] not in self._blocked_dirs_at_time[time]:
                return True
        return False

    def _get_blocked_dirs_at_time(self) -> List[Action]:
        blocked_dirs_at_time = defaultdict(list)
        for f in self.incoming_allied_fleets:
            blocked_dirs_at_time[f.eta - 1].append(get_opposite_action(f.route.last_action()))
        return blocked_dirs_at_time

    # Assumes an idle turn if it can't spawn and it didn't just get a fleet
    @cached_property
    def idle_turns(self) -> List[int]:
        player = self.player
        board = self.board

        time_to_fleet_kore = defaultdict(int)
        for sh in player.all_shipyards:
            for f in sh.incoming_allied_fleets:
                time_to_fleet_kore[f.eta] += f.expected_kore()

        shipyard_reinforcements = defaultdict(int)
        for f in self.incoming_allied_fleets:
            shipyard_reinforcements[f.eta] += f.ship_count

        idle_count = 0
        spawn_cost = board.spawn_cost
        player_kore = player.kore
        idle_turns = [idle_count]
        for t in range(1, 2 * board.size + 1):
            player_kore += time_to_fleet_kore[t]

            can_spawn = max_ships_to_spawn(self.turns_controlled + t)
            spawn_count = min(int(player_kore // spawn_cost), can_spawn)
            if shipyard_reinforcements[t] == 0 and spawn_count == 0:
                idle_count += 1

            player_kore -= spawn_count * spawn_cost
            idle_turns.append(idle_count)

        return idle_turns

    def get_idle_turns_before(self, turn: int) -> int:
        if turn < 0:
            return 0
        if turn >= len(self.idle_turns):
            return self.idle_turns[-1]
        return self.idle_turns[turn]


class FutureShipyard(PositionObj):
    def __init__(self, *args, time_to_build: int, fleet_power: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_to_build = time_to_build
        self._fleet_power = fleet_power
        self._ship_count = fleet_power - 50

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id}, position={self._point}, player={self._player_id}, time_to_build={self._time_to_build}, fleet_power={self._fleet_power})"

    @property
    def time_to_build(self):
        return self._time_to_build

    @property
    def turns_controlled(self):
        return -self._time_to_build

    @property
    def ship_count(self):
        return self._ship_count

    @cached_property
    def incoming_allied_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id == self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets

    @cached_property
    def incoming_hostile_fleets(self) -> List["Fleet"]:
        fleets = []
        for f in self.board.fleets:
            if f.player_id != self.player_id and f.route.end == self.point:
                fleets.append(f)
        return fleets

    @cached_property
    def future_ship_count(self):
        player = self.player
        board = self.board

        time_to_fleet_kore = defaultdict(int)
        for sh in player.all_shipyards:
            for f in sh.incoming_allied_fleets:
                time_to_fleet_kore[f.eta] += f.expected_kore()

        shipyard_reinforcements = defaultdict(int)
        for f in self.incoming_allied_fleets:
            # FutureShipyard considers its own fleet as incoming
            if f.game_id != self.game_id:
                shipyard_reinforcements[f.eta] += f.ship_count
        for f in self.incoming_hostile_fleets:
            shipyard_reinforcements[f.eta] -= f.ship_count

        spawn_cost = board.spawn_cost
        player_kore = player.kore
        ship_count = self.ship_count
        ship_counts = []
        for t in range(0, board.size + 1):
            if t < self.time_to_build:
                ship_counts.append(0)
                continue

            ship_count += shipyard_reinforcements[t]
            ship_counts.append(ship_count)
            player_kore += time_to_fleet_kore[t]
 
            can_spawn = max_ships_to_spawn(t + self.turns_controlled)
            spawn_count = min(int(player_kore // spawn_cost), can_spawn)
            player_kore -= spawn_count * spawn_cost
            ship_count += spawn_count

        return ship_counts

    def estimate_shipyard_power(self, time):
        if time < 0:
            return 0
        if len(self.future_ship_count) <= time:
            return self.future_ship_count[-1]
        return self.future_ship_count[time]


class Fleet(PositionObj):
    def __init__(
        self,
        *args,
        ship_count: int,
        kore: int,
        route: BoardRoute,
        direction: Action,
        **kwargs,
    ):
        assert ship_count > 0
        assert kore >= 0

        super().__init__(*args, **kwargs)

        self._ship_count = ship_count
        self._kore = kore
        self._direction = direction
        self._route = route

    def __gt__(self, other):
        if self.ship_count != other.ship_count:
            return self.ship_count > other.ship_count
        if self.kore != other.kore:
            return self.kore > other.kore
        return self.direction.game_id > other.direction.game_id

    def __lt__(self, other):
        return other.__gt__(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id}, position={self._point}, player={self._player_id}, ship_count={self._ship_count}, kore={self._kore})"

    @property
    def ship_count(self):
        return self._ship_count

    @property
    def kore(self):
        return self._kore

    @property
    def route(self):
        return self._route

    @property
    def eta(self):
        return len(self._route)

    def set_route(self, route: BoardRoute):
        self._route = route

    @property
    def direction(self):
        return self._direction

    @property
    def collection_rate(self) -> float:
        return collection_rate_for_ship_count(self._ship_count)

    def expected_kore(self):
        return self._kore + self._route.expected_kore(self._board, self._ship_count)

    def cost(self):
        return self.board.spawn_cost * self.ship_count

    def value(self):
        return self.kore / self.cost()

    def expected_value(self):
        return self.expected_kore() / self.cost()


class FleetPointer:
    def __init__(self, fleet: Fleet):
        self.obj = fleet
        self.point = fleet.point
        self.is_active = True
        self._paths = []
        self._points = self.points()
        self._build_shipyard = None
        self.coalesced_fleets = []

    def points(self):
        for path in self.obj.route.paths:
            self._paths.append([path.plan.direction, 0])
            for f in self.coalesced_fleets:
                f._paths.append([path.plan.direction, 0])
            for point in path.points:
                self._paths[-1][1] += 1
                for f in self.coalesced_fleets:
                    f._paths[-1][1] += 1
                yield point

    def update(self):
        if not self.is_active:
            self.point = None
            self._build_shipyard = None
            return
        try:
            self.point = next(self._points)
        except StopIteration:
            if self._paths[-1][0].command == Convert.command:
                self._build_shipyard = self.point
            self.point = None
            self.is_active = False

    def current_route(self):
        plan = PlanRoute([PlanPath(d, n) for d, n in self._paths if n > 0 or d == Convert])
        return BoardRoute(self.obj.point, plan)

    @property
    def build_shipyard(self):
        return self._build_shipyard

class Player(Obj):
    def __init__(self, *args, kore: float, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self._kore = kore
        self._kore_reserve = 0
        self._board = board
        self._start_time = time.time()
        self._board_risk = None
        self._board_risk_not_adj = None
        self._optimistic_board_risk = None
        self._optimistic_board_risk_not_adj = None
        self.state = None
        self.memory = None

    @property
    def kore(self):
        return self._kore

    @property
    def kore_reserve(self):
        return self._kore_reserve

    def inc_kore_reserve(self, amount):
        self._kore_reserve += amount

    def set_kore_reserve(self, kore_reserve):
        assert kore_reserve <= self._kore
        self._kore_reserve = kore_reserve

    def fleet_kore(self):
        return sum(x.kore for x in self.fleets)

    def fleet_expected_kore(self):
        return sum(x.expected_kore() for x in self.fleets)

    def is_active(self):
        return len(self.fleets) > 0 or len(self.shipyards) > 0

    @property
    def board(self):
        return self._board

    def _get_objects(self, name):
        d = []
        for x in self._board.__getattribute__(name):
            if x.player_id == self.game_id:
                d.append(x)
        return d

    @cached_property
    def fleets(self) -> List[Fleet]:
        return self._get_objects("fleets")

    @cached_property
    def shipyards(self) -> List[Shipyard]:
        return self._get_objects("shipyards")

    @cached_property
    def future_shipyards(self) -> List[FutureShipyard]:
        return self._get_objects("future_shipyards")

    @cached_property
    def all_shipyards(self):
        return self._get_objects("all_shipyards")

    @cached_property
    def ship_count(self) -> int:
        return sum(x.ship_count for x in itertools.chain(self.fleets, self.shipyards))

    @cached_property
    def opponents(self) -> List["Player"]:
        return [x for x in self.board.players if x != self]

    @cached_property
    def expected_fleets_positions(self) -> Dict[int, Dict[Point, Fleet]]:
        """
        time -> point -> fleet
        """
        time_to_fleet_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                time_to_fleet_positions[time][point] = f
        return time_to_fleet_positions

    @cached_property
    def expected_dmg_positions(self) -> Dict[int, Dict[Point, int]]:
        """
        time -> point -> dmg
        """
        time_to_dmg_positions = defaultdict(dict)
        for f in self.fleets:
            for time, point in enumerate(f.route):
                for adjacent_point in point.adjacent_points:
                    point_to_dmg = time_to_dmg_positions[time]
                    if adjacent_point not in point_to_dmg:
                        point_to_dmg[adjacent_point] = 0
                    point_to_dmg[adjacent_point] += f.ship_count
        return time_to_dmg_positions
    
    @cached_property
    def shipyard_production_capacity(self):
        return sum(x.max_ships_to_spawn for x in self.shipyards)

    @cached_property
    def avg_shipyard_production_capacity(self):
        return self.shipyard_production_capacity / max(len(self.shipyards), 1)

    @cached_property
    def adj_shipyard_production_capacity(self, min_prod: int = 5):
        return sum(max(min_prod, x.max_ships_to_spawn) for x in self.shipyards)

    def actions(self):
        if self.available_kore() < 0:
            logger.warning("Negative balance. Some ships will not spawn.")

        shipyard_id_to_action = {}
        for sy in self.shipyards:
            if not sy.action or isinstance(sy.action, DoNothing):
                continue

            shipyard_id_to_action[sy.game_id] = sy.action.to_str()
        return shipyard_id_to_action

    def spawn_ship_count(self):
        return sum(
            x.action.ship_count for x in self.shipyards if isinstance(x.action, Spawn)
        )

    def need_kore_for_spawn(self):
        return self.board.spawn_cost * self.spawn_ship_count()

    def available_kore(self):
        return self._kore - self.need_kore_for_spawn() - self.kore_reserve

    def time_remaining(self):
        return self.board.act_timeout - (time.time() - self._start_time)

    def update_state(self):
        self.state.act(self)
        if self.state.is_finished():
            self.state = self.state.next_state()

    def update_state_if_is(self, state_class):
        res = self.state.__repr__() != "State"
        if isinstance(self.state, state_class):
            self.update_state()
            return True
        return res

    def estimate_power_for_point_at_time(self, point: Point, time: int) -> int:
        if time < 0:
            return 0

        power = max(
            (sy.estimate_shipyard_power(time - sy.distance_from(point))
            for sy in self.all_shipyards),
            default=0
        )

        return power

    def estimate_optimistic_power_for_point_at_time(self, point: Point, time: int) -> int:
        if time < 0:
            return 0

        power = max(
            ((sy.estimate_shipyard_power(time - sy.distance_from(point)) - sy.ship_count)
            for sy in self.all_shipyards),
            default=0
        )

        return power

    def is_board_risk_worth(self, risk: int, num_ships: int, sy: Shipyard) -> bool:
        if risk >= num_ships:
            if self.board.step < 50 or self.ship_count < 50 or \
                num_ships > 0.2 * self.ship_count or num_ships > 0.5 * sy.estimate_shipyard_power(10):
                return False
            return num_ships > risk * 0.75
        return True

    def estimate_board_risk(self, p: Point, time: int, max_time: int = 40, pessimistic: bool = True) -> int:
        if self._board_risk is None:
            self._board_risk, self._board_risk_not_adj = self._estimate_board_risk()
            self._optimistic_board_risk, self._optimistic_board_risk_not_adj = self._estimate_board_risk(pessimistic=False)
        if time < 0:
            return 0
        time = min(time, max_time)
        return self._board_risk[p][time] if pessimistic else self._optimistic_board_risk[p][time]

    def estimate_board_risk_not_adj(self, p: Point, time: int, max_time: int = 40, pessimistic: bool = True) -> int:
        if self._board_risk is None:
            self._board_risk, self._board_risk_not_adj = self._estimate_board_risk()
            self._optimistic_board_risk, self._optimistic_board_risk_not_adj = self._estimate_board_risk(pessimistic=False)
        if time < 0:
            return 0
        time = min(time, max_time)
        return self._board_risk_not_adj[p][time] if pessimistic else self._optimistic_board_risk_not_adj[p][time]

    def _estimate_board_risk(self, max_time: int = 40, pessimistic: bool = True) -> Dict[Point, Dict[int, int]]:
        board = self.board
        opps = self.opponents
        if len(opps) < 1:
            return {}
        opp = opps[0]
        if pessimistic:
            func = lambda p, dt: opp.estimate_power_for_point_at_time(p, dt)
        else:
            func = lambda p, dt: opp.estimate_optimistic_power_for_point_at_time(p, dt)

        point_to_time_to_score = defaultdict(dict)
        for p in board:
            for dt in range(max_time + 1):
                point_to_time_to_score[p][dt] = func(p, dt)

        adj_point_to_time_to_score = defaultdict(dict)
        for p in board:
            for dt in range(max_time + 1):
                adj_point_to_time_to_score[p][dt] = max(
                    point_to_time_to_score[adj_p][dt]
                    for adj_p in p.adjacent_points
                )

        return adj_point_to_time_to_score, point_to_time_to_score


_FIELD = None


class Board:
    def __init__(self, obs, conf):
        self._conf = Configuration(conf)
        self._step = obs["step"]

        global _FIELD
        if _FIELD is None or self._step == 0:
            _FIELD = Field(self._conf.size)
        else:
            assert _FIELD.size == self._conf.size

        self._field: Field = _FIELD

        id_to_point = {x.game_id: x for x in self._field}

        for point_id, kore in enumerate(obs["kore"]):
            point = id_to_point[point_id]
            point.set_kore(kore)

        self._players = []
        self._fleets = []
        self._shipyards = []
        self._future_shipyards = []
        for player_id, player_data in enumerate(obs["players"]):
            player_kore, player_shipyards, player_fleets = player_data
            player = Player(game_id=player_id, kore=player_kore, board=self)
            self._players.append(player)

            for fleet_id, fleet_data in player_fleets.items():
                point_id, kore, ship_count, direction, flight_plan = fleet_data
                position = id_to_point[point_id]
                direction = GAME_ID_TO_ACTION[direction]
                build_shipyard = False
                if Convert.command in flight_plan:
                    if ship_count < self.shipyard_cost:
                        # can't convert
                        flight_plan = "".join(
                            [x for x in flight_plan if x != Convert.command]
                        )
                    else:
                        # Delete everything after the convert command
                        flight_plan = flight_plan[:flight_plan.index(Convert.command) + 1]
                        build_shipyard = True
                plan = PlanRoute.from_str(flight_plan, direction)
                route = BoardRoute(position, plan)
                fleet = Fleet(
                    game_id=fleet_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    kore=kore,
                    route=route,
                    direction=direction,
                    board=self,
                )
                self._fleets.append(fleet)

                if build_shipyard:
                    future_shipyard = FutureShipyard(
                        game_id=fleet_id,
                        point=route.end,
                        player_id=player_id,
                        time_to_build=len(route) + 1,
                        fleet_power=ship_count,
                        board=self,
                    )
                    self._future_shipyards.append(future_shipyard)

            for shipyard_id, shipyard_data in player_shipyards.items():
                point_id, ship_count, turns_controlled = shipyard_data
                position = id_to_point[point_id]
                shipyard = Shipyard(
                    game_id=shipyard_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    turns_controlled=turns_controlled,
                    board=self,
                )
                self._shipyards.append(shipyard)

        self._players = [x for x in self._players if x.is_active()]

        self._update_fleets_destination()

    def __getitem__(self, item):
        return self._field[item]

    def __iter__(self):
        return self._field.__iter__()

    @property
    def field(self):
        return self._field

    @property
    def size(self):
        return self._field.size

    @property
    def step(self):
        return self._step

    @property
    def steps_left(self):
        return self._conf.episode_steps - self._step - 1

    @property
    def shipyard_cost(self):
        return self._conf.convert_cost

    @property
    def spawn_cost(self):
        return self._conf.spawn_cost

    @property
    def regen_rate(self):
        return self._conf.regen_rate

    @property
    def max_cell_kore(self):
        return self._conf.max_cell_kore

    @property
    def act_timeout(self):
        return self._conf.act_timeout

    @property
    def players(self) -> List[Player]:
        return self._players

    @property
    def fleets(self) -> List[Fleet]:
        return self._fleets

    @property
    def shipyards(self) -> List[Shipyard]:
        return self._shipyards

    @property
    def future_shipyards(self) -> List[FutureShipyard]:
        return self._future_shipyards

    @property
    def all_shipyards(self):
        return itertools.chain(self._shipyards, self._future_shipyards)

    @cached_property
    def total_kore(self) -> int:
        return sum(x.kore for x in self)

    def get_player(self, game_id) -> Player:
        for p in self._players:
            if p.game_id == game_id:
                return p
        raise KeyError(f"Player `{game_id}` does not exist.")

    def get_obj_at_point(self, point: Point) -> Optional[Union[Fleet, Shipyard]]:
        for x in itertools.chain(self.fleets, self.shipyards):
            if x.point == point:
                return x

    def _update_fleets_destination(self):
        """
        trying to predict future positions
        very inaccurate
        """

        shipyard_positions = {x.point for x in self.shipyards}

        fleets = [FleetPointer(f) for f in self.fleets]

        while any(x.is_active for x in fleets):
            for f in fleets:
                f.update()

                # fleet shipyard conversions
                if f.build_shipyard:
                    shipyard_positions.add(f.build_shipyard)

            # fleet to shipyard
            for f in fleets:
                if f.point in shipyard_positions:
                    f.is_active = False

            # allied fleets
            for player in self.players:
                point_to_fleets = defaultdict(list)
                for f in fleets:
                    if f.is_active and f.obj.player_id == player.game_id:
                        point_to_fleets[f.point].append(f)
                for point_fleets in point_to_fleets.values():
                    if len(point_fleets) > 1:
                        sorted_fleets = sorted(point_fleets, key=lambda x: x.obj)
                        last_fleet = sorted_fleets[-1]
                        for f in sorted_fleets[:-1]:
                            f.is_active = False
                            last_fleet.coalesced_fleets.append(f)
                            f._paths.append([last_fleet._paths[-1][0], 0])

            # fleet to fleet
            point_to_fleets = defaultdict(list)
            for f in fleets:
                if f.is_active:
                    point_to_fleets[f.point].append(f)
            for point_fleets in point_to_fleets.values():
                if len(point_fleets) > 1:
                    for f in sorted(point_fleets, key=lambda x: x.obj)[:-1]:
                        f.is_active = False

            # adjacent damage
            point_to_fleet = {}
            for f in fleets:
                if f.is_active:
                    point_to_fleet[f.point] = f

            point_to_dmg = defaultdict(int)
            for point, fleet in point_to_fleet.items():
                for p in point.adjacent_points:
                    if p in point_to_fleet:
                        adjacent_fleet = point_to_fleet[p]
                        if adjacent_fleet.obj.player_id != fleet.obj.player_id:
                            point_to_dmg[p] += fleet.obj.ship_count

            for point, fleet in point_to_fleet.items():
                dmg = point_to_dmg[point]
                if fleet.obj.ship_count <= dmg:
                    fleet.is_active = False

        for f in fleets:
            f.obj.set_route(f.current_route())

from src.Alpha.geometry import *
from src.Alpha.board import *
from src.Alpha.logger import *

field = Field(21)
p = Point(5, 5, 10, field)
print(p)
path = PlanRoute.from_str("2W", North)
route = BoardRoute(p, path)
print(path)
print(route, route.end)

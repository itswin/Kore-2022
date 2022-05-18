from src.Alpha.main import agent as Alpha
from src.KoreBeta.main import agent as KoreBeta
from src.Miner.main import agent as Miner

from datetime import datetime
from kaggle_environments import make

env = make("kore_fleets")
env.run([Alpha, Miner])

now = datetime.now()
file_name = now.strftime("games/game_%m-%d_%H:%M:%S.html")
game_out = env.render(mode="html")
with open(file_name, "w") as f:
    f.write(game_out)
 
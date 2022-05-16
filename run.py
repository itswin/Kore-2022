from src.KoreBeta.main import agent
from datetime import datetime
from kaggle_environments import make

env = make("kore_fleets")
env.run([agent, agent])

game_out = env.render(mode="html")
with open("game_out.html", "w") as f:
    f.write(game_out)
 
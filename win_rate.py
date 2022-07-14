from kaggle_environments import make
from tqdm.contrib.concurrent import process_map
import numpy as np
from datetime import datetime
import traceback

from src.Alpha.multi import make_agent as Alpha
from src.Beta.multi import make_agent as Beta
from src.KoreBeta.main import agent as KoreBeta
from src.Miner.main import agent as Miner

# agenta, agentb = Alpha, Beta

No_games_to_run = 50
show_result_per_game = True

def runs(i):
    env = make("kore_fleets")
    agenta = Alpha()
    agentb = Beta()
    env.run([agenta, agentb])

    now = datetime.now()
    file_name = now.strftime(f"games/game{i}_%m-%d_%H:%M:%S.html")
    game_out = env.render(mode="html")
    with open(file_name, "w") as f:
        f.write(game_out)

    rewards = [x["reward"] for x in env.steps[-1]]
    scores1, scores2 = rewards[0], rewards[1]

    if scores2 == None:
        wins = True
        raise Exception(f"Error for P2: No rewards in game no. {i}")
    elif scores1 == None:
        wins = False
        raise Exception(f"Error for P1: No rewards in game no. {i}")
    else:
        wins = scores1 > scores2

    if show_result_per_game:
        what = 'Win' if wins==1 else 'Lost'
        print(f'Game no. #{i} : {what} with score {scores1:.0f} vs {scores2:.0f}')

    return scores1, scores2, wins


if __name__ == '__main__':
    env = make("kore_fleets", debug=True)
    print(env.name, env.version)

    results = process_map(runs, range(No_games_to_run))
    print(f' Win rate {100*np.array(results)[:,2].sum()/No_games_to_run}% with mean score {np.array(results)[:,0].mean():.0f} vs {np.array(results)[:,1].mean():.0f} ')

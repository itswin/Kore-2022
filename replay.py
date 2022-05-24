import json
from src.Alpha.main import agent

FROM, TO = 270, 400         # Replay steps range
PLAYER = 1                  # Player number
FILE="games/37262303.json"          # replay file name, can be '*.json' or '*.html'

with open(FILE, "r") as cin:
    f = cin.read()

if FILE.endswith('.html'):
    start = 'window.kaggle = '
    end = 'window.kaggle.renderer = '
    n_start = f.find(start) + len(start)
    n_end = f.find(end) - 4
    f = f[n_start:n_end]

r = json.loads(f)

env = r.get('environment', r)

conf = env['configuration']

# Initialize logger
obs = env['steps'][0][0]['observation']
obs["player"] = PLAYER
actions = agent(obs, conf)

for step in range(FROM, TO+1):
    obs = env['steps'][step][0]['observation']
    obs["player"] = PLAYER

    actions = agent(obs, conf)

    print(f'{step}: {actions}')

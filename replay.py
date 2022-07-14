#!/usr/bin/env python

import json
import cProfile
import pstats
from pstats import SortKey
from src.Alpha.main import agent

FROM, TO = 273, 280    # Replay steps range
PLAYER = 0                 # Player number
FILE="games/42197321.json"          # replay file name, can be '*.json' or '*.html'

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

for step in range(FROM-1, TO):
    obs = env['steps'][step][0]['observation']
    # print(obs)
    obs["player"] = PLAYER

    # cProfile.run("actions = agent(obs, conf)", "restats")
    # p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats(SortKey.TIME).print_stats()
    actions = agent(obs, conf)

    print(f'{step}: {actions}')

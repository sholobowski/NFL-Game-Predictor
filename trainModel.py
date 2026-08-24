import gameData as gd
import pandas as pd
import nflreadpy as nfl
import numpy as np
import re

years = [2024, 2023, 2022, 2021]

game_data = gd.create_game_data(years)
game_data= game_data.dropna()

features = [n for n in game_data.columns if re.match(r"^ewma", n)]
for feat in features:
    print(feat)

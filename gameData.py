import pandas as pd
import nflreadpy as nfl

all_stats = nfl.load_team_stats(seasons=True).to_pandas()

all_stats.to_csv("output.txt", sep="\t", index=False)
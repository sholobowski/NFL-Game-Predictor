import pandas as pd
import nflreadpy as nfl
import numpy as np

# collect all pbp data from previous 4 years and convert to a pandas DataFrame
all_stats = nfl.load_pbp([2025, 2024, 2023, 2022]).to_pandas()

# all_stats.to_csv("output.txt", sep="\t", index=False)

def create_dynamic_window(series):
    # create empty values array
    vals = np.zeros(len(series))
    # loop through each row in passed frame
    for i, (_, data) in enumerate(series.iterrows()):
        # grabs every epa from previous weeks up until current indexed week
        epa_this_week = series.epa_last_week[:i+1]
        # using 5 as the rolling average, i.e. if a game is before week 6, the previous season's data rollover will be factored in.
        # if a game is on or after week 6, only the current season's game data will be used
        if data.week < 6:
            # produces exponentially weighted averages, and grabs the last value as that contains the current average
            vals[i] = epa_this_week.ewm(min_periods=1,span=6).mean().values[-1]
        else:
            vals[i] = epa_this_week.ewn(min_periods=1,span=data.week).mean().values[-1]
    # return epas in a pandas series
    return pd.Series(vals, index=series.index)

# collect epa from plays containg a rush/pass attempt from the offense/defense perspective
# shift the epa from previous week up to next week as to prevent look-ahead bias
off_rush_epa = all_stats.loc[all_stats['rush_attempt'] == 1, :].groupby(['posteam','season','week'], as_index=False)['epa'].mean()
off_rush_epa['epa_last_week'] = off_rush_epa.groupby('posteam')['epa'].shift()

def_rush_epa = all_stats.loc[all_stats['rush_attempt'] == 1, :].groupby(['defteam','season','week'], as_index=False)['epa'].mean()
def_rush_epa['epa_last_week'] = def_rush_epa.groupby('defteam')['epa'].shift()

off_pass_epa = all_stats.loc[all_stats['pass_attempt'] == 1, :].groupby(['posteam','season','week'], as_index=False)['epa'].mean()
off_pass_epa['epa_last_week'] = off_pass_epa.groupby('posteam')['epa'].shift()

def_pass_epa = all_stats.loc[all_stats['pass_attempt'] == 1, :].groupby(['defteam','season','week'], as_index=False)['epa'].mean()
def_pass_epa['epa_last_week'] = def_pass_epa.groupby('defteam')['epa'].shift()

off_rush_epa.to_csv("output.txt", sep="\t", index=False)


# def create_dynamic_window(frame):
#    vals = np.zeros(len(frame))
#    for i, (_, r) in enumerate(frame.iterrows()):


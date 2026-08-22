import pandas as pd
import nflreadpy as nfl
import numpy as np

# range of seasons which can be modified to increase/decrease dataset as well as
# manipulate later resulting EPA DataFrame
years = [2024, 2023, 2022, 2021]

# collect all pbp data from previous 4 years and convert to a pandas DataFrame
all_stats = nfl.load_pbp(years).to_pandas()

# all_stats.to_csv("output.txt", sep="\t", index=False)

def create_dynamic_window(series):
    # create empty values array
    vals = np.zeros(len(series))
    # loop through each row in passed frame
    for i, (_, data) in enumerate(series.iterrows()):
        # grabs every EPA from previous weeks up until current indexed week
        epa_this_week = series.epa_last_week[:i+1]
        # using 5 as the rolling average, i.e. if a game is before week 6, the 
        # previous season's data rollover will be factored in. if a game is on 
        # or after week 6, only the current season's game data will be used
        if data.week < 6:
            # produces exponentially weighted averages, and grabs the last value
            #  as that contains the current average
            vals[i] = epa_this_week.ewm(min_periods=1,span=6).mean().values[-1]
        else:
            vals[i] = epa_this_week.ewm(min_periods=1,span=data.week).\
                mean().values[-1]
    # return epas in a pandas series
    return pd.Series(vals, index=series.index)

# collect EPA from plays containg a rush/pass attempt from the offense/defense 
# perspective
off_rush_epa = all_stats.loc[all_stats['rush_attempt'] == 1, :].groupby(\
    ['posteam','season','week'], as_index=False)['epa'].mean()
# shift the EPA from previous week up to next week as to prevent 
# look-ahead bias
off_rush_epa['epa_last_week'] = off_rush_epa.groupby('posteam')['epa'].shift()
# applies EWMA with dynamic window and stores in list using previously defined 
# create_dynamic_window def
off_rush_epa['ewma'] = off_rush_epa.groupby('posteam').apply(\
    create_dynamic_window).values
# rename columns for future merge
off_rush_epa = off_rush_epa.rename(columns={'posteam':'team',\
    'epa':'off_rush_epa','epa_last_week':'off_rush_epa_last_week',\
        'ewma':'off_rush_ewma'})

# defense rushing EPA and EWMA:
def_rush_epa = all_stats.loc[all_stats['rush_attempt'] == 1, :].groupby(\
    ['defteam','season','week'], as_index=False)['epa'].mean()
def_rush_epa['epa_last_week'] = def_rush_epa.groupby('defteam')['epa'].shift()
def_rush_epa['ewma'] = def_rush_epa.groupby('defteam').apply(\
    create_dynamic_window).values
def_rush_epa = def_rush_epa.rename(columns={'defteam':'team',\
    'epa':'def_rush_epa','epa_last_week':'def_rush_epa_last_week',\
        'ewma':'def_rush_ewma'})

# offense passing EPA and EWMA:
off_pass_epa = all_stats.loc[all_stats['pass_attempt'] == 1, :].groupby(\
    ['posteam','season','week'], as_index=False)['epa'].mean()
off_pass_epa['epa_last_week'] = off_pass_epa.groupby('posteam')['epa'].shift()
off_pass_epa['ewma'] = off_pass_epa.groupby('posteam').apply(\
    create_dynamic_window).values
off_pass_epa = off_pass_epa.rename(columns={'posteam':'team',\
    'epa':'off_pass_epa','epa_last_week':'off_pass_epa_last_week',\
        'ewma':'off_pass_ewma'})

# defense passing EPA and EWMA:
def_pass_epa = all_stats.loc[all_stats['pass_attempt'] == 1, :].groupby(\
    ['defteam','season','week'], as_index=False)['epa'].mean()
def_pass_epa['epa_last_week'] = def_pass_epa.groupby('defteam')['epa'].shift()
def_pass_epa['ewma'] = def_pass_epa.groupby('defteam').apply(\
    create_dynamic_window).values
def_pass_epa = def_pass_epa.rename(columns={'defteam':'team',\
    'epa':'def_pass_epa','epa_last_week':'def_pass_epa_last_week',\
        'ewma':'def_pass_ewma'})

off_rush_epa.to_csv("off_rush_epa.txt", sep="\t", index=False)
def_rush_epa.to_csv("def_rush_epa.txt", sep="\t", index=False)
off_pass_epa.to_csv("off_pass_epa.txt", sep="\t", index=False)
def_pass_epa.to_csv("def_pass_epa.txt", sep="\t", index=False)

# merge offense/defense rush/pass EPAs together, then merge offense/defense
off_epa = off_rush_epa.merge(off_pass_epa, on=['team','season','week'])
def_epa = def_rush_epa.merge(def_pass_epa, on=['team','season','week'])
team_epa = off_epa.merge(def_epa, on=['team','season','week'])

# first season is removed below because any EPA/EWMA calculated has no previous 
# data for week 1. due to this, the season does not utilize previous historical 
# data when calculating EWMA. therefore, the season is removed entirely such 
# that every season used historical data from previous seasons in the EWMAs
team_epa = team_epa[team_epa['season'] != years[-1]]
team_epa = team_epa.reset_index(drop=True)

team_epa.to_csv("team_epa.txt", sep="\t", index=False)
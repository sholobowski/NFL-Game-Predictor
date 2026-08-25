import gameData as gd
import pandas as pd
import nflreadpy as nfl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

years = list(range(2025,2020, -1))

game_data = gd.create_game_data(years)
game_data = game_data.dropna()

features = game_data.columns[game_data.columns.str.contains("ewma")].tolist()

f_data = game_data.loc[game_data['season']!= 2025, features].values
t_data = game_data.loc[game_data['season']!= 2025, 'win'].values

log_reg = LogisticRegression()
log_reg.fit(f_data,t_data)

accuracy = cross_val_score(log_reg, f_data, t_data, cv=6)
losses = cross_val_score(log_reg, f_data, t_data, cv=6, scoring='neg_log_loss')

last_year_data = game_data.loc[(game_data['season'] == 2025)].assign(\
    predicted_winner = lambda n: log_reg.predict(n[features]),\
    home_team_win_prob = lambda n: log_reg.predict_proba(n[features])[:,1])\
    [['home_team','away_team','week','predicted_winner','home_team_win_prob','win']]

last_year_data['actual_winner'] = last_year_data.apply(lambda n: n.home_team if n.win else n.away_team, axis=1)
last_year_data['predicted_winner'] = last_year_data.apply(lambda n: n.home_team if n.predicted_winner == 1 else n.away_team, axis=1)
last_year_data['win_prob'] = last_year_data.apply(lambda n: n.home_team_win_prob if n.predicted_winner == n.home_team else 1-n.home_team_win_prob, axis=1)
last_year_data['correct_prediction'] = (last_year_data['predicted_winner']==last_year_data['actual_winner']).astype(int)

last_year_data = last_year_data.drop(columns=['home_team_win_prob','win'])
last_year_data.sort_values(by='win_prob',ascending=False).reset_index(drop=True)

last_year_data.to_csv("2025_data.xlsx", sep="\t", index=False)
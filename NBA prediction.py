import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import requests
import io

print("NBA Game Outcome Prediction.")

print("\nDownloading and loading dataset directly from source...")
url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
response = requests.get(url)

data = pd.read_csv(io.StringIO(response.text))
print(f"Dataset loaded: {data.shape[0]:,} games x {data.shape[1]} variables")

print("\nData preview (first 3 games):")
showcolumns = ['date_game', 'fran_id', 'opp_fran', 'game_location', 'pts', 'opp_pts']
print(data[showcolumns].head(3))
print("\nfran_id: Franchise name, opp_fran: Opponent franchise name")

print("\nPrepare data for modeling:")
print("1: Compute home team Elo rating advantage.")
print("2: Create target variable: home team win (1) or loss (0).")

data['is_home_team'] = (data['game_location'] == 'H').astype(int)
data['elo_home'] = np.where(data['is_home_team'] == 1, data['elo_i'], data['opp_elo_i'])
data['elo_away'] = np.where(data['is_home_team'] == 1, data['opp_elo_i'], data['elo_i'])
data['elo_diff'] = data['elo_home'] - data['elo_away']
data['team_win'] = (data['pts'] > data['opp_pts']).astype(int)
data['home_win'] = np.where(data['is_home_team'] == 1, data['team_win'], 1 - data['team_win'])
data['date_game'] = pd.to_datetime(data['date_game'], errors='coerce')

print(f"Home team win rate: {data['home_win'].mean():.1%}")

print("\nFull NBA history victory prediction (1946-2015):")
print("Train first 80% of chronological data, test remaining 20%.\n")

split_full = int(0.8 * len(data))
X_full = data[['elo_diff', 'forecast']].fillna(0)
y_full = data['home_win']

lr_full = LogisticRegression(random_state=42, max_iter=200)
lr_full.fit(X_full.iloc[:split_full], y_full.iloc[:split_full])
full_accuracy = accuracy_score(y_full.iloc[split_full:], lr_full.predict(X_full.iloc[split_full:]))
full_baseline = y_full.iloc[split_full:].mean()

print(f"Model accuracy:           {full_accuracy:.1%}")
print(f"Always predict home win: {full_baseline:.1%}")
print(f"Model improvement:        {full_accuracy - full_baseline:.1%}")

print("\nModern NBA history victory prediction (2010-2015):")
print("Using only raw team strength difference (no pre-computed predictions).\n")

recent_data = data[data['date_game'] >= '2010-01-01'].copy()
print(f"Modern dataset: {len(recent_data):,} games")

X_recent = recent_data[['elo_diff']].fillna(0)
y_recent = recent_data['home_win']
split_recent = int(0.8 * len(recent_data))

lr_recent = LogisticRegression(random_state=42)
lr_recent.fit(X_recent.iloc[:split_recent], y_recent.iloc[:split_recent])
recent_accuracy = accuracy_score(y_recent.iloc[split_recent:], lr_recent.predict(X_recent.iloc[split_recent:]))
recent_baseline = y_recent.iloc[split_recent:].mean()

print(f"Modern model accuracy:    {recent_accuracy:.1%}")
print(f"Modern baseline:          {recent_baseline:.1%}")
print(f"Improvement:              {recent_accuracy - recent_baseline:.1%}")

print("\nDetailed classification report for the Modern Era:")
print(classification_report(y_recent.iloc[split_recent:], 
                          lr_recent.predict(X_recent.iloc[split_recent:]), 
                          target_names=['Home Loss', 'Home Win']))

print("\nrecall: when the situation actually happens (home win/loss), how often the model correctly predicts it.")
print("f1-score: balances false positives and false negatives taking into account precision and recall.")

print("\n\nDataset Source:", url)
print("\nMain results:")
print(f"  Full history accuracy: {full_accuracy:.1%}")
print(f"  Modern era accuracy:   {recent_accuracy:.1%}")

print("\nFor reproducibility:")
print("  Random seed: 42")
print("\nDirect data downloaded from source, so no need to download separately.")
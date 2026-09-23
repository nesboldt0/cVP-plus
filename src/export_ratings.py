import time
import joblib
import numpy as np
import pandas as pd
from pybaseball import playerid_reverse_lookup

t_start = time.time()

df = pd.read_csv("cleaned_pitches_2025.csv")
m = joblib.load("xrv_quality_model_2025.joblib")

# pitcher pov horizontal movement
is_rhp = (df['p_throws'] == 'R').astype(int)
is_rhb = (df['stand'] == 'R').astype(int)
platoon = (is_rhp == is_rhb).astype(int)

hb_p = []
for i in range(len(df)):
    if is_rhp.iloc[i] == 1:
        hb_p.append(-df['pfx_x'].iloc[i])
    else:
        hb_p.append(df['pfx_x'].iloc[i])
hb_p = pd.Series(hb_p)

# zone edge dist
x_dist = np.abs(df['plate_x']) - 0.708
dx = np.maximum(0, x_dist)

z_dist = np.abs(df['plate_z'] - 2.5) - 0.85
dz = np.maximum(0, z_dist)

d_edge = np.sqrt(dx * dx + dz * dz)

feats = pd.DataFrame()
feats['b'] = df['balls']
feats['s'] = df['strikes']
feats['platoon'] = platoon
feats['rhp'] = is_rhp
feats['velo'] = df['release_speed']
feats['spin'] = df['release_spin_rate']
feats['ext'] = df['release_extension']
feats['hb'] = hb_p.values
feats['ivb'] = df['pfx_z']
feats['px'] = df['plate_x']
feats['pz'] = df['plate_z']
feats['d_edge'] = d_edge

# invert: positive xrv = run prevention
preds = m.predict(feats)
df['xrv'] = -preds

# identify starts using pitch volume per game_date
# outings with >= 50 pitches count as a start
game_counts = df.groupby(['pitcher', 'game_date']).size().reset_index()
game_counts.columns = ['pitcher', 'game_date', 'game_pitches']

started_games = game_counts[game_counts['game_pitches'] >= 50]
starts = started_games.groupby('pitcher').size().reset_index()
starts.columns = ['pitcher', 'gs']

# group per pitcher
grp = df.groupby('pitcher').agg({
    'xrv': ['count', 'sum'],
    'release_speed': 'mean',
    'is_whiff': 'mean',
    'in_zone': 'mean'
}).reset_index()

grp.columns = ['pitcher', 'n_pitches', 'tot_xrv', 'velo', 'whiff', 'zone']

grp = grp.merge(starts, on='pitcher', how='left')
grp['gs'] = grp['gs'].fillna(0)
grp['gs'] = grp['gs'].astype(int)

# 400 pitch min across league for scaling
MIN_P = 400
qual = grp[grp['n_pitches'] >= MIN_P].copy()
qual['cpv_per_100'] = (qual['tot_xrv'] / qual['n_pitches']) * 100.0

# 100 index scale
mu = qual['cpv_per_100'].mean()
sd = qual['cpv_per_100'].std()

z_score = (qual['cpv_per_100'] - mu) / sd
qual['cPV_plus'] = np.round(100.0 + (z_score * 15.0), 1)

# mlbam id lookup
p_ids = qual['pitcher'].unique().tolist()
lut = playerid_reverse_lookup(p_ids, key_type='mlbam')

first_names = lut['name_first'].str.capitalize()
last_names = lut['name_last'].str.capitalize()
lut['full_name'] = first_names + ' ' + last_names

id2name = {}
for i in range(len(lut)):
    mlb_id = lut['key_mlbam'].iloc[i]
    name = lut['full_name'].iloc[i]
    id2name[mlb_id] = name

qual['player_name'] = qual['pitcher'].map(id2name)
qual['player_name'] = qual['player_name'].fillna("Unknown")

cols = ['player_name', 'pitcher', 'gs', 'n_pitches', 'velo', 'whiff', 'zone', 'tot_xrv', 'cPV_plus']
qual = qual.sort_values('cPV_plus', ascending=False)
qual = qual[cols]

# split starters and relievers
is_starter = (qual['gs'] >= 5) & (qual['n_pitches'] >= 1000)
starters = qual[is_starter].copy()

is_reliever = (qual['gs'] < 5) & (qual['n_pitches'] >= 400)
relievers = qual[is_reliever].copy()

# repertoire splits
sub = df[df['pitcher'].isin(qual['pitcher'])].copy()

ars = sub.groupby(['pitcher', 'pitch_type']).agg({
    'xrv': ['count', 'sum'],
    'release_speed': 'mean',
    'pfx_x': 'mean',
    'pfx_z': 'mean',
    'is_whiff': 'mean',
    'in_zone': 'mean'
}).reset_index()

ars.columns = ['pitcher', 'pitch_type', 'n_pitches', 'tot_xrv', 'velo', 'hb', 'ivb', 'whiff', 'zone']

ars['cpv_per_100'] = (ars['tot_xrv'] / ars['n_pitches']) * 100.0
ars_z = (ars['cpv_per_100'] - mu) / sd
ars['pitch_cPV_plus'] = np.round(100.0 + (ars_z * 15.0), 1)

ars['player_name'] = ars['pitcher'].map(id2name)
ars['player_name'] = ars['player_name'].fillna("Unknown")

# exports
qual.to_csv("pitcher_cPV_ratings_2025.csv", index=False)
starters.to_csv("starter_cPV_ratings_2025.csv", index=False)
relievers.to_csv("reliever_cPV_ratings_2025.csv", index=False)
ars.to_csv("pitcher_arsenal_ratings_2025.csv", index=False)

print("Exported 2025 leaderboards successfully.")
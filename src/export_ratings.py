import time
import joblib
import numpy as np
import pandas as pd
from pybaseball import playerid_reverse_lookup

t_start = time.time()

# load raw full season set + booster
df = pd.read_csv("cleaned_pitches_2025.csv")
m = joblib.load("xrv_quality_model_2025.joblib")

# pitcher pov horizontal movement
is_rhp = (df['p_throws'] == 'R').astype(int)
is_rhb = (df['stand'] == 'R').astype(int)
platoon = (is_rhp == is_rhb).astype(int)
hb_p = np.where(is_rhp == 1, -df['pfx_x'], df['pfx_x'])

# zone edge dist
dx = np.maximum(0, np.abs(df['plate_x']) - 0.708)
dz = np.maximum(0, np.abs(df['plate_z'] - 2.5) - 0.85)
d_edge = np.sqrt(dx*dx + dz*dz)

feats = pd.DataFrame({
    'b': df['balls'],
    's': df['strikes'],
    'platoon': platoon,
    'rhp': is_rhp,
    'velo': df['release_speed'],
    'spin': df['release_spin_rate'],
    'ext': df['release_extension'],
    'hb': hb_p,
    'ivb': df['pfx_z'],
    'px': df['plate_x'],
    'pz': df['plate_z'],
    'd_edge': d_edge
})

# invert: positive xrv = run prevention
df['xrv'] = -m.predict(feats)

# identify starts using pitch volume per game_date
# outings with >= 50 pitches count as a start
game_counts = df.groupby(['pitcher', 'game_date']).size().reset_index(name='game_pitches')
starts = game_counts[game_counts['game_pitches'] >= 50].groupby('pitcher').size().rename('gs')

# group per pitcher
grp = df.groupby('pitcher').agg(
    n_pitches=('xrv', 'count'),
    tot_xrv=('xrv', 'sum'),
    velo=('release_speed', 'mean'),
    whiff=('is_whiff', 'mean'),
    zone=('in_zone', 'mean')
).reset_index()

grp = grp.merge(starts, on='pitcher', how='left').fillna({'gs': 0})
grp['gs'] = grp['gs'].astype(int)

# 400 pitch min across league for scaling
MIN_P = 400
qual = grp[grp['n_pitches'] >= MIN_P].copy()
qual['cpv_per_100'] = (qual['tot_xrv'] / qual['n_pitches']) * 100.0

# 100 index scale
mu = qual['cpv_per_100'].mean()
sd = qual['cpv_per_100'].std()
qual['cPV_plus'] = np.round(100.0 + ((qual['cpv_per_100'] - mu) / sd) * 15.0, 1)

# mlbam id lookup
p_ids = qual['pitcher'].unique().tolist()
lut = playerid_reverse_lookup(p_ids, key_type='mlbam')
lut['full_name'] = lut['name_first'].str.capitalize() + ' ' + lut['name_last'].str.capitalize()
id2name = dict(zip(lut['key_mlbam'], lut['full_name']))

qual['player_name'] = qual['pitcher'].map(id2name).fillna("Unknown")

cols = ['player_name', 'pitcher', 'gs', 'n_pitches', 'velo', 'whiff', 'zone', 'tot_xrv', 'cPV_plus']
qual = qual.sort_values('cPV_plus', ascending=False)[cols]

# split starters and relievers
starters = qual[(qual['gs'] >= 5) & (qual['n_pitches'] >= 1000)].copy()
relievers = qual[(qual['gs'] < 5) & (qual['n_pitches'] >= 400)].copy()

# repertoire splits
sub = df[df['pitcher'].isin(qual['pitcher'])].copy()
ars = sub.groupby(['pitcher', 'pitch_type']).agg(
    n_pitches=('xrv', 'count'),
    tot_xrv=('xrv', 'sum'),
    velo=('release_speed', 'mean'),
    hb=('pfx_x', 'mean'),
    ivb=('pfx_z', 'mean'),
    whiff=('is_whiff', 'mean'),
    zone=('in_zone', 'mean')
).reset_index()

ars['cpv_per_100'] = (ars['tot_xrv'] / ars['n_pitches']) * 100.0
ars['pitch_cPV_plus'] = np.round(100.0 + ((ars['cpv_per_100'] - mu) / sd) * 15.0, 1)
ars['player_name'] = ars['pitcher'].map(id2name).fillna("Unknown")

# exports
qual.to_csv("pitcher_cPV_ratings_2025.csv", index=False)
starters.to_csv("starter_cPV_ratings_2025.csv", index=False)
relievers.to_csv("reliever_cPV_ratings_2025.csv", index=False)
ars.to_csv("pitcher_arsenal_ratings_2025.csv", index=False)

print("done. starters: %d | relievers: %d | mu: %.3f, sd: %.3f | time: %.1fs" % (
    len(starters), len(relievers), mu, sd, (time.time() - t_start)
))
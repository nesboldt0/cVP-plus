import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

t0 = time.time()

# 2025 statcast set & serialized booster
df = pd.read_csv("cleaned_pitches_2025.csv")
mdl = joblib.load("xrv_quality_model_2025.joblib")

# align physics / perspective transforms
rhp = (df['p_throws'] == 'R').astype(int)
rhb = (df['stand'] == 'R').astype(int)
platoon = (rhp == rhb).astype(int)

# pitcher POV break
hb_p = []
for i in range(len(df)):
    if rhp.iloc[i] == 1:
        hb_p.append(-df['pfx_x'].iloc[i])
    else:
        hb_p.append(df['pfx_x'].iloc[i])
hb_p = pd.Series(hb_p)

# zone border dist
x_dist = np.abs(df['plate_x']) - 0.708
dx = np.maximum(0, x_dist)

z_dist = np.abs(df['plate_z'] - 2.5) - 0.85
dz = np.maximum(0, z_dist)

d_edge = np.sqrt(dx * dx + dz * dz)

feat_cols = pd.DataFrame()
feat_cols['b'] = df['balls']
feat_cols['s'] = df['strikes']
feat_cols['platoon'] = platoon
feat_cols['rhp'] = rhp
feat_cols['velo'] = df['release_speed']
feat_cols['spin'] = df['release_spin_rate']
feat_cols['ext'] = df['release_extension']
feat_cols['hb'] = hb_p.values
feat_cols['ivb'] = df['pfx_z']
feat_cols['px'] = df['plate_x']
feat_cols['pz'] = df['plate_z']
feat_cols['d_edge'] = d_edge

# invert: positive xrv = run prevention
preds = mdl.predict(feat_cols)
df['xrv'] = -preds

# qualify pitchers (min 750 pitches)
min_pitches = 750
pitch_counts = df['pitcher'].value_counts()
qualified_pitchers = []

for pitcher_id, count in pitch_counts.items():
    if count >= min_pitches:
        qualified_pitchers.append(pitcher_id)

df_qual = df[df['pitcher'].isin(qualified_pitchers)].copy()

# odd/even split
df_qual['p_seq'] = df_qual.groupby('pitcher').cumcount()
odd_half = df_qual[df_qual['p_seq'] % 2 == 1]
even_half = df_qual[df_qual['p_seq'] % 2 == 0]

# group odd
o_grp = odd_half.groupby('pitcher').agg({
    'xrv': ['count', 'sum'],
    'is_whiff': 'mean',
    'delta_run_exp': 'sum'
}).reset_index()

o_grp.columns = ['pitcher', 'n', 'xrv', 'whiff', 're']
o_grp['odd_cpv'] = (o_grp['xrv'] / o_grp['n']) * 100.0
o_grp['odd_re'] = (o_grp['re'] / o_grp['n']) * 100.0
o_grp['odd_whiff'] = o_grp['whiff']

# group even
e_grp = even_half.groupby('pitcher').agg({
    'xrv': ['count', 'sum'],
    'is_whiff': 'mean',
    'delta_run_exp': 'sum'
}).reset_index()

e_grp.columns = ['pitcher', 'n', 'xrv', 'whiff', 're']
e_grp['even_cpv'] = (e_grp['xrv'] / e_grp['n']) * 100.0
e_grp['even_re'] = (e_grp['re'] / e_grp['n']) * 100.0
e_grp['even_whiff'] = e_grp['whiff']

splits = pd.merge(o_grp, e_grp, on='pitcher')

# compute r & spearman brown: 2r / (1 + r)
eval_metrics = [
    ('Raw RE', 'odd_re', 'even_re'),
    ('Whiff Rate', 'odd_whiff', 'even_whiff'),
    ('cPV+ (xRV)', 'odd_cpv', 'even_cpv')
]
# the reason why this file is even included in the first place is this is testing to see how "good" of a metric cPV+ is
# it tests it against whiff rate and raw run value. Whiff rate is a great metric but it doesn't actually tell you how many runs are prevented and raw run rate
# has its own issues. cPV+ aims to be as reliable as reliable as whiff rate but measure run prevention instead of how often the batter missed
print("N pitchers:", len(splits), f"(min {min_pitches} pitches)")
print("-" * 52)
for name, c1, c2 in eval_metrics:
    r, p = pearsonr(splits[c1], splits[c2])
    if (1.0 + r) != 0:
        sb = (2.0 * r) / (1.0 + r)
    else:
        sb = 0
    print(f"{name:<18} | r = {r:6.3f} (p={p:.1e}) | SB = {sb:6.3f}")
print("-" * 52)

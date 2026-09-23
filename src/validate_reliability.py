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
hb_p = np.where(rhp == 1, -df['pfx_x'], df['pfx_x'])

# zone border dist
dx = np.maximum(0, np.abs(df['plate_x']) - 0.708)
dz = np.maximum(0, np.abs(df['plate_z'] - 2.5) - 0.85)
d_edge = np.sqrt(dx**2 + dz**2)

feat_cols = pd.DataFrame({
    'b': df['balls'],
    's': df['strikes'],
    'platoon': platoon,
    'rhp': rhp,
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
df['xrv'] = -mdl.predict(feat_cols)

# qualify pitchers (min 750 pitches)
min_pitches = 750
qual_p = df['pitcher'].value_counts()
df_qual = df[df['pitcher'].isin(qual_p[qual_p >= min_pitches].index)].copy()

# odd/even split
df_qual['p_seq'] = df_qual.groupby('pitcher').cumcount()
odd_half = df_qual[df_qual['p_seq'] % 2 == 1]
even_half = df_qual[df_qual['p_seq'] % 2 == 0]

# group odd
o_grp = odd_half.groupby('pitcher').agg(
    n=('xrv', 'count'),
    xrv=('xrv', 'sum'),
    whiff=('is_whiff', 'mean'),
    re=('delta_run_exp', 'sum')
).reset_index()
o_grp['odd_cpv'] = (o_grp['xrv'] / o_grp['n']) * 100.0
o_grp['odd_re'] = (o_grp['re'] / o_grp['n']) * 100.0
o_grp['odd_whiff'] = o_grp['whiff']

# group even
e_grp = even_half.groupby('pitcher').agg(
    n=('xrv', 'count'),
    xrv=('xrv', 'sum'),
    whiff=('is_whiff', 'mean'),
    re=('delta_run_exp', 'sum')
).reset_index()
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

print("N pitchers: %d (min %d pitches)" % (len(splits), min_pitches))
print("-" * 52)
for name, c1, c2 in eval_metrics:
    r, p = pearsonr(splits[c1], splits[c2])
    sb = (2.0 * r) / (1.0 + r) if (1.0 + r) != 0 else 0
    print("%-18s | r = %6.3f (p=%.1e) | SB = %6.3f" % (name, r, p, sb))
print("-" * 52)
print("elapsed: %.2fs" % (time.time() - t0))
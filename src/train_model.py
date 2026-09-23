import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

t0 = time.time()

# load processed season data
raw = pd.read_csv("cleaned_pitches_2025.csv")

# flip hb to pitcher perspective and calc zone edge distance
rhp = (raw['p_throws'] == 'R').astype(int)
rhb = (raw['stand'] == 'R').astype(int)
platoon = (rhp == rhb).astype(int)

# pitcher POV horizontal break
hb_p = np.where(rhp == 1, -raw['pfx_x'], raw['pfx_x'])

# dist from zone border (box approx 17in wide, knees to chest)
dx = np.maximum(0, np.abs(raw['plate_x']) - 0.708)
dz = np.maximum(0, np.abs(raw['plate_z'] - 2.5) - 0.85)
d_edge = np.sqrt(dx**2 + dz**2)

feat_cols = ['b', 's', 'platoon', 'rhp', 'velo', 'spin', 'ext', 'hb', 'ivb', 'px', 'pz', 'd_edge']

X = pd.DataFrame({
    'b': raw['balls'],
    's': raw['strikes'],
    'platoon': platoon,
    'rhp': rhp,
    'velo': raw['release_speed'],
    'spin': raw['release_spin_rate'],
    'ext': raw['release_extension'],
    'hb': hb_p,
    'ivb': raw['pfx_z'],
    'px': raw['plate_x'],
    'pz': raw['plate_z'],
    'd_edge': d_edge
})

y = raw['delta_run_exp']

# drop any remaining nan records
valid = X.notna().all(axis=1) & y.notna()
X = X[valid].reset_index(drop=True)
y = y[valid].reset_index(drop=True)

# 85/15 validation split
np.random.seed(42)
idx = np.random.permutation(len(X))
split_pt = int(len(X) * 0.85)
tr_idx, val_idx = idx[:split_pt], idx[split_pt:]

X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

# fit booster
mdl = HistGradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.07,
    max_iter=150,
    max_leaf_nodes=42,
    min_samples_leaf=75,
    l2_regularization=0.3,
    random_state=42
)

mdl.fit(X_train, y_train)

val_preds = mdl.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print("n_pitches: %d | val_n: %d | val_rmse: %.4f | elapsed: %.1fs" % (
    len(X), len(X_val), rmse, time.time() - t0
))

# save model artifact
joblib.dump(mdl, "xrv_quality_model_2025.joblib")
print("saved -> xrv_quality_model_2025.joblib")
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
hb_p = []
for i in range(len(raw)):
    if rhp.iloc[i] == 1:
        hb_p.append(-raw['pfx_x'].iloc[i])
    else:
        hb_p.append(raw['pfx_x'].iloc[i])
hb_p = pd.Series(hb_p)

# dist from zone border (box approx 17in wide, knees to chest)
x_dist = np.abs(raw['plate_x']) - 0.708
dx = np.maximum(0, x_dist)

z_dist = np.abs(raw['plate_z'] - 2.5) - 0.85
dz = np.maximum(0, z_dist)

d_edge = np.sqrt(dx**2 + dz**2)

feat_cols = ['b', 's', 'platoon', 'rhp', 'velo', 'spin', 'ext', 'hb', 'ivb', 'px', 'pz', 'd_edge']

X = pd.DataFrame()
X['b'] = raw['balls']
X['s'] = raw['strikes']
X['platoon'] = platoon
X['rhp'] = rhp
X['velo'] = raw['release_speed']
X['spin'] = raw['release_spin_rate']
X['ext'] = raw['release_extension']
X['hb'] = hb_p.values
X['ivb'] = raw['pfx_z']
X['px'] = raw['plate_x']
X['pz'] = raw['plate_z']
X['d_edge'] = d_edge

y = raw['delta_run_exp']

# drop any remaining nan records
X['target_y'] = y
X = X.dropna()

y = X['target_y']
X = X[feat_cols]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# 85/15 validation split
np.random.seed(42)
total_rows = len(X)
idx = np.random.permutation(total_rows)

split_pt = int(total_rows * 0.85)
tr_idx = idx[:split_pt]
val_idx = idx[split_pt:]

X_train = X.iloc[tr_idx]
y_train = y.iloc[tr_idx]

X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

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

print(f"Validation RMSE: {rmse:.4f}")

# save model artifact
joblib.dump(mdl, "xrv_quality_model_2025.joblib")
print("Saved model -> xrv_quality_model_2025.joblib")
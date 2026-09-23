import os
import glob
import time
import requests
import pandas as pd
from datetime import datetime
from pybaseball import statcast

# Full season regular season bounds (omits spring training & postseason)
SEASON_START = "2025-03-27"
SEASON_END = "2025-09-28"
RAW_DIR = "raw_chunks_2025"
OUT_PATH = "cleaned_pitches_2025.csv"

# Statcast pitch flags we actually care about for xRV modeling
# Drops pitchouts, intentional walks, and unknown tracking noise
DISCARD_PITCH_TYPES = {"PO", "IN", "EP", "UN", "KN"}

# Whiffs: swinging strikes, fouls tipped into glove with 2 strikes / missed bunts
WHIFF_EVENTS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt"
}


def pull_season_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Generate 6-day intervals to stay under the Statcast ~25k-40k row query ceiling
    dates = pd.date_range(SEASON_START, SEASON_END, freq="7D")
    intervals = []
    for i in range(len(dates)):
        w_start = dates[i].strftime("%Y-%m-%d")
        w_end = (dates[i] + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        if w_end > SEASON_END:
            w_end = SEASON_END
        intervals.append((w_start, w_end))

    print(f"Ingesting {len(intervals)} weekly blocks from Statcast...")
    
    for start_dt, end_dt in intervals:
        chunk_file = os.path.join(RAW_DIR, f"sc_{start_dt}_{end_dt}.csv")
        if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1024:
            continue

        print(f"Fetching: {start_dt} to {end_dt}")
        attempts = 0
        success = False
        while attempts < 3 and not success:
            try:
                # pybaseball wraps baseball savant's search endpoint
                df_chunk = statcast(start_dt=start_dt, end_dt=end_dt)
                
                # Check for All-Star break / empty windows
                if df_chunk is None or df_chunk.empty:
                    print(f"  Empty window returned ({start_dt} to {end_dt}) - likely ASG break")
                    # Touch empty file so we don't re-pull on restart
                    with open(chunk_file, 'w') as f:
                        f.write("")
                    success = True
                    break
                    
                df_chunk.to_csv(chunk_file, index=False)
                print(f"  Saved {len(df_chunk)} pitches")
                success = True
                # Brief sleep to avoid hitting Savant rate limits
                time.sleep(1.5)
            except (requests.exceptions.RequestException, Exception) as e:
                attempts += 1
                wait = attempts * 5
                print(f"  Request failed ({e}). Retrying in {wait}s... [{attempts}/3]")
                time.sleep(wait)


def process_and_combine():
    csvs = glob.glob(os.path.join(RAW_DIR, "sc_*.csv"))
    if not csvs:
        raise SystemExit("No downloaded chunk files found in directory.")

    chunks = []
    for f in csvs:
        if os.path.getsize(f) == 0:
            continue
        # Only parse necessary columns to conserve RAM during concat
        cols = [
            'pitcher', 'player_name', 'pitch_type', 'game_date',
            'release_speed', 'release_spin_rate', 'release_extension',
            'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
            'balls', 'strikes', 'stand', 'p_throws',
            'delta_run_exp', 'description', 'type'
        ]
        try:
            chunks.append(pd.read_csv(f, usecols=cols, low_memory=False))
        except Exception:
            # Fallback for chunks with column schema mismatches
            raw = pd.read_csv(f, low_memory=False)
            valid = [c for c in cols if c in raw.columns]
            chunks.append(raw[valid])

    df = pd.concat(chunks, ignore_index=True)
    n_raw = len(df)
    print(f"Raw consolidated pitch count: {n_raw:,}")

    # Data hygiene filters
    df = df[~df['pitch_type'].isin(DISCARD_PITCH_TYPES)]
    
    # Must have non-null tracking metrics and ground-truth run values
    vital_cols = [
        'release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'delta_run_exp', 'balls', 'strikes', 'stand', 'p_throws'
    ]
    df = df.dropna(subset=vital_cols)

    # Impute missing Hawk-Eye extension and spin using pitch-type medians rather than flat league mean
    # (e.g. curveballs spin way faster than changeups; flat median distorts physics)
    spin_medians = df.groupby('pitch_type')['release_spin_rate'].transform('median')
    df['release_spin_rate'] = df['release_spin_rate'].fillna(spin_medians).fillna(df['release_spin_rate'].median())
    
    ext_medians = df.groupby('pitch_type')['release_extension'].transform('median')
    df['release_extension'] = df['release_extension'].fillna(ext_medians).fillna(df['release_extension'].median())

    # Sabermetric binary targets
    df['is_whiff'] = df['description'].isin(WHIFF_EVENTS).astype(int)
    df['in_zone'] = df['type'].isin(['S', 'X']).astype(int)

    # Quick summary of drop rate
    print(f"Clean records remaining: {len(df):,} ({len(df)/n_raw:.1%} retained)")
    
    df.to_csv(OUT_PATH, index=False)
    print(f"Exported training table -> {OUT_PATH}")


if __name__ == "__main__":
    pull_season_data()
    process_and_combine()
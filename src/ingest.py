import os
import glob
import time
import requests
import pandas as pd
from datetime import datetime
from pybaseball import statcast


SEASON_START = "2025-03-27"
SEASON_END = "2025-09-28"
RAW_DIR = "raw_chunks_2025"
OUT_PATH = "cleaned_pitches_2025.csv"

# Filters out metrics that are not necessary
DISCARD_PITCH_TYPES = {"PO", "IN", "EP", "UN", "KN"}

# Whiffs: swinging strikes, fouls tipped into glove with 2 strikes / missed bunts
WHIFF_EVENTS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt"
}


def pull_season_data():
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    
    # Pull season data in intervals
    dates = pd.date_range(SEASON_START, SEASON_END, freq="7D")
    intervals = []
    
    for dt in dates:
        w_start = dt.strftime("%Y-%m-%d")
        end_date = dt + pd.Timedelta(days=6)
        w_end = end_date.strftime("%Y-%m-%d")
        
        if w_end > SEASON_END:
            w_end = SEASON_END
            
        intervals.append((w_start, w_end))

    for start_dt, end_dt in intervals:
        filename = "sc_" + start_dt + "_" + end_dt + ".csv"
        chunk_file = os.path.join(RAW_DIR, filename)
        
        if os.path.exists(chunk_file):
            if os.path.getsize(chunk_file) > 1024:
                continue

        attempts = 0
        success = False
        
        while attempts < 3 and not success:
            try:
                # pybaseball wraps baseball savant's search endpoint
                df_chunk = statcast(start_dt=start_dt, end_dt=end_dt)
                
                # Check for All-Star break / empty windows
                if df_chunk is None or len(df_chunk) == 0:
                    # Touch empty file so we don't re-pull on restart
                    f = open(chunk_file, "w")
                    f.write("")
                    f.close()
                    success = True
                    break
                    
                df_chunk.to_csv(chunk_file, index=False)
                success = True
                
                # Brief sleep to avoid hitting Savant rate limits
                time.sleep(1.5)
                
            except (requests.exceptions.RequestException, Exception):
                attempts = attempts + 1
                wait = attempts * 5
                time.sleep(wait)


def process_and_combine():
    search_pattern = os.path.join(RAW_DIR, "sc_*.csv")
    csvs = glob.glob(search_pattern)
    
    if len(csvs) == 0:
        raise SystemExit("No downloaded chunk files found in directory.")

    chunks = []
    cols = [
        'pitcher', 'player_name', 'pitch_type', 'game_date',
        'release_speed', 'release_spin_rate', 'release_extension',
        'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'balls', 'strikes', 'stand', 'p_throws',
        'delta_run_exp', 'description', 'type'
    ]

    for f in csvs:
        if os.path.getsize(f) == 0:
            continue
            
        # Only parse necessary columns to conserve RAM during concat
        try:
            chunk = pd.read_csv(f, usecols=cols, low_memory=False)
            chunks.append(chunk)
        except Exception:
            # Fallback for chunks with column schema mismatches
            raw = pd.read_csv(f, low_memory=False)
            valid = []
            for c in cols:
                if c in raw.columns:
                    valid.append(c)
            chunks.append(raw[valid])

    df = pd.concat(chunks, ignore_index=True)

    # Data hygiene filters
    df = df[~df['pitch_type'].isin(DISCARD_PITCH_TYPES)]
    
    # Must have non-null tracking metrics and ground-truth run values
    vital_cols = [
        'release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'delta_run_exp', 'balls', 'strikes', 'stand', 'p_throws'
    ]
    df = df.dropna(subset=vital_cols)

    # (e.g. curveballs spin way faster than changeups; flat median distorts physics)
    # calculate median by pitch type in a loop rather than transform
    for p_type in df['pitch_type'].unique():
        type_mask = df['pitch_type'] == p_type
        
        spin_med = df.loc[type_mask, 'release_spin_rate'].median()
        df.loc[type_mask, 'release_spin_rate'] = df.loc[type_mask, 'release_spin_rate'].fillna(spin_med)
        
        ext_med = df.loc[type_mask, 'release_extension'].median()
        df.loc[type_mask, 'release_extension'] = df.loc[type_mask, 'release_extension'].fillna(ext_med)

    overall_spin = df['release_spin_rate'].median()
    df['release_spin_rate'] = df['release_spin_rate'].fillna(overall_spin)
    
    overall_ext = df['release_extension'].median()
    df['release_extension'] = df['release_extension'].fillna(overall_ext)

    # Sabermetric binary targets
    df['is_whiff'] = df['description'].isin(WHIFF_EVENTS).astype(int)
    df['in_zone'] = df['type'].isin(['S', 'X']).astype(int)

    # Quick summary of drop rate
    df.to_csv(OUT_PATH, index=False)
    print(f"Exported training table -> {OUT_PATH}")


if __name__ == "__main__":
    pull_season_data()
    process_and_combine()
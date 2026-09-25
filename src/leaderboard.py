import argparse
import pandas as pd


PITCHERS_FILE = "pitcher_cPV_ratings_2025.csv"
ARSENAL_FILE = "pitcher_arsenal_ratings_2025.csv"

pitchers = pd.read_csv(PITCHERS_FILE)
arsenal = pd.read_csv(ARSENAL_FILE)

# this will allow us to sort through all the qualified pitchers (theres a lot) and sort them by their cPV+
def show_leaderboard(min_pitches=750, top_n=15, sort_by="cPV_plus", ascending=False):
    """Print the league leaderboard ordered by the specified metric."""
    if sort_by not in pitchers.columns:
        print(f"Column '{sort_by}' not found. Defaulting to 'cPV_plus'.")
        sort_by = "cPV_plus"

    sub = pitchers[pitchers["n_pitches"] >= min_pitches]
    sub = sub.sort_values(by=sort_by, ascending=ascending)
    sub = sub.head(top_n)

    cols = [
        "player_name",
        "n_pitches",
        "velo",
        "whiff",
        "zone",
        "tot_xrv",
        "cPV_plus",
    ]

    # Format percentages and floats for clean console reading
    display_df = sub[cols].copy()
    
    formatted_whiff = []
    for val in display_df["whiff"]:
        formatted_whiff.append(f"{val:.1%}")
    display_df["whiff"] = formatted_whiff

    formatted_zone = []
    for val in display_df["zone"]:
        formatted_zone.append(f"{val:.1%}")
    display_df["zone"] = formatted_zone

    formatted_velo = []
    for val in display_df["velo"]:
        formatted_velo.append(f"{val:.1f}")
    display_df["velo"] = formatted_velo

    formatted_xrv = []
    for val in display_df["tot_xrv"]:
        formatted_xrv.append(f"{val:+.2f}")
    display_df["tot_xrv"] = formatted_xrv

    # Rename display columns for clean console header presentation
    display_df.columns = ["Player", "Pitches", "Velo", "Whiff%", "Zone%", "Net xRV", "cPV+"]

    print(f"\n{'='*78}")
    print(
        f"  2025 MLB cPV+ LEADERBOARD (Min {min_pitches} Pitches | Sorted by {sort_by})"
    )
    print(f"{'='*78}")
    print(display_df.to_string(index=False))
    print(f"{'='*78}\n")

# this is just to take all the data we already have and put it on a readable "scout card"
# this makes looking up pitchers stats actually bearable to look at
# if you don't beleive me, go look at the raw CSV files and try and read them
def get_scout_card(player_name):
    """Print a detailed player summary and pitch-level repertoire card."""
    matches = pitchers[
        pitchers["player_name"].str.contains(player_name, case=False, na=False)
    ]
    if len(matches) == 0:
        print(f"\nNo qualified pitcher found matching '{player_name}'.")
        return

    p = matches.iloc[0]
    p_id = p["pitcher"]
    
    p_arsenal = arsenal[arsenal["pitcher"] == p_id]
    p_arsenal = p_arsenal.sort_values(by="n_pitches", ascending=False)
    
    total_p = p_arsenal["n_pitches"].sum()

    print("\n" + "=" * 70)
    print(
        f" {p['player_name'].upper()} | cPV+: {p['cPV_plus']:.1f} | Net Runs Saved (xRV): {p['tot_xrv']:+.2f}"
    )
    print(
        f" 2025 Pitches: {int(p['n_pitches']):,} | Mean Velo: {p['velo']:.1f} mph | Whiff%: {p['whiff']:.1%} | Zone%: {p['zone']:.1%}"
    )
    print("=" * 70)
    print(
        f"{'Pitch':<6} | {'Usage%':<7} | {'Velo':<6} | {'H-Break':<8} | {'V-Break':<8} | {'Whiff%':<7} | {'cPV+'}"
    )
    print("-" * 70)

    for i in range(len(p_arsenal)):
        r = p_arsenal.iloc[i]
        usage = (r["n_pitches"] / total_p) * 100
        print(
            f"{r['pitch_type']:<6} | {usage:>5.1f}% | {r['velo']:>5.1f} | "
            f"{r['hb']:>7.2f}\" | {r['ivb']:>7.2f}\" | "
            f"{r['whiff']:>6.1%} | {r['pitch_cPV_plus']:>6.1f}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query 2025 cPV+ Pitching Metrics"
    )
    parser.add_argument("-p", "--player", type=str, help="Pitcher name to lookup")
    parser.add_argument(
        "-t", "--top", type=int, default=15, help="Number of top rows to display"
    )
    parser.add_argument(
        "-m",
        "--min",
        type=int,
        default=750,
        help="Minimum pitch volume cutoff (default: 750)",
    )
    parser.add_argument(
        "-s",
        "--sort",
        type=str,
        default="cPV_plus",
        help="Column to sort by (default: cPV_plus)",
    )
    parser.add_argument(
        "--bottom",
        action="store_true",
        help="Show bottom of the leaderboard instead of top",
    )
    args = parser.parse_args()
    # If we want to get a player, get the player, otehrwise we can just print the leaderboard
    if args.player:
        get_scout_card(args.player)
    else:
        show_leaderboard(
            min_pitches=args.min,
            top_n=args.top,
            sort_by=args.sort,
            ascending=args.bottom,
        )

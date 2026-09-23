import argparse
import pandas as pd

# Load full-season 2025 tables
PITCHERS_FILE = "pitcher_cPV_ratings_2025.csv"
ARSENAL_FILE = "pitcher_arsenal_ratings_2025.csv"

pitchers = pd.read_csv(PITCHERS_FILE)
arsenal = pd.read_csv(ARSENAL_FILE)


def show_leaderboard(min_pitches=750, top_n=15, sort_by="cPV_plus", ascending=False):
    """Print the league leaderboard ordered by the specified metric."""
    if sort_by not in pitchers.columns:
        print(f"Column '{sort_by}' not found. Defaulting to 'cPV_plus'.")
        sort_by = "cPV_plus"

    sub = (
        pitchers[pitchers["pitches"] >= min_pitches]
        .sort_values(by=sort_by, ascending=ascending)
        .head(top_n)
    )

    cols = [
        "player_name",
        "pitches",
        "avg_velo",
        "whiff_rate",
        "zone_rate",
        "total_xrv",
        "cPV_plus",
    ]

    # Format percentages and floats for clean console reading
    display_df = sub[cols].copy()
    display_df["whiff_rate"] = display_df["whiff_rate"].map("{:.1%}".format)
    display_df["zone_rate"] = display_df["zone_rate"].map("{:.1%}".format)
    display_df["avg_velo"] = display_df["avg_velo"].map("{:.1f}".format)
    display_df["total_xrv"] = display_df["total_xrv"].map("{:+.2f}".format)

    print(f"\n{'='*78}")
    print(
        f"  2025 MLB cPV+ LEADERBOARD (Min {min_pitches} Pitches | Sorted by {sort_by})"
    )
    print(f"{'='*78}")
    print(display_df.to_string(index=False))
    print(f"{'='*78}\n")


def get_scout_card(player_name):
    """Print a detailed player summary and pitch-level repertoire card."""
    matches = pitchers[
        pitchers["player_name"].str.contains(player_name, case=False, na=False)
    ]
    if matches.empty:
        print(f"\nNo qualified pitcher found matching '{player_name}'.")
        return

    p = matches.iloc[0]
    p_id = p["pitcher"]
    p_arsenal = arsenal[arsenal["pitcher"] == p_id].sort_values(
        by="pitches", ascending=False
    )
    total_p = p_arsenal["pitches"].sum()

    print("\n" + "=" * 70)
    print(
        f" {p['player_name'].upper()} | cPV+: {p['cPV_plus']:.1f} | Net Runs Saved (xRV): {p['total_xrv']:+.2f}"
    )
    print(
        f" 2025 Pitches: {int(p['pitches']):,} | Mean Velo: {p['avg_velo']:.1f} mph | Whiff%: {p['whiff_rate']:.1%} | Zone%: {p['zone_rate']:.1%}"
    )
    print("=" * 70)
    print(
        f"{'Pitch':<6} | {'Usage%':<7} | {'Velo':<6} | {'H-Break':<8} | {'V-Break':<8} | {'Whiff%':<7} | {'cPV+'}"
    )
    print("-" * 70)

    for _, r in p_arsenal.iterrows():
        usage = (r["pitches"] / total_p) * 100
        print(
            f"{r['pitch_type']:<6} | {usage:>5.1f}% | {r['avg_velo']:>5.1f} | "
            f"{r['avg_pfx_x']:>7.2f}\" | {r['avg_pfx_z']:>7.2f}\" | "
            f"{r['whiff_rate']:>6.1%} | {r['pitch_cPV_plus']:>6.1f}"
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

    if args.player:
        get_scout_card(args.player)
    else:
        show_leaderboard(
            min_pitches=args.min,
            top_n=args.top,
            sort_by=args.sort,
            ascending=args.bottom,
        )
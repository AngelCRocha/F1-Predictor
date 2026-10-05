"""
Joins qualifying + race CSVs and builds per-driver-per-race features.
Outputs data/features.csv ready for model.py.

Usage:
    python feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_finish_position(val) -> float:
    """
    ClassifiedPosition can be an int, "R" (retired), "D" (DSQ), etc.
    Map non-finishes to 20 (last) so rolling averages degrade on DNFs.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return 20.0


def best_q_time(row) -> float | None:
    """Return the fastest of a driver's Q1/Q2/Q3 times in seconds, or None if all are missing."""
    times = [row.get("Q1_s"), row.get("Q2_s"), row.get("Q3_s")]
    valid = [t for t in times if t is not None and not np.isnan(t)]
    return min(valid) if valid else None


# ---------------------------------------------------------------------------
# Rolling / historical features
# ---------------------------------------------------------------------------

def add_driver_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised rolling finish position for each driver."""
    df = df.sort_values(["Abbreviation", "year", "round"]).copy()

    def _rolling(grp):
        s = grp["finish_position"]
        return pd.DataFrame({
            "rolling_finish_3": s.shift(1).rolling(3, min_periods=1).mean(),
            "rolling_finish_5": s.shift(1).rolling(5, min_periods=1).mean(),
        }, index=grp.index)

    result = df.groupby("Abbreviation", group_keys=False).apply(_rolling, include_groups=False)
    df[["rolling_finish_3", "rolling_finish_5"]] = result
    return df


def add_constructor_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 5-race average finish for the constructor (both drivers combined)."""
    df = df.sort_values(["TeamName", "year", "round"]).copy()

    team_race_avg = (
        df.groupby(["TeamName", "year", "round"])["finish_position"]
        .mean()
        .reset_index()
        .rename(columns={"finish_position": "team_avg_finish"})
        .sort_values(["TeamName", "year", "round"])
    )

    def _roll(grp):
        return pd.DataFrame({
            "constructor_rolling_5": grp["team_avg_finish"].shift(1).rolling(5, min_periods=1).mean(),
        }, index=grp.index)

    result = team_race_avg.groupby("TeamName", group_keys=False).apply(_roll, include_groups=False)
    team_race_avg["constructor_rolling_5"] = result["constructor_rolling_5"]
    team_race_avg.drop(columns=["team_avg_finish"], inplace=True)

    df = df.merge(team_race_avg, on=["TeamName", "year", "round"], how="left")
    return df


def add_circuit_historical(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (driver, circuit), compute the mean finish position from ALL
    prior years at that event. Uses EventName as the circuit key.
    """
    df = df.sort_values(["Abbreviation", "EventName", "year"]).copy()

    records = []
    for _, grp in df.groupby(["Abbreviation", "EventName"]):
        grp = grp.sort_values("year").copy()
        grp["circuit_historical_avg"] = (
            grp["finish_position"].expanding().mean().shift(1)
        )
        records.append(grp)

    result = pd.concat(records).sort_values(["year", "round", "Abbreviation"])
    return result


def add_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    """DNF rate within the current season up to (but not including) this race."""
    df = df.sort_values(["Abbreviation", "year", "round"]).copy()

    df["is_dnf"] = (~df["Status"].astype(str).str.startswith("Finished")) & \
                   (~df["Status"].astype(str).str.startswith("+"))

    def _dnf(grp):
        cum_dnf   = grp.groupby("year")["is_dnf"].cumsum().shift(1).fillna(0)
        cum_races = grp.groupby("year").cumcount().shift(1).fillna(0) + 1
        return pd.DataFrame({
            "dnf_rate_season": (cum_dnf / cum_races).fillna(0),
        }, index=grp.index)

    result = df.groupby("Abbreviation", group_keys=False).apply(_dnf, include_groups=False)
    df["dnf_rate_season"] = result["dnf_rate_season"]
    df.drop(columns=["is_dnf"], inplace=True)
    return df


def add_season_points(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative points earned in the season BEFORE this race."""
    df = df.sort_values(["Abbreviation", "year", "round"]).copy()

    def _pts(grp):
        return pd.DataFrame({
            "season_points_cumulative": grp.groupby("year")["Points"].cumsum().shift(1).fillna(0),
        }, index=grp.index)

    result = df.groupby("Abbreviation", group_keys=False).apply(_pts, include_groups=False)
    df["season_points_cumulative"] = result["season_points_cumulative"]
    return df


# ---------------------------------------------------------------------------
# New features: FP2 pace, teammate normalization, regulation era
# ---------------------------------------------------------------------------

def add_fp2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge FP2 long-run pace and compute gap to the fastest pace that session."""
    fp2_path = DATA_DIR / "fp2_raw.csv"
    if not fp2_path.exists():
        df["fp2_avg_pace_s"] = np.nan
        df["fp2_pace_gap_to_best_s"] = np.nan
        return df

    fp2 = pd.read_csv(fp2_path)
    df = df.merge(
        fp2[["Abbreviation", "year", "round", "fp2_avg_pace_s"]],
        on=["Abbreviation", "year", "round"],
        how="left",
    )

    best_fp2 = (
        df.groupby(["year", "round"])["fp2_avg_pace_s"]
        .min()
        .reset_index()
        .rename(columns={"fp2_avg_pace_s": "_fp2_best"})
    )
    df = df.merge(best_fp2, on=["year", "round"], how="left")
    df["fp2_pace_gap_to_best_s"] = df["fp2_avg_pace_s"] - df["_fp2_best"]
    df.drop(columns=["_fp2_best"], inplace=True)
    return df


def add_teammate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each race, compare each driver to their teammate:
      - q_delta_vs_teammate_s: own qualifying time minus teammate's (positive = slower)
      - teammate_rolling_finish_diff: own rolling_finish_5 minus teammate's (positive = worse)

    Must be called after add_driver_rolling so rolling_finish_5 exists.
    """
    df = df.copy()

    def _deltas(grp):
        out = pd.DataFrame(
            {"q_delta_vs_teammate_s": np.nan, "teammate_rolling_finish_diff": np.nan},
            index=grp.index,
        )
        if len(grp) < 2:
            return out
        for idx in grp.index:
            tm = grp[grp.index != idx].iloc[0]
            own_q, tm_q = grp.loc[idx, "q_best_time_s"], tm["q_best_time_s"]
            out.loc[idx, "q_delta_vs_teammate_s"] = (
                own_q - tm_q if pd.notna(own_q) and pd.notna(tm_q) else np.nan
            )
            own_r5, tm_r5 = grp.loc[idx, "rolling_finish_5"], tm["rolling_finish_5"]
            out.loc[idx, "teammate_rolling_finish_diff"] = (
                own_r5 - tm_r5 if pd.notna(own_r5) and pd.notna(tm_r5) else np.nan
            )
        return out

    result = df.groupby(["year", "round", "TeamName"], group_keys=False).apply(_deltas, include_groups=False)
    df[["q_delta_vs_teammate_s", "teammate_rolling_finish_diff"]] = result
    return df


def add_reg_era(df: pd.DataFrame) -> pd.DataFrame:
    """0 = pre-2022 regulation era, 1 = 2022+ ground-effect era."""
    df = df.copy()
    df["reg_era"] = (df["year"] >= 2022).astype(int)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "grid_position",
    "q_best_time_s",
    "q_gap_to_pole_s",
    "fp2_avg_pace_s",
    "fp2_pace_gap_to_best_s",
    "q_delta_vs_teammate_s",
    "teammate_rolling_finish_diff",
    "rolling_finish_3",
    "rolling_finish_5",
    "constructor_rolling_5",
    "circuit_historical_avg",
    "dnf_rate_season",
    "season_points_cumulative",
    "race_number",
    "reg_era",
]


def build_features() -> pd.DataFrame:
    """Merge qualifying and race CSVs, compute all features, and save data/features.csv."""
    qual = pd.read_csv(DATA_DIR / "qualifying_raw.csv")
    race = pd.read_csv(DATA_DIR / "race_raw.csv")

    df = pd.merge(
        race,
        qual[["Abbreviation", "year", "round", "EventName",
              "GridPosition", "Q1_s", "Q2_s", "Q3_s", "TeamName"]],
        on=["Abbreviation", "year", "round"],
        how="left",
    )

    df["EventName"] = df["EventName_x"].combine_first(df["EventName_y"])
    df.drop(columns=["EventName_x", "EventName_y"], errors="ignore", inplace=True)

    df["finish_position"] = df["ClassifiedPosition"].apply(parse_finish_position)
    df["grid_position"]   = pd.to_numeric(df["GridPosition"], errors="coerce").fillna(20)

    df["q_best_time_s"] = df.apply(best_q_time, axis=1)
    pole_times = (
        df.groupby(["year", "round"])["q_best_time_s"]
        .min()
        .reset_index()
        .rename(columns={"q_best_time_s": "pole_time_s"})
    )
    df = df.merge(pole_times, on=["year", "round"], how="left")
    df["q_gap_to_pole_s"] = df["q_best_time_s"] - df["pole_time_s"]
    df.drop(columns=["pole_time_s"], inplace=True)

    df = add_driver_rolling(df)
    df = add_constructor_rolling(df)
    df = add_circuit_historical(df)
    df = add_dnf_rate(df)
    df = add_season_points(df)

    # New features
    df = add_fp2_features(df)
    df = add_teammate_features(df)   # needs rolling_finish_5 and q_best_time_s
    df = add_reg_era(df)

    # Race number within season
    race_nums = (
        df.groupby(["year", "round"])
        .size()
        .reset_index()[["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
    )
    race_nums["race_number"] = race_nums.groupby("year").cumcount() + 1
    df = df.merge(race_nums, on=["year", "round"], how="left")

    n_drivers = df.groupby(["year", "round"])["finish_position"].transform("count")
    df["rank_label"] = (n_drivers - df["finish_position"] + 1).clip(lower=0).fillna(0).astype(int)

    df = df.sort_values(["year", "round", "finish_position"]).reset_index(drop=True)

    out_path = DATA_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns -> {out_path}")
    print(f"Races: {df[['year','round']].drop_duplicates().shape[0]}")
    print(f"Missing values:\n{df[FEATURE_COLS].isnull().sum().to_string()}")
    return df


if __name__ == "__main__":
    build_features()

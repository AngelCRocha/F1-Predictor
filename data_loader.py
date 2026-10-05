"""
Fetches qualifying, race, and FP2 pace data from the FastF1 API for a range
of seasons and saves them as CSVs. Run this script once; subsequent runs use
the cache.

Usage:
    python data_loader.py
    python data_loader.py --years 2021 2022 2023 2024
"""

import argparse
import warnings
from pathlib import Path

import fastf1
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
DATA_DIR = Path("data")

CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


def q_time_to_seconds(td) -> float | None:
    """Convert a pandas Timedelta qualifying time to total seconds."""
    if pd.isnull(td):
        return None
    try:
        return td.total_seconds()
    except Exception:
        return None


def fetch_qualifying(year: int, round_number: int) -> pd.DataFrame | None:
    """Fetch qualifying session results for one race and return driver grid/Q-time data."""
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=False, messages=False, laps=False)
        results = session.results.copy()

        results = results[
            ["DriverNumber", "Abbreviation", "FullName", "TeamName",
             "GridPosition", "Q1", "Q2", "Q3"]
        ].copy()

        results["year"] = year
        results["round"] = round_number
        results["EventName"] = session.event["EventName"]

        for col in ("Q1", "Q2", "Q3"):
            results[f"{col}_s"] = results[col].apply(q_time_to_seconds)

        results.drop(columns=["Q1", "Q2", "Q3"], inplace=True)
        return results

    except Exception as e:
        print(f"  [warn] Qualifying {year} R{round_number}: {e}")
        return None


def fetch_race(year: int, round_number: int) -> pd.DataFrame | None:
    """Fetch race session results for one race and return driver finish/status data."""
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False, laps=False)
        results = session.results.copy()

        results = results[
            ["DriverNumber", "Abbreviation", "Position",
             "ClassifiedPosition", "Points", "Status", "Laps"]
        ].copy()

        results["year"] = year
        results["round"] = round_number
        results["EventName"] = session.event["EventName"]
        return results

    except Exception as e:
        print(f"  [warn] Race {year} R{round_number}: {e}")
        return None


def fetch_fp2_pace(year: int, round_number: int) -> pd.DataFrame | None:
    """
    Extract race-simulation pace from FP2 by finding long stints (5+ laps),
    stripping the in-lap and out-lap, and averaging the middle laps per driver.
    Returns the fastest such average per driver (in seconds).
    """
    try:
        session = fastf1.get_session(year, round_number, "FP2")
        session.load(telemetry=False, weather=False, messages=False, laps=True)
        laps = session.laps.copy()

        rows = []
        for abbr, driver_laps in laps.groupby("Driver"):
            for _, stint_laps in driver_laps.groupby("Stint"):
                valid = stint_laps[stint_laps["LapTime"].notna()]
                if len(valid) < 5:
                    continue
                # Strip first (out-lap) and last (in-lap) from the stint
                middle = valid.iloc[1:-1]
                pace = middle["LapTime"].dt.total_seconds().mean()
                rows.append({"Abbreviation": abbr, "fp2_avg_pace_s": pace})

        if not rows:
            return None

        # Keep only the fastest long-run average per driver
        df = (
            pd.DataFrame(rows)
            .groupby("Abbreviation")["fp2_avg_pace_s"]
            .min()
            .reset_index()
        )
        df["year"] = year
        df["round"] = round_number
        return df

    except Exception as e:
        print(f"  [warn] FP2 {year} R{round_number}: {e}")
        return None


def load_season(year: int) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Fetch qualifying, race, and FP2 data for every round in a season."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rounds = schedule[schedule["RoundNumber"] > 0]["RoundNumber"].tolist()

    qual_frames, race_frames, fp2_frames = [], [], []
    for r in rounds:
        event_name = schedule.loc[schedule["RoundNumber"] == r, "EventName"].values[0]
        print(f"  Fetching {year} R{r}: {event_name}")
        q   = fetch_qualifying(year, r)
        rc  = fetch_race(year, r)
        fp2 = fetch_fp2_pace(year, r)
        if q   is not None: qual_frames.append(q)
        if rc  is not None: race_frames.append(rc)
        if fp2 is not None: fp2_frames.append(fp2)

    return qual_frames, race_frames, fp2_frames


def main(years: list[int]):
    """Fetch all sessions for the given years and save qualifying_raw.csv, race_raw.csv, fp2_raw.csv."""
    all_qual, all_race, all_fp2 = [], [], []

    for year in years:
        print(f"\n=== Season {year} ===")
        q_frames, r_frames, fp2_frames = load_season(year)
        all_qual.extend(q_frames)
        all_race.extend(r_frames)
        all_fp2.extend(fp2_frames)

    qual_df = pd.concat(all_qual, ignore_index=True)
    race_df = pd.concat(all_race, ignore_index=True)
    fp2_df  = pd.concat(all_fp2,  ignore_index=True) if all_fp2 else pd.DataFrame()

    qual_path = DATA_DIR / "qualifying_raw.csv"
    race_path = DATA_DIR / "race_raw.csv"
    fp2_path  = DATA_DIR / "fp2_raw.csv"

    qual_df.to_csv(qual_path, index=False)
    race_df.to_csv(race_path, index=False)
    fp2_df.to_csv(fp2_path,   index=False)

    print(f"\nSaved {len(qual_df)} qualifying rows  -> {qual_path}")
    print(f"Saved {len(race_df)} race rows        -> {race_path}")
    print(f"Saved {len(fp2_df)}  FP2 pace rows    -> {fp2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024]
    )
    args = parser.parse_args()
    main(args.years)

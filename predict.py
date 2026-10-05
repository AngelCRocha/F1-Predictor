"""
Predict the finishing order for a specific F1 race using the trained
LambdaMART model.

The script fetches live qualifying and FP2 data for the target race, builds
features using rolling stats from the historical dataset, and prints the
predicted finishing order.

Usage:
    python predict.py --year 2024 --round 5
    python predict.py --year 2024 --round 5 --show-actual
"""

import argparse
import json
import warnings
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
DATA_DIR = Path("data")

fastf1.Cache.enable_cache(str(CACHE_DIR))

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def q_time_to_seconds(td) -> float | None:
    if pd.isnull(td):
        return None
    try:
        return td.total_seconds()
    except Exception:
        return None


def parse_finish_position(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 20.0


def best_q_time(row) -> float | None:
    times = [row.get("Q1_s"), row.get("Q2_s"), row.get("Q3_s")]
    valid = [t for t in times if t is not None and not np.isnan(float(t))]
    return min(valid) if valid else None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def fetch_qualifying_for_race(year: int, round_number: int) -> pd.DataFrame:
    session = fastf1.get_session(year, round_number, "Q")
    session.load(telemetry=False, weather=False, messages=False, laps=False)
    res = session.results.copy()

    for col in ("Q1", "Q2", "Q3"):
        res[f"{col}_s"] = res[col].apply(q_time_to_seconds)

    res["year"] = year
    res["round"] = round_number
    res["EventName"] = session.event["EventName"]
    res["grid_position"] = pd.to_numeric(res["GridPosition"], errors="coerce").fillna(20)
    res["q_best_time_s"] = res.apply(best_q_time, axis=1)
    pole = res["q_best_time_s"].min()
    res["q_gap_to_pole_s"] = res["q_best_time_s"] - pole
    return res


def fetch_fp2_pace_for_race(year: int, round_number: int) -> dict[str, float]:
    """
    Returns {abbreviation: fp2_avg_pace_s} for the target race.
    Returns empty dict on failure (features will be NaN, imputed by model).
    """
    try:
        session = fastf1.get_session(year, round_number, "FP2")
        session.load(telemetry=False, weather=False, messages=False, laps=True)
        laps = session.laps.copy()

        best_pace: dict[str, float] = {}
        for abbr, driver_laps in laps.groupby("Driver"):
            for _, stint_laps in driver_laps.groupby("Stint"):
                valid = stint_laps[stint_laps["LapTime"].notna()]
                if len(valid) < 5:
                    continue
                middle = valid.iloc[1:-1]
                pace = middle["LapTime"].dt.total_seconds().mean()
                if abbr not in best_pace or pace < best_pace[abbr]:
                    best_pace[abbr] = pace

        return best_pace
    except Exception as e:
        print(f"  [warn] FP2 fetch failed ({e}), fp2 features will be NaN")
        return {}


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_predict_features(
    qual: pd.DataFrame,
    fp2_pace: dict[str, float],
    history: pd.DataFrame,
    year: int,
    round_number: int,
) -> pd.DataFrame:
    event_name = qual["EventName"].iloc[0]

    past = history[
        (history["year"] < year) |
        ((history["year"] == year) & (history["round"] < round_number))
    ].copy()

    # FP2 gap to best in this session
    fp2_best = min(fp2_pace.values()) if fp2_pace else np.nan

    # Build a quick lookup for teammate qualifying times (same TeamName)
    team_q: dict[str, list[float]] = {}
    team_r5: dict[str, list[float]] = {}
    for _, q_row in qual.iterrows():
        team = str(q_row.get("TeamName", ""))
        qt = q_row.get("q_best_time_s")
        if team and pd.notna(qt):
            team_q.setdefault(team, []).append(float(qt))
        abbr = str(q_row["Abbreviation"])
        driver_past = past[past["Abbreviation"] == abbr].sort_values(["year", "round"])
        if len(driver_past) >= 1:
            r5 = driver_past["finish_position"].iloc[-5:].mean()
        else:
            r5 = 10.0
        team_r5.setdefault(team, []).append(r5)

    rows = []
    for _, q_row in qual.iterrows():
        abbr  = str(q_row["Abbreviation"])
        team  = str(q_row.get("TeamName", "Unknown"))

        driver_past = past[past["Abbreviation"] == abbr].sort_values(["year", "round"])
        team_past   = past[past["TeamName"] == team].sort_values(["year", "round"])

        # Rolling finish averages
        if len(driver_past) >= 1:
            pos_series = driver_past["finish_position"]
            rolling_3 = pos_series.iloc[-3:].mean()
            rolling_5 = pos_series.iloc[-5:].mean()
        else:
            rolling_3 = rolling_5 = 10.0

        # Constructor rolling
        if len(team_past) >= 1:
            team_avg      = team_past.groupby(["year", "round"])["finish_position"].mean()
            constructor_roll = team_avg.iloc[-5:].mean()
        else:
            constructor_roll = 10.0

        # Circuit historical
        circuit_past = driver_past[driver_past["EventName"] == event_name]
        circuit_avg  = circuit_past["finish_position"].mean() if len(circuit_past) > 0 else np.nan

        # DNF rate & points this season
        season_past = driver_past[driver_past["year"] == year]
        if len(season_past) > 0:
            n_dnf = (~season_past["Status"].astype(str).str.startswith("Finished")).sum()
            n_dnf -= season_past["Status"].astype(str).str.startswith("+").sum()
            dnf_rate  = max(0, n_dnf) / len(season_past)
            season_pts = season_past["Points"].sum()
        else:
            dnf_rate = 0.0
            season_pts = 0.0

        # FP2 pace features
        fp2_pace_val = fp2_pace.get(abbr, np.nan)
        fp2_gap      = (fp2_pace_val - fp2_best) if pd.notna(fp2_pace_val) and pd.notna(fp2_best) else np.nan

        # Teammate features
        own_q = q_row.get("q_best_time_s")
        tm_qs = [v for v in team_q.get(team, []) if v != float(own_q)] if pd.notna(own_q) else []
        q_delta_tm = (float(own_q) - tm_qs[0]) if tm_qs else np.nan

        own_r5  = rolling_5
        tm_r5s  = [v for v in team_r5.get(team, []) if v != own_r5]
        tm_finish_diff = (own_r5 - tm_r5s[0]) if tm_r5s else np.nan

        rows.append({
            "Abbreviation": abbr,
            "FullName":     q_row.get("FullName", abbr),
            "TeamName":     team,
            "grid_position":              q_row["grid_position"],
            "q_best_time_s":              own_q,
            "q_gap_to_pole_s":            q_row["q_gap_to_pole_s"],
            "fp2_avg_pace_s":             fp2_pace_val,
            "fp2_pace_gap_to_best_s":     fp2_gap,
            "q_delta_vs_teammate_s":      q_delta_tm,
            "teammate_rolling_finish_diff": tm_finish_diff,
            "rolling_finish_3":           rolling_3,
            "rolling_finish_5":           rolling_5,
            "constructor_rolling_5":      constructor_roll,
            "circuit_historical_avg":     circuit_avg,
            "dnf_rate_season":            dnf_rate,
            "season_points_cumulative":   season_pts,
            "race_number":                past[past["year"] == year]["round"].nunique() + 1,
            "reg_era":                    1,   # predictions always 2022+
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(year: int, round_number: int, show_actual: bool = False, n_runs: int = 10):
    features_path = DATA_DIR / "features.csv"

    if not features_path.exists():
        raise FileNotFoundError("features.csv not found. Run `python feature_engineering.py` first.")

    history = pd.read_csv(features_path)

    print(f"Fetching qualifying data for {year} Round {round_number}...")
    qual = fetch_qualifying_for_race(year, round_number)

    print("Fetching FP2 long-run pace...")
    fp2_pace = fetch_fp2_pace_for_race(year, round_number)

    race_df = build_predict_features(qual, fp2_pace, history, year, round_number)

    # Train on all races that occurred before this one (including same year)
    train_history = history[
        (history["year"] < year) |
        ((history["year"] == year) & (history["round"] < round_number))
    ]

    # Hold out the last race as a validation set for model selection
    last_key = (
        train_history.sort_values(["year", "round"])[["year", "round"]]
        .drop_duplicates()
        .iloc[-1]
    )
    val_mask = (
        (train_history["year"] == last_key["year"]) &
        (train_history["round"] == last_key["round"])
    )
    val_df = train_history[val_mask]
    fit_df = train_history[~val_mask]

    imputer = SimpleImputer(strategy="median")
    scaler  = MinMaxScaler()
    X_fit      = scaler.fit_transform(imputer.fit_transform(fit_df[FEATURE_COLS]))
    y_fit      = fit_df["rank_label"].values
    fit_groups = fit_df.groupby(["year", "round"], sort=False).size().values
    fit_weight = np.where(fit_df["year"] == 2021, 0.5, 1.0)

    X_val = scaler.transform(imputer.transform(val_df[FEATURE_COLS]))
    y_val = val_df["rank_label"].values

    params_path = Path("models") / "best_params.json"
    hp = json.load(open(params_path)) if params_path.exists() else {}

    hp_base = {
        "n_estimators":      hp.get("n_estimators", 500),
        "learning_rate":     hp.get("learning_rate", 0.05),
        "num_leaves":        hp.get("num_leaves", 31),
        "min_child_samples": hp.get("min_child_samples", 5),
        "subsample":         hp.get("subsample", 0.8),
        "colsample_bytree":  hp.get("colsample_bytree", 0.8),
    }
    hp_extra = {k: v for k, v in hp.items() if k not in hp_base}

    best_model, best_score = None, -np.inf
    print(f"Training {n_runs} runs, selecting best NDCG (all positions) on "
          f"{int(last_key['year'])} R{int(last_key['round'])}...")
    for i in range(n_runs):
        m = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[3, 5],
            random_state=i,
            verbose=-1,
            **hp_base,
            **hp_extra,
        )
        m.fit(X_fit, y_fit, group=fit_groups, sample_weight=fit_weight)
        score = ndcg_score([y_val], [m.predict(X_val)])
        marker = " *" if score > best_score else ""
        print(f"  run {i+1:>2}/{n_runs}  NDCG = {score:.4f}{marker}")
        if score > best_score:
            best_score, best_model = score, m

    print(f"Best NDCG: {best_score:.4f}")
    model = best_model

    X = scaler.transform(imputer.transform(race_df[FEATURE_COLS]))
    scores = model.predict(X)

    race_df["predicted_score"]    = scores
    race_df["predicted_position"] = race_df["predicted_score"].rank(ascending=False).astype(int)
    race_df = race_df.sort_values("predicted_position")

    event_name = qual["EventName"].iloc[0]

    if show_actual:
        try:
            print("Fetching actual race results...")
            session = fastf1.get_session(year, round_number, "R")
            session.load(telemetry=False, weather=False, messages=False, laps=False)
            actual = session.results[["Abbreviation", "FullName", "ClassifiedPosition"]].copy()
            actual["finish"] = actual["ClassifiedPosition"].apply(parse_finish_position)
            actual = actual.sort_values("finish").reset_index(drop=True)
        except Exception as e:
            print(f"Could not fetch actual results: {e}")
            actual = None
    else:
        actual = None

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    print(f"\n{'='*78}")
    print(f"  {event_name} {year}")
    print(f"{'='*78}")

    if actual is not None:
        actual_by_abbr = {
            row["Abbreviation"]: (int(row["finish"]), str(row["FullName"]))
            for _, row in actual.iterrows()
        }

        print(f"  {'#':>2}  {'── PREDICTED ──':<32}   {'#':>2}  {'── ACTUAL ──'}")
        print(f"  {'-'*30}   {'-'*30}")

        pred_rows = race_df.sort_values("predicted_position").reset_index(drop=True)
        act_rows  = actual.reset_index(drop=True)

        for i in range(max(len(pred_rows), len(act_rows))):
            if i < len(pred_rows):
                p        = pred_rows.iloc[i]
                pred_abbr = str(p["Abbreviation"])
                act_pos, _ = actual_by_abbr.get(pred_abbr, (99, ""))
                match    = "✓" if int(p["predicted_position"]) <= 3 and act_pos <= 3 else " "
                pred_str = f"{match} {int(p['predicted_position']):>2}  {str(p['FullName']):<26}"
            else:
                pred_str = " " * 32

            act_str = (
                f"{int(act_rows.iloc[i]['finish']):>2}  {str(act_rows.iloc[i]['FullName']):<26}"
                if i < len(act_rows) else ""
            )
            print(f"  {pred_str}   {act_str}")

        # Accuracy summary
        pred_by_abbr = {
            str(row["Abbreviation"]): int(row["predicted_position"])
            for _, row in pred_rows.iterrows()
        }
        common     = set(pred_by_abbr) & set(actual_by_abbr)
        n_drivers  = len(common)
        exact      = sum(1 for a in common if pred_by_abbr[a] == actual_by_abbr[a][0])
        top3_hits  = len({a for a in common if pred_by_abbr[a] <= 3}  & {a for a in common if actual_by_abbr[a][0] <= 3})
        top5_hits  = len({a for a in common if pred_by_abbr[a] <= 5}  & {a for a in common if actual_by_abbr[a][0] <= 5})
        top10_hits = len({a for a in common if pred_by_abbr[a] <= 10} & {a for a in common if actual_by_abbr[a][0] <= 10})
        rho, _     = spearmanr(
            [pred_by_abbr[a]      for a in common],
            [actual_by_abbr[a][0] for a in common],
        )

        print(f"\n{'='*78}")
        print(f"  Accuracy Summary")
        print(f"  {'-'*40}")
        print(f"  Exact position matches : {exact}/{n_drivers} ({exact/n_drivers*100:.1f}%)")
        print(f"  Top-3  correct         : {top3_hits}/3  ({top3_hits/3*100:.1f}%)")
        print(f"  Top-5  correct         : {top5_hits}/5  ({top5_hits/5*100:.1f}%)")
        print(f"  Top-10 correct         : {top10_hits}/10 ({top10_hits/10*100:.1f}%)")
        print(f"  Spearman rank corr     : {rho:.3f}")
    else:
        print(f"  {'#':>2}  {'Driver':<26}  {'Team':<22} {'Grid':>4}")
        print(f"  {'-'*60}")
        for _, row in race_df.sort_values("predicted_position").iterrows():
            print(
                f"  {int(row['predicted_position']):>2}  "
                f"{str(row['FullName']):<26}  "
                f"{str(row['TeamName']):<22} "
                f"{int(row['grid_position']):>4}"
            )

    print(f"{'='*78}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",  type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument(
        "--show-actual", action="store_true",
        help="Also print the actual race result for comparison"
    )
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of training runs to compare (default: 10)")
    args = parser.parse_args()
    predict(args.year, args.round, args.show_actual, args.runs)

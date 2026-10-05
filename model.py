"""
Trains a LambdaMART ranking model (LightGBM LGBMRanker) to predict F1 race
finishing order. Also computes a grid-position baseline for comparison.

Usage:
    python model.py
    python model.py --train-years 2021 2022 2023 --test-year 2024
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import optuna

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRanker, early_stopping, log_evaluation
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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
# Data loading
# ---------------------------------------------------------------------------

def load_split(
    train_years: list[int], test_year: int, predict_round: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:

    df = pd.read_csv(DATA_DIR / "features.csv")

    if predict_round is not None:
        train_mask = (df["year"] < test_year) | (
            (df["year"] == test_year) & (df["round"] < predict_round)
        )
        test_mask = (df["year"] == test_year) & (df["round"] == predict_round)
    else:
        train_mask = df["year"].isin(train_years)
        test_mask  = df["year"] == test_year

    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()

    # Impute missing features with column medians from training data
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[FEATURE_COLS])
    X_test = imputer.transform(test_df[FEATURE_COLS])

    # Scale features (helps LightGBM converge, not strictly required)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = train_df["rank_label"].values
    y_test = test_df["rank_label"].values

    # Group array: number of drivers per race, in dataset order
    train_groups = (
        train_df.groupby(["year", "round"], sort=False).size().values
    )
    test_groups = (
        test_df.groupby(["year", "round"], sort=False).size().values
    )

    return X_train, X_test, y_train, y_test, train_groups, test_groups, train_df, test_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true_labels: np.ndarray,
    y_pred_scores: np.ndarray,
    groups: np.ndarray,
    finish_positions: np.ndarray,
) -> dict:
    """
    Computes NDCG@3, NDCG@5, podium accuracy, and Spearman rho.
    y_pred_scores: higher = predicted better rank.
    """
    ndcg3_scores, ndcg5_scores = [], []
    podium_hits = 0
    spearman_vals = []
    idx = 0

    for g in groups:
        labels = y_true_labels[idx: idx + g]
        scores = y_pred_scores[idx: idx + g]
        pos = finish_positions[idx: idx + g]

        # NDCG needs 2D arrays
        ndcg3_scores.append(
            ndcg_score([labels], [scores], k=3) if g >= 3 else np.nan
        )
        ndcg5_scores.append(
            ndcg_score([labels], [scores], k=5) if g >= 5 else np.nan
        )

        # Podium accuracy: >=2 of true top-3 appear in predicted top-3
        true_top3 = set(np.argsort(pos)[:3])
        pred_top3 = set(np.argsort(-scores)[:3])
        if len(true_top3 & pred_top3) >= 2:
            podium_hits += 1

        # Spearman rank correlation for this race
        rho, _ = spearmanr(-scores, pos)
        spearman_vals.append(rho)

        idx += g

    return {
        "NDCG@3": float(np.nanmean(ndcg3_scores)),
        "NDCG@5": float(np.nanmean(ndcg5_scores)),
        "Podium accuracy": podium_hits / len(groups),
        "Spearman rho": float(np.nanmean(spearman_vals)),
    }


# ---------------------------------------------------------------------------
# Baseline: predict final position == grid position
# ---------------------------------------------------------------------------

def baseline_metrics(test_df: pd.DataFrame, test_groups: np.ndarray) -> dict:
    y_true = test_df["rank_label"].values
    # Higher grid_position number = worse; invert so higher score = better
    grid_scores = -test_df["grid_position"].values
    finish_pos = test_df["finish_position"].values
    return compute_metrics(y_true, grid_scores, test_groups, finish_pos)


# ---------------------------------------------------------------------------
# LambdaMART model
# ---------------------------------------------------------------------------

def train_lambdamart(
    X_train, y_train, train_groups,
    X_val, y_val, val_groups,
    sample_weight=None,
    **hyperparams,
) -> LGBMRanker:
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3, 5],
        n_estimators=hyperparams.pop("n_estimators", 500),
        learning_rate=hyperparams.pop("learning_rate", 0.05),
        num_leaves=hyperparams.pop("num_leaves", 31),
        min_child_samples=hyperparams.pop("min_child_samples", 5),
        subsample=hyperparams.pop("subsample", 0.8),
        colsample_bytree=hyperparams.pop("colsample_bytree", 0.8),
        random_state=42,
        verbose=-1,
        **hyperparams,
    )
    model.fit(
        X_train, y_train,
        group=train_groups,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(50)],
    )
    return model


# ---------------------------------------------------------------------------
# Hyperparameter optimisation
# ---------------------------------------------------------------------------

def optimize_hyperparams(n_trials: int = 50) -> dict:
    df = pd.read_csv(DATA_DIR / "features.csv")
    folds = [
        ([2021],             2022),
        ([2021, 2022],       2023),
        ([2021, 2022, 2023], 2024),
    ]

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l1":         trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2":         trial.suggest_float("lambda_l2", 0.0, 5.0),
        }
        scores = []
        for train_years, val_year in folds:
            train_df = df[df["year"].isin(train_years)].copy()
            val_df   = df[df["year"] == val_year].copy()
            imputer  = SimpleImputer(strategy="median")
            scaler   = MinMaxScaler()
            X_tr = scaler.fit_transform(imputer.fit_transform(train_df[FEATURE_COLS]))
            X_va = scaler.transform(imputer.transform(val_df[FEATURE_COLS]))
            y_tr = train_df["rank_label"].values
            y_va = val_df["rank_label"].values
            tr_g = train_df.groupby(["year", "round"], sort=False).size().values
            va_g = val_df.groupby(["year", "round"], sort=False).size().values
            w    = np.where(train_df["year"] == 2021, 0.5, 1.0)
            m = LGBMRanker(
                objective="lambdarank", metric="ndcg", ndcg_eval_at=[3, 5],
                n_estimators=1000, random_state=42, verbose=-1, **params,
            )
            m.fit(
                X_tr, y_tr, group=tr_g, sample_weight=w,
                eval_set=[(X_va, y_va)], eval_group=[va_g],
                callbacks=[early_stopping(50, verbose=False)],
            )
            fold_metrics = compute_metrics(
                y_va, m.predict(X_va), va_g, val_df["finish_position"].values
            )
            scores.append(fold_metrics["NDCG@3"])
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\nBest NDCG@3 (CV): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_feature_importance(model: LGBMRanker, save_path: Path):
    importances = model.feature_importances_
    feat_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": importances}
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    ax.set_xlabel("Importance (split count)")
    ax.set_title("LambdaMART Feature Importance")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Feature importance plot saved -> {save_path}")
    plt.close(fig)


def plot_metrics_comparison(lambdamart_m: dict, baseline_m: dict, save_path: Path):
    metric_names = list(lambdamart_m.keys())
    lm_vals = [lambdamart_m[k] for k in metric_names]
    bl_vals = [baseline_m[k] for k in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, bl_vals, width, label="Baseline (grid pos)", color="salmon")
    ax.bar(x + width / 2, lm_vals, width, label="LambdaMART", color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("LambdaMART vs. Grid-Position Baseline (2024 test season)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Metrics comparison plot saved -> {save_path}")
    plt.close(fig)


def plot_predicted_vs_actual(
    test_df: pd.DataFrame, scores: np.ndarray, test_groups: np.ndarray,
    save_path: Path,
):
    """Scatter: predicted rank vs actual finishing position for all test races."""
    predicted_positions = []
    idx = 0
    for g in test_groups:
        s = scores[idx: idx + g]
        # rank within group (1 = highest score = predicted 1st)
        ranks = pd.Series(s).rank(ascending=False).values
        predicted_positions.extend(ranks.tolist())
        idx += g

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        test_df["finish_position"], predicted_positions,
        alpha=0.3, s=10, color="steelblue",
    )
    ax.plot([1, 20], [1, 20], "r--", linewidth=1)
    ax.set_xlabel("Actual Finishing Position")
    ax.set_ylabel("Predicted Position")
    ax.set_title("Predicted vs Actual Position (2024)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Scatter plot saved -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    train_years: list[int],
    test_year: int,
    predict_round: int | None = None,
    tune: bool = False,
    n_trials: int = 50,
):
    params_path = MODEL_DIR / "best_params.json"

    if tune:
        print(f"Running Optuna search ({n_trials} trials)...")
        best_params = optimize_hyperparams(n_trials)
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved -> {params_path}")
    elif params_path.exists():
        with open(params_path) as f:
            best_params = json.load(f)
        print(f"Loaded tuned params from {params_path}")
    else:
        best_params = {}

    print("Loading features...")
    (X_train, X_test, y_train, y_test,
     train_groups, test_groups, train_df, test_df) = load_split(train_years, test_year, predict_round)

    print(f"Train: {X_train.shape[0]} rows, {len(train_groups)} races")
    print(f"Test : {X_test.shape[0]} rows, {len(test_groups)} races\n")

    # Baseline
    print("--- Grid-position baseline ---")
    bl_metrics = baseline_metrics(test_df, test_groups)
    for k, v in bl_metrics.items():
        print(f"  {k:<22} {v:.4f}")

    # Reg-era sample weights: downweight 2021 (pre-regulation-reset) data
    train_weights = np.where(train_df["year"] == 2021, 0.5, 1.0)

    # LambdaMART
    print("\nTraining LambdaMART...")
    model = train_lambdamart(X_train, y_train, train_groups,
                             X_test, y_test, test_groups,
                             sample_weight=train_weights,
                             **best_params)

    scores = model.predict(X_test)
    lm_metrics = compute_metrics(y_test, scores, test_groups,
                                 test_df["finish_position"].values)

    print("\n--- LambdaMART (2024 test) ---")
    for k, v in lm_metrics.items():
        print(f"  {k:<22} {v:.4f}")

    print("\n--- Delta (LambdaMART - Baseline) ---")
    for k in lm_metrics:
        delta = lm_metrics[k] - bl_metrics[k]
        sign = "+" if delta >= 0 else ""
        print(f"  {k:<22} {sign}{delta:.4f}")

    # Save model
    model_path = MODEL_DIR / "lambdamart_f1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved -> {model_path}")

    # Plots
    plot_feature_importance(model, MODEL_DIR / "feature_importance.png")
    plot_metrics_comparison(lm_metrics, bl_metrics, MODEL_DIR / "metrics_comparison.png")
    plot_predicted_vs_actual(test_df, scores, test_groups,
                             MODEL_DIR / "predicted_vs_actual.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--predict-round", type=int, default=None)
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search before training")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()
    main(args.train_years, args.test_year, args.predict_round, args.tune, args.n_trials)

import argparse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import joblib

from features import build_player_week_features, FEATURE_COLS

def load_features(years):
    csv = build_player_week_features(years)
    df = pd.read_csv(csv)
    return df

def _prepare_xy(df: pd.DataFrame):
    # Ensure all feature columns exist and are numeric
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df['fantasy_points'], errors="coerce")

    # Drop rows with missing target
    mask = y.notna()
    X, y = X[mask], y[mask]

    return X, y

def train_and_save(df: pd.DataFrame, models_dir: str = "models"):
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    X, y = _prepare_xy(df)

    # Pipelines with imputers so GBM gets no NaNs
    mean_model = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(random_state=42, n_estimators=300, max_depth=3)
    )
    p20 = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(loss="quantile", alpha=0.2, random_state=42, n_estimators=300, max_depth=3)
    )
    p80 = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingRegressor(loss="quantile", alpha=0.8, random_state=42, n_estimators=300, max_depth=3)
    )

    mean_model.fit(X, y)
    p20.fit(X, y)
    p80.fit(X, y)

    # For now, share models across positions (specialize later)
    for pos in ["QB","RB","WR","TE"]:
        joblib.dump(mean_model, Path(models_dir)/f"{pos}_mean.joblib")
        joblib.dump(p20, Path(models_dir)/f"{pos}_p20.joblib")
        joblib.dump(p80, Path(models_dir)/f"{pos}_p80.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='+', type=int, required=True, help='e.g. 2019 2020 2021 2022 2023 2024')
    ap.add_argument('--models_dir', default='models')
    args = ap.parse_args()

    df = load_features(args.years)
    train_and_save(df, models_dir=args.models_dir)
    print(f"Saved models to {args.models_dir}")

# projections.py

# Models: load per-position regressors and generate projections.
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from features import FEATURE_COLS

POSITIONS = ["QB", "RB", "WR", "TE"]


class ProjectionModels:
    # Init: hold model paths and dicts for mean / p20 / p80.
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.mean_models: dict[str, object] = {}
        self.p20_models: dict[str, object] = {}
        self.p80_models: dict[str, object] = {}
        self._load()

    # Load: pull joblib artifacts if they exist (per position).
    def _load(self) -> None:
        if not self.models_dir.exists():
            return
        for pos in POSITIONS:
            m_mean = self.models_dir / f"{pos}_mean.joblib"
            m_p20 = self.models_dir / f"{pos}_p20.joblib"
            m_p80 = self.models_dir / f"{pos}_p80.joblib"
            if m_mean.exists():
                self.mean_models[pos] = joblib.load(m_mean)
            if m_p20.exists():
                self.p20_models[pos] = joblib.load(m_p20)
            if m_p80.exists():
                self.p80_models[pos] = joblib.load(m_p80)

    # Status: at least one mean model found?
    def has_models(self) -> bool:
        return len(self.mean_models) > 0

    # Project: create proj_mean / p20 / p80 using training feature set.
    def project(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Input columns (minimum): ['player_id','name','position'] + FEATURE_COLS
        Output: same rows with ['proj_mean','proj_p20','proj_p80'] added when models exist.
        """
        out = df_features.copy()

        # Columns: ensure output placeholders exist.
        for c in ("proj_mean", "proj_p20", "proj_p80"):
            if c not in out.columns:
                out[c] = np.nan

        # Prep: normalize position labels defensively.
        if "position" in out.columns:
            out["position"] = out["position"].astype(str).str.upper()

        # Featurize: build X exactly as during training (order + types).
        def make_X(df: pd.DataFrame) -> pd.DataFrame:
            X = df.copy()
            for col in FEATURE_COLS:
                if col not in X.columns:
                    X[col] = 0.0
            # enforce column order & numeric dtype
            X = X[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return X

        # Predict: loop per position to use the right model set.
        for pos, sub in out.groupby("position", dropna=False):
            if pos not in POSITIONS:
                continue  # skip DEF/K/etc.
            X = make_X(sub)
            idx = sub.index
            if pos in self.mean_models:
                out.loc[idx, "proj_mean"] = self.mean_models[pos].predict(X)
            if pos in self.p20_models:
                out.loc[idx, "proj_p20"] = self.p20_models[pos].predict(X)
            if pos in self.p80_models:
                out.loc[idx, "proj_p80"] = self.p80_models[pos].predict(X)

        return out

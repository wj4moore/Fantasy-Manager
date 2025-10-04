# projections.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from features import FEATURE_COLS

POSITIONS = ["QB", "RB", "WR", "TE"]

class ProjectionModels:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.mean_models = {}
        self.p20_models = {}
        self.p80_models = {}
        self._load()

    def _load(self):
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

    def has_models(self) -> bool:
        return len(self.mean_models) > 0

    def project(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a DataFrame with columns: ['player_id','name','position'] + FEATURE_COLS
        Returns the same with ['proj_mean','proj_p20','proj_p80'] filled in where models exist.
        """
        out = df_features.copy()
        for c in ("proj_mean", "proj_p20", "proj_p80"):
            if c not in out.columns:
                out[c] = np.nan

        # Build X using exactly the training features
        def make_X(df: pd.DataFrame) -> pd.DataFrame:
            X = df.copy()
            # ensure features exist and in the right order
            for col in FEATURE_COLS:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return X

        for pos, sub in out.groupby("position"):
            X = make_X(sub)
            if pos in self.mean_models:
                out.loc[sub.index, "proj_mean"]  = self.mean_models[pos].predict(X)
            if pos in self.p20_models:
                out.loc[sub.index, "proj_p20"]   = self.p20_models[pos].predict(X)
            if pos in self.p80_models:
                out.loc[sub.index, "proj_p80"]   = self.p80_models[pos].predict(X)

        return out

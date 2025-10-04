# waivers.py
from __future__ import annotations
import pandas as pd
import numpy as np

DEFAULT_FAAB_BUDGET = 100

def _as_num(s, default=np.nan):
    s = pd.to_numeric(s, errors="coerce")
    if isinstance(s, pd.Series):
        return s
    return default if pd.isna(s) else s

def _sanitize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out

def replacement_levels(roster_df: pd.DataFrame, quantile: float = 0.35) -> pd.DataFrame:
    """
    Compute replacement level by position from your roster.
    Uses given quantile of 'projected_adj' (if present) else 'projected'.
    Default: 35th percentile (0.35) instead of median.
    """
    df = roster_df.copy()
    base_col = "projected_adj" if "projected_adj" in df.columns else "projected"
    df[base_col] = pd.to_numeric(df[base_col], errors="coerce").replace([np.inf,-np.inf], np.nan)

    if df[base_col].notna().sum() == 0:
        # fallback if everything is NaN
        return pd.DataFrame({
            "position": ["QB","RB","WR","TE","K","DEF"],
            "repl": [18.0,12.0,12.0,9.0,8.0,7.0]
        })

    # use quantile instead of median
    repl = (
        df.groupby("position")[base_col]
        .quantile(quantile, interpolation="linear")
        .rename("repl")
        .reset_index()
    )

    # fallback if a position has NaN
    overall = df[base_col].quantile(quantile)
    repl["repl"] = pd.to_numeric(repl["repl"], errors="coerce").fillna(overall).fillna(8.0)
    return repl


def compute_waiver_values(
    fa_df: pd.DataFrame,
    repl_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns fa_df with:
      - projected_waiver (uses projected_adj if present else proj_mean/projected)
      - PAR (proj - replacement at position)
    All numeric columns sanitized to avoid NaN/inf.
    """
    base_col_fa = (
        "projected_adj" if "projected_adj" in fa_df.columns
        else "proj_mean" if "proj_mean" in fa_df.columns
        else "projected"
    )

    out = fa_df.copy()
    out = out.merge(repl_df, on="position", how="left")

    out = _sanitize(out, [base_col_fa, "repl"])
    # final fallbacks in case 'repl' missing
    out["repl"] = out["repl"].fillna(out[base_col_fa].median()).fillna(8.0)

    out["projected_waiver"] = pd.to_numeric(out.get(base_col_fa), errors="coerce")
    out["projected_waiver"] = out["projected_waiver"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)

    out["PAR"] = out["projected_waiver"] - out["repl"]
    out["PAR"] = out["PAR"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out

def suggest_faab_bids(
    fa_values: pd.DataFrame,
    budget_remaining: int = DEFAULT_FAAB_BUDGET,
    aggressiveness: float = 0.35,
) -> pd.DataFrame:
    """
    Convert PAR to FAAB suggestions.
    - aggressiveness in [0,1]; higher → bid more per PAR point.
    - Simple rule: dollars_per_point = aggressiveness * (budget_remaining / 25)
    """
    df = fa_values.copy()
    df = _sanitize(df, ["PAR"])

    # guardrails on inputs
    try:
        budget_remaining = int(budget_remaining)
    except Exception:
        budget_remaining = DEFAULT_FAAB_BUDGET
    aggressiveness = float(aggressiveness)
    aggressiveness = min(max(aggressiveness, 0.0), 1.0)

    dollars_per_point = aggressiveness * (budget_remaining / 25.0)
    df["FAAB_suggest"] = np.maximum(0.0, df["PAR"]) * float(dollars_per_point)
    # clean before int cast
    df["FAAB_suggest"] = pd.to_numeric(df["FAAB_suggest"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["FAAB_suggest"] = df["FAAB_suggest"].round().astype(int)

    return df

def rank_waivers(
    fa_df: pd.DataFrame,
    repl_df: pd.DataFrame,
    budget_remaining: int = DEFAULT_FAAB_BUDGET,
    aggressiveness: float = 0.35,
    top: int = 10
) -> pd.DataFrame:
    """
    Produce a ranked waiver table with PAR and FAAB suggestions (NaN/inf-proof).
    """
    vals = compute_waiver_values(fa_df, repl_df)
    vals = suggest_faab_bids(vals, budget_remaining=budget_remaining, aggressiveness=aggressiveness)
    cols = [c for c in ["player_id","name","position","projected_waiver","PAR","FAAB_suggest"] if c in vals.columns]
    vals = vals.sort_values(["PAR","projected_waiver"], ascending=[False, False])
    if top is not None:
        vals = vals.head(int(top))
    return vals[cols]

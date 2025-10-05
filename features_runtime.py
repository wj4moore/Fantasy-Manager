# features_runtime.py

# Imports: core libs and nfl_data_py (pandas backend expected).
from __future__ import annotations
import pandas as pd
import numpy as np
import nfl_data_py as nfl

# Schema: columns the manager expects from runtime features.
FEATURE_COLS_RUNTIME = [
    "is_home",
    "opp_pass_rank", "opp_rush_rank",
    "rolling_fp_3", "rolling_oppty_3",
    "vegas_total",
    "weather_wind",
]

# Scoring: vanilla fantasy weights (tweak for PPR/boosts).
POS_FP_WEIGHTS = {
    "passing_yards": 0.04, "passing_tds": 4, "interceptions": -1,
    "rushing_yards": 0.1,  "rushing_tds": 6,
    "receiving_yards": 0.1, "receiving_tds": 6, "receptions": 0,  # set to 1 for PPR
}

# Utils: safe fantasy points from row-wise stat columns.
def _fantasy_points(df: pd.DataFrame) -> pd.Series:
    s = (
        df.get("passing_yards", 0)   * POS_FP_WEIGHTS["passing_yards"] +
        df.get("passing_tds", 0)     * POS_FP_WEIGHTS["passing_tds"]   +
        df.get("interceptions", 0)   * POS_FP_WEIGHTS["interceptions"] +
        df.get("rushing_yards", 0)   * POS_FP_WEIGHTS["rushing_yards"] +
        df.get("rushing_tds", 0)     * POS_FP_WEIGHTS["rushing_tds"]   +
        df.get("receiving_yards", 0) * POS_FP_WEIGHTS["receiving_yards"] +
        df.get("receiving_tds", 0)   * POS_FP_WEIGHTS["receiving_tds"] +
        df.get("receptions", 0)      * POS_FP_WEIGHTS["receptions"]
    )
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

# Utils: return numeric series for a column or zeros if missing.
def _series_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)

# Features: opportunity proxy (RB/WR/TE uses carries+targets, QB uses rush+0.5*att).
def _opportunity(df: pd.DataFrame) -> pd.Series:
    rush_att = _series_or_zero(df, "rushing_attempts")
    targets  = _series_or_zero(df, "targets")
    attempts = _series_or_zero(df, "attempts")  # passing attempts
    pos = df["position"] if "position" in df.columns else pd.Series("", index=df.index)
    qb_mask = pos.eq("QB")
    opp = rush_att + targets
    opp = opp.where(~qb_mask, rush_att + 0.5 * attempts)
    return opp.fillna(0.0)

# Defense Ranks: rolling fantasy allowed by defense; lower rank = tougher defense.
def _def_rank_allowed(weekly: pd.DataFrame, weeks_window: int = 5) -> pd.DataFrame:
    df = weekly.copy()
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")
    df["position"] = df.get("position", "").astype(str).fillna("")

    # Defense key: prefer 'opponent'; fallback to 'defteam'.
    if "opponent" in df.columns:
        df["defteam"] = df["opponent"].astype(str)
    else:
        df["defteam"] = df.get("defteam", "").astype(str)

    if "defteam" not in df.columns or df["defteam"].eq("").all():
        return pd.DataFrame(columns=["defteam", "week", "pass_rank", "rush_rank"])

    df["fantasy"] = _fantasy_points(df)

    conceded = (
        df.groupby(["defteam", "week", "position"], as_index=False)["fantasy"]
          .sum()
          .sort_values(["position", "defteam", "week"])
    )

    conceded["rolling_fantasy_allowed"] = (
        conceded.groupby(["position", "defteam"])["fantasy"]
                .transform(lambda s: s.rolling(weeks_window, min_periods=1).mean())
    )

    pass_pos = {"QB", "WR", "TE"}
    rush_pos = {"RB"}

    pass_allowed = (
        conceded[conceded["position"].isin(pass_pos)]
        .groupby(["defteam", "week"], as_index=False)["rolling_fantasy_allowed"]
        .mean()
        .rename(columns={"rolling_fantasy_allowed": "pass_allowed"})
    )

    rush_allowed = (
        conceded[conceded["position"].isin(rush_pos)]
        .groupby(["defteam", "week"], as_index=False)["rolling_fantasy_allowed"]
        .mean()
        .rename(columns={"rolling_fantasy_allowed": "rush_allowed"})
    )

    ranks = pass_allowed.merge(rush_allowed, on=["defteam", "week"], how="outer")

    if ranks.empty:
        return pd.DataFrame(columns=["defteam", "week", "pass_rank", "rush_rank"])

    # Ranking: smaller allowed → smaller rank; fill by weekly medians, else 16.
    ranks["pass_rank"] = ranks.groupby("week")["pass_allowed"].rank(method="min", ascending=True)
    ranks["rush_rank"] = ranks.groupby("week")["rush_allowed"].rank(method="min", ascending=True)
    for col in ["pass_rank", "rush_rank"]:
        if ranks[col].notna().any():
            ranks[col] = ranks[col].fillna(ranks.groupby("week")[col].transform("median"))
        ranks[col] = ranks[col].fillna(16.0)

    return ranks

# Loader: fetch weekly stats & schedule; build robust per-player features for week.
def load_runtime_features(season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Empty: pre-shaped outputs with last_error in attrs for upstream logs.
    def _empty(err=None):
        feats = pd.DataFrame(columns=[
            "gsis_id","player_name","team","position","opponent","week",
            "is_home","rolling_fp_3","rolling_oppty_3","opp_pass_rank","opp_rush_rank",
            "vegas_total","weather_wind"
        ])
        sched = pd.DataFrame(columns=["game_id","week","home_team","away_team","total_line"])
        feats.attrs["last_error"] = err
        return feats, sched

    # Fetch: weekly player data (can 404 early off-season).
    try:
        weekly = nfl.import_weekly_data([season])
    except Exception as e:
        return _empty(e)

    weekly = weekly[weekly["week"].astype(int) <= int(week)].copy()

    # Select: keep stable, commonly-present columns.
    keep = [
        "player_id","player_name","recent_team","position","week","season",
        "opponent","home_away","game_id",
        "passing_yards","passing_tds","interceptions",
        "rushing_yards","rushing_tds","rushing_attempts",
        "receiving_yards","receiving_tds","receptions","targets","attempts"
    ]
    weekly = weekly[[c for c in keep if c in weekly.columns]].copy()
    weekly = weekly.rename(columns={"player_id": "gsis_id", "recent_team": "team"})

    # Fetch: schedule for current week (safe if missing).
    try:
        schedule = nfl.import_schedules([season])
    except Exception:
        schedule = pd.DataFrame(columns=["game_id","week","home_team","away_team","total_line"])

    sched_wk = schedule[
        schedule.get("week", pd.Series(dtype=int)).astype("Int64").fillna(-1).astype(int) == int(week)
    ].copy()
    sched_cols = ["game_id","week","home_team","away_team","total_line"]
    sched_wk = sched_wk[[c for c in sched_cols if c in sched_wk.columns]]

    # Join: derive is_home/opponent using weekly & schedule data.
    if "home_away" in weekly.columns and "opponent" in weekly.columns:
        weekly["is_home"] = weekly["home_away"].fillna("@").ne("@").astype(int)
        if "game_id" in weekly.columns and "game_id" in sched_wk.columns:
            weekly = weekly.merge(
                sched_wk[["game_id","home_team","away_team","total_line"]],
                on="game_id", how="left"
            )
        else:
            weekly = weekly.merge(
                sched_wk[["week","home_team","away_team","total_line"]],
                on="week", how="left"
            )
    else:
        if "game_id" in weekly.columns and "game_id" in sched_wk.columns:
            tmp = weekly.merge(
                sched_wk[["game_id","home_team","away_team","total_line"]],
                on="game_id", how="left"
            )
        else:
            tmp = weekly.merge(
                sched_wk[["week","home_team","away_team","total_line"]],
                on="week", how="left"
            )
        tmp["is_home"] = (tmp["team"] == tmp["home_team"]).astype(int)
        tmp["opponent"] = np.where(tmp["is_home"] == 1, tmp["away_team"], tmp["home_team"])
        weekly = tmp

    # Rolling: player 3-week fantasy & opportunity.
    weekly = weekly.sort_values(["gsis_id", "week"])
    weekly["fantasy"] = _fantasy_points(weekly)
    weekly["oppty"]   = _opportunity(weekly)
    weekly["rolling_fp_3"]    = weekly.groupby("gsis_id")["fantasy"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    weekly["rolling_oppty_3"] = weekly.groupby("gsis_id")["oppty"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    # Defense: build rolling allowed ranks and join to current week rows.
    ranks = _def_rank_allowed(weekly, weeks_window=5)
    wk = int(week)
    ranks_wk = ranks[ranks["week"] == wk][["defteam","pass_rank","rush_rank"]].rename(columns={"defteam": "opponent"})

    cur = weekly[weekly["week"] == wk].copy()
    cur = cur.merge(ranks_wk, on="opponent", how="left")
    cur["opp_pass_rank"] = pd.to_numeric(cur.get("pass_rank"), errors="coerce")
    cur["opp_rush_rank"] = pd.to_numeric(cur.get("rush_rank"), errors="coerce")
    cur["opp_pass_rank"] = cur["opp_pass_rank"].fillna(cur["opp_pass_rank"].median() if cur["opp_pass_rank"].notna().any() else 16.0)
    cur["opp_rush_rank"] = cur["opp_rush_rank"].fillna(cur["opp_rush_rank"].median() if cur["opp_rush_rank"].notna().any() else 16.0)

    # Vegas: team totals per game with safe median fallback.
    if "total_line" in sched_wk.columns and not sched_wk["total_line"].empty:
        _series = pd.to_numeric(sched_wk["total_line"], errors="coerce")
        total_med = float(_series.median()) if _series.notna().any() else 44.0
    else:
        total_med = 44.0

    tot_rows = []
    if not sched_wk.empty and {"home_team", "away_team"}.issubset(sched_wk.columns):
        for _, row in sched_wk.iterrows():
            tot = pd.to_numeric(row.get("total_line"), errors="coerce")
            if pd.isna(tot):
                tot = total_med
            tot_rows.append({"team": row.get("home_team"), "vegas_total": tot})
            tot_rows.append({"team": row.get("away_team"), "vegas_total": tot})

    team_totals = pd.DataFrame(tot_rows, columns=["team","vegas_total"]) if tot_rows else pd.DataFrame({"team": [], "vegas_total": []})
    cur = cur.merge(team_totals, on="team", how="left")
    cur["vegas_total"] = pd.to_numeric(cur.get("vegas_total"), errors="coerce").fillna(total_med)

    # Weather: placeholder for stadium wind (0 for now; wire in real API later).
    cur["weather_wind"] = 0.0

    # Output: one row per player for requested week with all runtime features.
    out = cur[[
        "gsis_id","player_name","team","position","opponent","week",
        "is_home","rolling_fp_3","rolling_oppty_3","opp_pass_rank","opp_rush_rank",
        "vegas_total","weather_wind"
    ]].drop_duplicates(["gsis_id", "week"]).reset_index(drop=True)

    return out, sched_wk.reset_index(drop=True)

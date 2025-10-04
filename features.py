import pandas as pd
import numpy as np
from pathlib import Path
from data_pipeline import load_pbp

# Robust feature builder for nflreadpy pbp schema (works across seasons/versions)
# - Detects the right ID/yard/TD columns
# - Builds per-player-week aggregates
# - Adds simple rolling features and opponent rank proxies

FEATURE_COLS = [
    "is_home", "opp_pass_rank", "opp_rush_rank",
    "rolling_fp_3", "rolling_oppty_3", "vegas_total", "weather_wind"
]


def _as_int(s: pd.Series) -> pd.Series:
    try:
        return s.fillna(0).astype(int)
    except Exception:
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _pick_col(df: pd.DataFrame, candidates: list[str], default=0):
    for c in candidates:
        if c in df.columns:
            return df[c]
    if isinstance(default, (int, float)):
        return pd.Series(default, index=df.index)
    return default


def compute_fantasy_points(agg: pd.DataFrame) -> pd.Series:
    # Half-PPR-ish starter scoring
    return (
        0.04 * _as_float(agg.get('passing_yards', 0))
        + 4.0 * _as_float(agg.get('passing_tds', 0))
        - 1.0 * _as_float(agg.get('interceptions', 0))
        + 0.10 * _as_float(agg.get('rushing_yards', 0))
        + 6.0 * _as_float(agg.get('rushing_tds', 0))
        + 0.10 * _as_float(agg.get('receiving_yards', 0))
        + 6.0 * _as_float(agg.get('receiving_tds', 0))
        + 0.50 * _as_float(agg.get('receptions', 0))
    )


def build_player_week_features(years, out_csv: str = "data/features_player_week.csv") -> str:
    pbp = load_pbp(years)

    # --- Identify player IDs on each play ---
    rusher_id = _pick_col(pbp, ['rusher_player_id', 'rusher_id'])
    receiver_id = _pick_col(pbp, ['receiver_player_id', 'receiver_id'])
    passer_id = _pick_col(pbp, ['passer_player_id', 'passer_id'])

    any_player_id = rusher_id.copy()
    any_player_id = any_player_id.combine_first(receiver_id)
    any_player_id = any_player_id.combine_first(passer_id)

    pbp = pbp.assign(any_player_id=any_player_id)
    pbp = pbp.dropna(subset=['any_player_id'])

    # --- Standardized team/season/week fields ---
    posteam = _pick_col(pbp, ['posteam', 'offense_team', 'off_team'])
    defteam = _pick_col(pbp, ['defteam', 'defense_team', 'def_team'])
    week = _pick_col(pbp, ['week', 'game_week']).astype(int)
    season = _pick_col(pbp, ['season', 'year']).astype(int)
    home_team = _pick_col(pbp, ['home_team'])
    away_team = _pick_col(pbp, ['away_team'])

    # --- Yardage / touchdowns / peripherals ---
    passing_yards = _as_float(_pick_col(pbp, ['passing_yards', 'pass_yards']))
    rush_yards = _as_float(_pick_col(pbp, ['rushing_yards', 'rush_yards']))
    rec_yards = _as_float(_pick_col(pbp, ['receiving_yards', 'rec_yards']))

    pass_td = _as_int(_pick_col(pbp, ['pass_touchdown', 'passing_tds']))
    rush_td = _as_int(_pick_col(pbp, ['rush_touchdown', 'rushing_tds']))
    rec_td = _as_int(_pick_col(pbp, ['receive_touchdown', 'receiving_tds']))

    interceptions = _as_int(_pick_col(pbp, ['interception', 'interceptions']))

    # Receptions: use 'reception' if present; otherwise fall back to completed passes to receiver is messy, so 0
    receptions = _as_int(_pick_col(pbp, ['reception', 'receptions'], default=0))

    df = pd.DataFrame({
        'any_player_id': any_player_id,
        'posteam': posteam,
        'defteam': defteam,
        'week': week,
        'season': season,
        'home_team': home_team,
        'away_team': away_team,
        'passing_yards': passing_yards,
        'rushing_yards': rush_yards,
        'receiving_yards': rec_yards,
        'passing_tds': pass_td,
        'rushing_tds': rush_td,
        'receiving_tds': rec_td,
        'interceptions': interceptions,
        'receptions': receptions,
    })

    grp_cols = ['any_player_id','week','season','posteam','defteam','home_team','away_team']
    agg = df.groupby(grp_cols, dropna=False).sum(numeric_only=True).reset_index()

    # Fantasy points
    agg['fantasy_points'] = compute_fantasy_points(agg)

    # Home flag
    agg['is_home'] = (agg['posteam'] == agg['home_team']).astype(float)

    # Opponent rank proxy: average FP allowed by defense within season
    def_allowed = agg.groupby(['defteam','season'], dropna=False)['fantasy_points'].mean().reset_index(name='fp_allowed')
    def_allowed['opp_rank'] = def_allowed.groupby('season')['fp_allowed'].rank(ascending=True)

    agg = agg.merge(def_allowed[['defteam','season','opp_rank']].rename(columns={'opp_rank':'opp_pass_rank'}), on=['defteam','season'], how='left')
    agg['opp_rush_rank'] = agg['opp_pass_rank']

    # Rolling features per player within season
    agg = agg.sort_values(['any_player_id','season','week'])
    agg['rolling_fp_3'] = agg.groupby(['any_player_id','season'])['fantasy_points'].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # Opportunity proxy: receptions + scaled rush yards
    agg['oppty'] = agg['receptions'].fillna(0) + agg['rushing_yards'].fillna(0)/5.0
    agg['rolling_oppty_3'] = agg.groupby(['any_player_id','season'])['oppty'].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # Placeholders for now
    agg['vegas_total'] = 45.0
    agg['weather_wind'] = 5.0

    out = agg[['any_player_id','season','week','posteam','defteam','is_home','opp_pass_rank','opp_rush_rank','rolling_fp_3','rolling_oppty_3','vegas_total','weather_wind','fantasy_points']].rename(columns={'any_player_id':'player_id'})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv
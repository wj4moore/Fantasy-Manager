# injury_rules.py

# Imports: core types and DataFrame.
from __future__ import annotations
from typing import Optional, Dict
import pandas as pd

# Defaults: game-status multipliers (cap risk into 0–1).
DEFAULT_STATUS_MULTIPLIER: Dict[Optional[str], float] = {
    "IR": 0.00,
    "OUT": 0.00,
    "SUSPENDED": 0.00,
    "DOUBTFUL": 0.10,
    "QUESTIONABLE": 0.85,
    "PROBABLE": 0.95,  # legacy
    "ACTIVE": 1.00,
    "NA": 1.00,
    None: 1.00,
}

# Defaults: practice-status multipliers (small tweaks).
DEFAULT_PRACTICE_MULTIPLIER: Dict[Optional[str], float] = {
    "DNP": 0.85,
    "LP": 0.95,  # Limited Participant
    "FP": 1.00,  # Full Participant
    None: 1.00,
}

# Utils: normalize a string to UPPER or None.
def _norm(s):
    if s is None:
        return None
    return str(s).strip().upper()

# Seed: pull best-effort status hints from Sleeper players_map.
def infer_status_from_players_map(players_map: dict, player_id: str) -> dict:
    """
    Returns a dict with:
      - game_status: OUT/QUESTIONABLE/etc (normalized)
      - practice_status: None (not present in players_map)
    """
    meta = players_map.get(player_id, {}) if isinstance(players_map, dict) else {}
    inj = _norm(meta.get("injury_status"))

    # Shorthand → full tag.
    short = {"Q": "QUESTIONABLE", "D": "DOUBTFUL", "O": "OUT"}
    inj = short.get(inj, inj)

    # Use 'status' as a fallback (Active/Inactive).
    status = _norm(meta.get("status"))
    if status == "INACTIVE" and inj is None:
        inj = "OUT"
    if status == "ACTIVE" and inj is None:
        inj = "ACTIVE"

    return {"game_status": inj, "practice_status": None}

# Loader: optional CSV to override/augment statuses.
def load_injury_csv(path: str) -> pd.DataFrame:
    """
    CSV columns (case-insensitive): player_id, game_status?, practice_status?
    Unknown columns are ignored.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "player_id" not in cols:
        raise ValueError("injury CSV must include at least 'player_id' column")

    out = pd.DataFrame({"player_id": df[cols["player_id"]].astype(str)})
    out["game_status"] = df[cols.get("game_status")].map(_norm) if "game_status" in cols else None
    out["practice_status"] = df[cols.get("practice_status")].map(_norm) if "practice_status" in cols else None
    return out

# Main: apply injury multipliers to projections and return adjusted values.
def apply_injury_adjustments(
    df_proj: pd.DataFrame,
    players_map: Optional[dict] = None,
    injury_csv: Optional[str] = None,
    status_multiplier: Optional[Dict[Optional[str], float]] = None,
    practice_multiplier: Optional[Dict[Optional[str], float]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Requires columns: ['player_id','position','projected'].
    Adds: 'injury_game_status','injury_practice_status','injury_mult','projected_adj'.
    Precedence: CSV overrides players_map; else assume ACTIVE (1.0).
    """
    status_multiplier = status_multiplier or DEFAULT_STATUS_MULTIPLIER
    practice_multiplier = practice_multiplier or DEFAULT_PRACTICE_MULTIPLIER

    df = df_proj.copy()
    df["injury_game_status"] = None
    df["injury_practice_status"] = None

    # Seed: players_map (if provided).
    if players_map:
        seeds = df["player_id"].apply(lambda pid: infer_status_from_players_map(players_map, pid))
        df["injury_game_status"] = seeds.map(lambda d: d.get("game_status"))
        df["injury_practice_status"] = seeds.map(lambda d: d.get("practice_status"))

    # Override: CSV (if provided).
    if injury_csv:
        try:
            inj = load_injury_csv(injury_csv)
            df = df.merge(inj, on="player_id", how="left", suffixes=("", "_csv"))
            # Prefer CSV values.
            if "game_status" in df.columns:
                df["injury_game_status"] = df["game_status"].combine_first(df["injury_game_status"])
            if "practice_status" in df.columns:
                df["injury_practice_status"] = df["practice_status"].combine_first(df["injury_practice_status"])
            df.drop(columns=[c for c in ["game_status", "practice_status"] if c in df.columns], inplace=True)
        except Exception as e:
            if verbose:
                print("[injury] CSV load failed; using players_map only:", e)

    # Map: normalized statuses → multipliers.
    df["injury_game_status"] = df["injury_game_status"].map(_norm)
    df["injury_practice_status"] = df["injury_practice_status"].map(_norm)

    df["injury_mult"] = df["injury_game_status"].map(status_multiplier).fillna(1.0)
    df["injury_mult"] *= df["injury_practice_status"].map(practice_multiplier).fillna(1.0)
    df["injury_mult"] = df["injury_mult"].clip(0.0, 1.0)  # cap to [0,1]

    # Apply: only skill positions get reduced unless explicitly OUT.
    df["projected_adj"] = df["projected"]
    mask_players = df["position"].isin(["QB", "RB", "WR", "TE"])
    df.loc[mask_players, "projected_adj"] = df.loc[mask_players, "projected"] * df.loc[mask_players, "injury_mult"]

    # Log: show impacted rows for clarity.
    if verbose:
        flagged = df[(df["injury_mult"] < 1.0) & mask_players][
            ["name", "position", "projected", "injury_game_status", "injury_practice_status", "injury_mult"]
        ]
        if not flagged.empty:
            print("\n[injury] Adjusted projections based on status/practice:")
            print(flagged.sort_values("injury_mult").to_string(index=False))

    return df

# id_map.py

# Imports: basics for DataFrame and name cleanup.
from __future__ import annotations
import pandas as pd
import re

# Utils: normalize a player's name to a lowercase, alphanumeric key.
def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Builder: create Sleeper ↔ GSIS mapping with a robust name_key fallback.
def build_id_map_from_sleeper(players_map: dict) -> pd.DataFrame:
    """
    Output columns:
      ['sleeper_id','full_name','position','team','gsis_id','name_key']
    Strategy:
      - Prefer gsis_id when Sleeper exposes it (gsis_id or gsisId).
      - Always include name_key = "<norm_name>|<POS>" as a fallback join key.
    """
    rows = []
    for pid, meta in (players_map or {}).items():
        if not isinstance(meta, dict):
            continue

        first = meta.get("first_name", "") or ""
        last  = meta.get("last_name", "") or ""
        full_name = (meta.get("full_name") or f"{first} {last}").strip()

        pos = (meta.get("position") or "").upper()
        tm  = meta.get("team")
        gsis = meta.get("gsis_id") or meta.get("gsisId")

        rows.append({
            "sleeper_id": pid,
            "full_name": full_name,
            "position": pos,
            "team": tm,
            "gsis_id": gsis,
            "name_key": f"{_norm_name(full_name)}|{pos}",
        })

    return pd.DataFrame(rows).drop_duplicates(subset=["sleeper_id"], keep="first")

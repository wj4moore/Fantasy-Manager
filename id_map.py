# id_map.py
from __future__ import annotations
import pandas as pd
import re

def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_id_map_from_sleeper(players_map: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['sleeper_id','full_name','position','team','gsis_id','name_key']
    Uses gsis_id if Sleeper provides it; else name+position key for joining.
    """
    rows = []
    for pid, meta in players_map.items():
        if not isinstance(meta, dict):
            continue
        full_name = meta.get("full_name") or f"{meta.get('first_name','?')} {meta.get('last_name','')}"
        pos = (meta.get("position") or "").upper()
        tm  = meta.get("team")
        gsis = meta.get("gsis_id") or meta.get("gsisId")  # Sleeper sometimes varies
        rows.append({
            "sleeper_id": pid,
            "full_name": (full_name or "").strip(),
            "position": pos,
            "team": tm,
            "gsis_id": gsis,
            "name_key": f"{_norm_name(full_name)}|{pos}",
        })
    return pd.DataFrame(rows).drop_duplicates("sleeper_id")

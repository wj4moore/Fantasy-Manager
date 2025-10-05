# data_pipeline.py

# Loader: Pandas-only PBP loader via nflreadpy[pandas]. Saves a parquet snapshot per year range.
# Install: pip install "nflreadpy[pandas]"  (and for parquet: pip install pyarrow)

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any

import pandas as pd


# Loader: Read play-by-play for given seasons and persist a parquet cache.
def load_pbp(years: Iterable[int], out_dir: str = "data/pbp") -> pd.DataFrame:
    import nflreadpy as nr  # local import to keep global deps light

    frames: list[pd.DataFrame] = []
    for y in years:
        df = nr.load_pbp(seasons=[int(y)])  # nflreadpy returns pandas (or polars if installed)
        # Utils: Convert polars → pandas if needed, without requiring polars at install time.
        try:
            import polars as pl  # optional
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
        except Exception:
            pass

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)  # Guardrail: coerce to pandas

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Utils: Ensure safe column names for parquet (must be strings).
    out.columns = out.columns.map(str)

    # IO: Write snapshot to parquet (requires pyarrow or fastparquet installed).
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"pbp_{min(years)}_{max(years)}.parquet"
    out.to_parquet(out_path, index=False)

    return out


# Weather: Historical point lookup from OpenWeather (optional helper).
# Usage: get_weather_for_game("2024-10-06T17:00:00Z", (40.4468, -80.0158), os.environ["OWM_KEY"])
def get_weather_for_game(game_datetime_utc: str, stadium_coords: Tuple[float, float], owm_key: str) -> Dict[str, Any]:
    import requests  # local import so 'requests' isn’t a hard dependency if unused

    lat, lon = stadium_coords
    ts = int(pd.to_datetime(game_datetime_utc, utc=True).timestamp())
    url = (
        "https://api.openweathermap.org/data/2.5/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={ts}&appid={owm_key}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

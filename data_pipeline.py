# Clean, pandas-only loader using nflreadpy[pandas].
# There is intentionally NO manual HTTP fallback here to avoid 404 churn.
# Make sure you've run:  pip install "nflreadpy[pandas]"

# data_pipeline.py
import pandas as pd
from pathlib import Path

def load_pbp(years, out_dir: str = "data/pbp") -> pd.DataFrame:
    import nflreadpy as nr
    frames = []
    for y in years:
        _df = nr.load_pbp(seasons=[y])  # could be pandas OR polars depending on install
        try:
            import polars as pl  # may or may not be installed
            if isinstance(_df, pl.DataFrame):
                _df = _df.to_pandas()
        except Exception:
            # polars not installed or not a polars df — assume pandas already
            pass
        if not isinstance(_df, pd.DataFrame):
            # last-resort guardrail
            _df = pd.DataFrame(_df)
        frames.append(_df)
    out = pd.concat(frames, ignore_index=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # ensure parquet-safe column names
    out.columns = out.columns.map(str)
    out.to_parquet(Path(out_dir) / f"pbp_{min(years)}_{max(years)}.parquet", index=False)

    return out


# Weather fetcher using OpenWeatherMap (user supplies key)
# def get_weather_for_game(game_datetime_utc: str, stadium_coords: tuple, owm_key: str):
    # stadium_coords = (lat, lon)
    lat, lon = stadium_coords
    ts = int(pd.to_datetime(game_datetime_utc).timestamp())
    url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={ts}&appid={owm_key}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()
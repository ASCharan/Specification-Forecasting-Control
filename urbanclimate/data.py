"""CPCB dataset construction, quality control and feature engineering.

The public mirror (Vonter/india-cpcb-aqi) stores hourly AQI in wide format:
one row per (station, date) with 24 hourly columns. The `Station ID` column in
that mirror is not unique within a city; `Station Name` is the real station
key, and this module uses it accordingly.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_URL = (
    "https://raw.githubusercontent.com/Vonter/india-cpcb-aqi/main/"
    "data/cpcb-aqi.parquet"
)
HOUR_COLS = [f"{h:02d}:00:00" for h in range(24)]

# Lag convention. LAG_OFFSET = 1 means "lag k is series[t-k+1]", so lag 1 is
# series[t]: at prediction time t the model knows the series up to and
# including t and predicts series[t+horizon]. This never leaks and it is the
# only convention used for reported results. LAG_OFFSET = 0 withholds the most
# recent observation and exists ONLY for the diagnostic of Table 2; mixing the
# two across rows of one table makes the comparison invalid.
LAG_OFFSET = 1

# A uniform threshold of five reporting stations per hour reproduces all three
# series reported in the manuscript exactly (hours, span, mean AQI, ACF).
CITY_SPEC = {
    "Delhi": dict(min_stations=5),
    "Mumbai": dict(min_stations=5),
    "Kolkata": dict(min_stations=5),
}


def download(raw_dir="data/raw") -> Path:
    """Fetch the hourly AQI parquet if not already on disk (about 15 MB)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "cpcb-aqi.parquet"
    if not dest.exists():
        print(f"downloading {PARQUET_URL} -> {dest}", flush=True)
        urllib.request.urlretrieve(PARQUET_URL, dest)
    return dest


def build_city_series(
    city: str,
    raw_dir="data/raw",
    min_stations: int | None = None,
    clip=(0.0, 1000.0),
    max_gap: int = 3,
) -> pd.Series:
    """Hourly cross-station median AQI for one city, quality controlled.

    Melts to long format, drops values outside `clip`, takes the cross-station
    median per hour subject to at least `min_stations` reporting stations,
    interpolates gaps of at most `max_gap` hours, and returns the longest
    contiguous run with no remaining missing values.
    """
    if min_stations is None:
        min_stations = CITY_SPEC.get(city, {}).get("min_stations", 5)

    df = pd.read_parquet(download(raw_dir))
    df = df[df["City"] == city]
    if df.empty:
        raise ValueError(f"no rows for city {city!r}")

    long = df.melt(
        id_vars=["Station Name", "Date"],
        value_vars=HOUR_COLS,
        var_name="hour",
        value_name="aqi",
    )
    long["aqi"] = pd.to_numeric(long["aqi"], errors="coerce")
    long = long[long["aqi"].between(*clip)]
    long["ts"] = pd.to_datetime(long["Date"]) + pd.to_timedelta(
        long["hour"].str.slice(0, 2).astype(int), unit="h"
    )

    grouped = long.groupby("ts")["aqi"]
    series = grouped.median()
    counts = grouped.size()
    series = series[counts >= min_stations]

    full = pd.date_range(series.index.min(), series.index.max(), freq="h")
    series = series.reindex(full)
    series = series.interpolate(limit=max_gap, limit_area="inside")
    return _longest_contiguous(series)


def _longest_contiguous(s: pd.Series) -> pd.Series:
    """Longest run of consecutive non-missing hourly values."""
    ok = s.notna().to_numpy()
    best_len = best_start = cur_start = cur_len = 0
    for i, flag in enumerate(ok):
        if flag:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    out = s.iloc[best_start : best_start + best_len]
    out.name = "aqi"
    return out


def synthetic_exogenous(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    """Diurnal templates for meteorology and traffic (Methods, Eqs. 2 to 4).

    These are deterministic functions of hour-of-day plus i.i.d. noise. They are
    not measurements, which is what limits the interpretation of the input
    ablation.
    """
    rng = np.random.default_rng(seed)
    t = index.hour.to_numpy().astype(float)
    n = len(index)

    temp = 28 + 8 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, 1.5, n)
    rh = 60 - 15 * np.sin(2 * np.pi * (t - 12) / 24) + rng.normal(0, 5.0, n)
    wind = 3.5 + 1.5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.8, n)

    morning = np.exp(-0.5 * ((t - 8.5) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((t - 18.5) / 2.5) ** 2)
    traffic = 73.0 * (0.25 + 0.75 * (morning + 0.95 * evening))
    traffic = traffic + rng.normal(0, 2.0, n)

    return pd.DataFrame(
        {"temp": temp, "rh": rh, "wind": wind, "traffic": traffic}, index=index
    )


def make_features(
    series: pd.Series,
    horizon: int = 1,
    lags=(1, 2, 3, 24),
    seed: int = 0,
    use_met: bool = True,
    use_traffic: bool = True,
    lag_offset: int = LAG_OFFSET,
):
    """Tabular design matrix and target for horizon-h prediction.

    `lag_offset` must be identical for every model in a comparison. See the
    module-level LAG_OFFSET note.
    """
    if lag_offset not in (0, 1):
        raise ValueError("lag_offset must be 0 or 1")
    exo = synthetic_exogenous(series.index, seed=seed)
    cols = {}
    if use_met:
        cols.update({c: exo[c] for c in ("temp", "rh", "wind")})
    if use_traffic:
        cols["traffic"] = exo["traffic"]
    # Lags are indexed from PREDICTION time, not target time: at row t the
    # model knows series up to and including series[t] and predicts
    # series[t+horizon], so lag 1 is series[t]. This is independent of the
    # horizon and never leaks. At h=1 it makes aqi_lag1 exactly the
    # persistence forecast and aqi_lag24 the seasonal-naive forecast; at
    # h=24 persistence and seasonal-naive coincide, both equal to aqi_lag1.
    # lag_offset=0 reproduces the shifted convention (lag 1 = series[t-1]) for
    # diagnostic comparison; it withholds the most recent observation.
    for lag in lags:
        cols[f"aqi_lag{lag}"] = series.shift(lag - lag_offset)

    X = pd.DataFrame(cols, index=series.index)
    y = series.shift(-horizon)
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep]


def chrono_split(n: int, fracs=(0.6, 0.2, 0.2)):
    """Chronological train/validation/test index ranges."""
    a = int(n * fracs[0])
    b = a + int(n * fracs[1])
    return slice(0, a), slice(a, b), slice(b, n)


def describe(series: pd.Series) -> dict:
    v = series.to_numpy(float)
    return {
        "hours": int(len(v)),
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "mean_aqi": round(float(v.mean()), 1),
        "acf_lag1": round(float(np.corrcoef(v[:-1], v[1:])[0, 1]), 3),
        "acf_lag24": round(float(np.corrcoef(v[:-24], v[24:])[0, 1]), 3),
    }

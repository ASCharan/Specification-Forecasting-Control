#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbanclimate import data as D

CITIES = ["Delhi", "Mumbai", "Kolkata"]
OUT = Path("results/r1_data_audit.json")


def long_frame(city, raw_dir="data/raw"):
    df = pd.read_parquet(D.download(raw_dir))
    df = df[df["City"] == city]
    long = df.melt(id_vars=["Station Name", "Date"], value_vars=D.HOUR_COLS,
                   var_name="hour", value_name="aqi")
    long["aqi"] = pd.to_numeric(long["aqi"], errors="coerce")
    long = long[long["aqi"].between(0.0, 1000.0)]
    long["ts"] = pd.to_datetime(long["Date"]) + pd.to_timedelta(
        long["hour"].str.slice(0, 2).astype(int), unit="h")
    return long


def audit(city, min_stations=5, max_gap=3):
    long = long_frame(city)
    grouped = long.groupby("ts")["aqi"]
    med, cnt = grouped.median(), grouped.size()

    kept = med[cnt >= min_stations]
    full = pd.date_range(kept.index.min(), kept.index.max(), freq="h")
    raw = kept.reindex(full)                      # NaN where dropped or absent
    filled = raw.interpolate(limit=max_gap, limit_area="inside")
    final = D._longest_contiguous(filled)

    # which rows of the retained span were interpolated rather than observed
    interp_mask = raw.reindex(final.index).isna()
    n_interp = int(interp_mask.sum())

    # gap-length distribution among the interpolated rows
    runs, cur = [], 0
    for f in interp_mask.to_numpy():
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    gap_hist = {int(k): int(v) for k, v in
                zip(*np.unique(runs, return_counts=True))} if runs else {}

    # station availability over the retained span
    cnt_span = cnt.reindex(final.index)
    # station counts are only meaningful on rows that were actually observed:
    # the interpolated rows are exactly those that fell below the threshold
    obs_idx = final.index[~raw.reindex(final.index).isna()]
    cnt_obs = cnt.reindex(obs_idx)
    monthly = cnt_obs.groupby(cnt_obs.index.to_period("M")).agg(
        ["min", "median", "max"])

    # hours in the raw record that fell below the threshold
    below = int((cnt < min_stations).sum())

    v = final.to_numpy(float)
    obs = ~interp_mask.to_numpy()

    def acf_clean(lag):
        """lag-k autocorrelation using only pairs where BOTH ends are observed."""
        a, b = v[:-lag], v[lag:]
        ok = obs[:-lag] & obs[lag:]
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    # balanced panel: stations reporting in >=90% of the retained hours
    span = long[(long["ts"] >= final.index[0]) & (long["ts"] <= final.index[-1])]
    cover = span.groupby("Station Name")["ts"].nunique() / len(final)
    stable = sorted(cover[cover >= 0.90].index.tolist())

    return {
        "city": city,
        "n_stations_total": int(long["Station Name"].nunique()),
        "span_start": str(final.index[0]),
        "span_end": str(final.index[-1]),
        "hours": int(len(final)),
        "n_interpolated": n_interp,
        "pct_interpolated": round(100 * n_interp / len(final), 2),
        "gap_length_hist": gap_hist,
        "hours_below_threshold": below,
        "stations_min": int(cnt_obs.min()),
        "stations_median": float(cnt_obs.median()),
        "stations_max": int(cnt_obs.max()),
        "monthly_station_median_min": int(monthly["median"].min()),
        "monthly_station_median_max": int(monthly["median"].max()),
        "monthly_station_min_overall": int(monthly["min"].min()),
        "acf1_full": round(float(np.corrcoef(v[:-1], v[1:])[0, 1]), 4),
        "acf1_observed_only": round(acf_clean(1), 4),
        "acf24_full": round(float(np.corrcoef(v[:-24], v[24:])[0, 1]), 4),
        "acf24_observed_only": round(acf_clean(24), 4),
        "n_stable_stations_90pct": len(stable),
        "stable_stations": stable,
    }, final, interp_mask, span, stable


def balanced_target(span, stable, min_stations, index):
    """City median restricted to the stable station panel, same QC rules."""
    sub = span[span["Station Name"].isin(stable)]
    g = sub.groupby("ts")["aqi"]
    med, cnt = g.median(), g.size()
    thr = min(min_stations, len(stable))
    kept = med[cnt >= thr]
    s = kept.reindex(index).interpolate(limit=3, limit_area="inside")
    return D._longest_contiguous(s), thr


def main():
    out, tables = {}, {}
    for city in CITIES:
        a, final, interp_mask, span, stable = audit(city)
        out[city] = a
        print(f"{city}: {a['hours']} h, {a['n_interpolated']} interpolated "
              f"({a['pct_interpolated']}%), gaps {a['gap_length_hist']}, "
              f"stations {a['stations_min']}-{a['stations_max']} "
              f"(median {a['stations_median']}), "
              f"acf1 {a['acf1_full']} -> {a['acf1_observed_only']} on observed "
              f"pairs, stable panel {a['n_stable_stations_90pct']}", flush=True)

        # write the interpolation mask and the balanced-panel series for re-runs
        pd.Series(interp_mask.to_numpy(), index=final.index,
                  name="interpolated").to_csv(
            f"data/processed/{city.lower()}_interp_mask.csv")
        bal, thr = balanced_target(span, stable, 5, final.index)
        out[city]["balanced_threshold"] = thr
        out[city]["balanced_hours"] = int(len(bal))
        bv = bal.to_numpy(float)
        out[city]["balanced_acf1"] = round(
            float(np.corrcoef(bv[:-1], bv[1:])[0, 1]), 4)
        out[city]["balanced_mean_aqi"] = round(float(bv.mean()), 1)
        bal.rename("aqi").to_csv(f"data/processed/{city.lower()}_aqi_balanced.csv")
        print(f"   balanced panel: {len(stable)} stations, threshold {thr}, "
              f"{len(bal)} h, mean {out[city]['balanced_mean_aqi']}, "
              f"acf1 {out[city]['balanced_acf1']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

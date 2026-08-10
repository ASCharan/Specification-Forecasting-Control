#!/usr/bin/env python3
"""Build the quality-controlled city series and print their descriptive stats.

Usage:  python scripts/01_build_dataset.py [--cities Delhi Mumbai Kolkata]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import data
from urbanclimate.utils import save_json, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["Delhi", "Mumbai", "Kolkata"])
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, times = {}, {}

    for city in args.cities:
        with timer(f"build {city}", times):
            series = data.build_city_series(city, raw_dir=args.raw_dir)
        series.to_csv(out / f"{city.lower()}_aqi.csv")
        summary[city] = data.describe(series)
        print(f"{city}: {summary[city]}", flush=True)

    save_json({"summary": summary, "seconds": times}, "results/dataset_summary.json")
    print("\nwrote data/processed/*.csv and results/dataset_summary.json")


if __name__ == "__main__":
    main()

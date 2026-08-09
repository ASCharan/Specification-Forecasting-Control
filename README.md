**UrbanClimate-RL** 

* Forecasting - hourly AQI prediction for Delhi, Mumbai and Kolkata at 1 h and 24 h horizons, comparing a Neural-ODE forecaster (C-Mod) against ten baselines on real CPCB observations.
* Control - a PPO–GAE agent acting on signal timing, speed limits and emission-zone enforcement inside a coupled traffic–pollution simulator.

---

## Quick start

```bash
git clone <repo-url> && cd urbanclimate-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q          # 5 tests, ~5 s
python scripts/01_build_dataset.py  # downloads 15 MB, ~3 s
python scripts/02_run_forecasting.py --city Delhi --horizon 1  --seeds 0 1 2
python scripts/02_run_forecasting.py --city Delhi --horizon 24 --seeds 0 1 2 3 4
python scripts/03_train_ppo.py --seeds 0 1 2 --episodes 300
python scripts/06_ablation.py --city Delhi --horizon 1 --seeds 0 1 2
python scripts/07_paired_bootstrap.py --city Delhi --horizon 24 --all
python scripts/08_make_figures.py --outdir figures
python scripts/12_manuscript_figures.py --outdir figures
python scripts/09_reward_diagnostics.py
python scripts/10_traffic_informativeness.py --city Delhi --horizon 24
```

## Repository layout

```
urbanclimate/
  data.py          CPCB download, QC, synthetic exogenous inputs, features
  models.py        C-Mod (Neural ODE + RK4), MLP, LSTM, Transformer
  baselines.py     climatology, persistence, seasonal-naive, ridge, SARIMA(X)
  forecasting.py   shared training loop, early stopping, checkpointing
  envs.py          pollutant dynamics simulator (Eq. 1), fixed policies
  ppo.py           PPO with GAE, from scratch; policy evaluation; seed namespaces
  utils.py         seeding, timing, metrics, paired bootstrap
scripts/
  01_build_dataset.py           build and describe the three city series
  02_run_forecasting.py         full model comparison for one city/horizon
                                (all ten models, both horizons)
  03_train_ppo.py               train controllers, compare against fixed rules
  04_reward_sensitivity.py      reproduce the regime-selection result
  05_dispersion_sensitivity.py  robustness of Eq. 1 to the dispersion closure
tests/test_smoke.py             fast end-to-end checks
configs/default.yaml            every hyperparameter in the Methods
```

Checkpoints land in `checkpoints/<city>/h<horizon>/<model>_seed<n>.pt` and `checkpoints/ppo/ppo_seed<n>.pt`. Each stores the state dict, the config that produced it, the test metrics and the validation history, so a checkpoint is self-describing.

---

## Data

Hourly AQI comes from the Central Pollution Control Board of India via the public mirror [Vonter/india-cpcb-aqi (https://github.com/Vonter/india-cpcb-aqi) (ODbL). `scripts/01_build_dataset.py` downloads a 15 MB parquet and applies the quality control described in the Methods: cross-station median per hour, at least five reporting stations, values clipped to [0, 1000], gaps of at most three hours interpolated, longest contiguous span retained.

That recipe reproduces all three series exactly:

| City | Hours | Span | Mean AQI | ACF(1) | ACF(24) |
|---|---|---|---|---|---|
| Delhi | 19,071 | 2021-05-11 → 2023-07-15 | 198.2 | 0.976 | 0.807 |
| Mumbai | 28,209 | 2020-10-12 → 2023-12-31 | 114.4 | 0.977 | 0.841 |
| Kolkata | 12,348 | 2022-08-04 → 2023-12-31 | 110.6 | 0.987 | 0.884 |

Two quirks of the mirror are handled in `data.py` and are worth knowing about:

1. The `Station ID` column is **not** unique within a city — every Delhi row carries `site_115`. `Station Name` is the real station key (39 in Delhi).
2. The file is in wide format: one row per (station, date) with 24 hourly columns, which must be melted before aggregation.

Meteorology and traffic are synthetic diurnal templates, not measurements. They are deterministic functions of hour-of-day plus noise, which is why the input ablation measures the value of a diurnal encoding rather than the value of weather or traffic information.

---

## Lag indexing

Lags are indexed from prediction time, not target time. At row `t`, the model knows the series up to and including `series[t]` and predicts `series[t+h]`, so `aqi_lag1 = series[t]`.

This matters more than it sounds. Under this convention `aqi_lag1` is exactly the persistence forecast at h=1, which lets the naive baselines be validated against the learned models on identical inputs. Shifting by one more step silently withholds the most recent observation from the tabular models while the windowed sequence models still see it, and the comparison stops being fair. The correctness check is in `tests/test_smoke.py`; a quick manual version is that `persistence` must equal the `aqi_lag1` column at h=1.

---

## Citation

```bibtex
@article{saicharan2026urbanclimaterl,
  title  = {Specification, not architecture, governs learned\\
air-quality forecasting and control},
  author = {Sai Charan, A and Harshitha, Sunkara and Debnath, Ramit},
  year   = {2026}
}
```

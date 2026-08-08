# UrbanClimate-RL

Reference implementation for *Forecast horizon governs which air-quality models
are useful for urban pollution control*.

> **Read `CRITICAL_ANALYSIS.md` before using any number from an earlier
> version of this repository.** The lag convention, the scoring rows and the
> PPO seed namespaces were all corrected on 1 August 2026, and results produced
> before that date are not comparable with results produced after it.

Two arms, evaluated independently:

* **Forecasting** — hourly AQI prediction for Delhi, Mumbai and Kolkata at 1 h
  and 24 h horizons, comparing a Neural-ODE forecaster (C-Mod) against ten
  baselines on real CPCB observations.
* **Control** — a PPO–GAE agent acting on signal timing, speed limits and
  emission-zone enforcement inside a coupled traffic–pollution simulator.

The learned forecaster is **not** inside the control loop. The reward's
forecast-error term uses persistence within the simulator, and because the
weather fields are exogenous that term is invariant to the action: it shifts
the return by a constant and cannot shape the policy. Run
`scripts/09_reward_diagnostics.py` to confirm.

There is **no computer-vision component** in this study. The traffic input is a
synthetic diurnal template throughout, and no camera, detector or tracker is
used anywhere in the pipeline. `scripts/10_traffic_informativeness.py` instead
bounds how informative a *measured* traffic series would have to be before it
could change any result reported here.

## Three invariants

1. **One lag convention per table.** `urbanclimate.data.LAG_OFFSET` is 1: lag
   *k* is `series[t-k+1]`, so lag 1 is the most recent observation. Passing
   `--lag-offset 0` prints a warning and is for the diagnostic table only.
   Mixing the two across rows of one comparison inflates the apparent gap
   between tabular and sequence models by roughly 12 RMSE points.
2. **One set of scoring rows.** `--align-test-rows` (default on) scores every
   model on the rows the 24-hour sequence window can reach.
3. **Disjoint training and evaluation episode seeds.** Enforced by
   `urbanclimate.ppo.assert_disjoint_seeds`.

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

Or `make all`. For the full submission sequence see `RERUN_MANIFEST.md`.

---

## "168 hours" and "72 hours" are simulated time, not compute time

This is the single most common misreading of the figures, so it is worth being
explicit.

| Where it appears | What the number means |
|---|---|
| Fig. 6, "300 episodes of 168 hours" | Each RL episode simulates **one week of hourly time steps** (168 steps). Training runs 300 such episodes, i.e. 50,400 environment steps. |
| Fig. 3, "final 72 hours" | A **three-day window of the test split** is plotted so the diurnal cycle is legible. Nothing is trained for 72 hours. |
| Config `episode_hours: 168` | Simulator steps per episode. |

Nothing in this project takes hours of wall clock. Measured on **one CPU core**
(no GPU, `OMP_NUM_THREADS=1`), the numbers below are from an actual run:

| Stage | Wall clock |
|---|---|
| Build all three city series from the raw parquet | 3 s |
| Ridge / climatology / persistence / seasonal-naive | < 1 s |
| C-Mod, one seed, Delhi h=1 (19,071 hours of data) | 21–35 s |
| MLP-64 / MLP-128, one seed | 7–17 s |
| LSTM, one seed | 11–32 s |
| **PPO, one seed, 300 episodes × 168 h = 50,400 steps** | **57 s** |
| Reward-sensitivity experiment (2 controllers) | ~2 min |
| Dispersion-sensitivity experiment | ~1 min |

Seasonal ARIMA at h=24 is the one genuinely slow component: rolling-origin
24-step forecasts at every test origin means ~3,800 model evaluations for
Delhi, which takes tens of minutes on one core. Everything else is minutes.

A full replication — 3 cities × 2 horizons × 5 models × up to 5 seeds, plus
3 PPO seeds and both sensitivity studies — is a few hours single-core and well
under an hour on any multi-core machine.

---

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

Checkpoints land in `checkpoints/<city>/h<horizon>/<model>_seed<n>.pt` and
`checkpoints/ppo/ppo_seed<n>.pt`. Each stores the state dict, the config that
produced it, the test metrics and the validation history, so a checkpoint is
self-describing.

---

## Data

Hourly AQI comes from the Central Pollution Control Board of India via the
public mirror [Vonter/india-cpcb-aqi](https://github.com/Vonter/india-cpcb-aqi)
(ODbL). `scripts/01_build_dataset.py` downloads a 15 MB parquet and applies the
quality control described in the Methods: cross-station median per hour,
at least five reporting stations, values clipped to [0, 1000], gaps of at most
three hours interpolated, longest contiguous span retained.

That recipe reproduces all three series exactly:

| City | Hours | Span | Mean AQI | ACF(1) | ACF(24) |
|---|---|---|---|---|---|
| Delhi | 19,071 | 2021-05-11 → 2023-07-15 | 198.2 | 0.976 | 0.807 |
| Mumbai | 28,209 | 2020-10-12 → 2023-12-31 | 114.4 | 0.977 | 0.841 |
| Kolkata | 12,348 | 2022-08-04 → 2023-12-31 | 110.6 | 0.987 | 0.884 |

Two quirks of the mirror are handled in `data.py` and are worth knowing about:

1. The `Station ID` column is **not** unique within a city — every Delhi row
   carries `site_115`. `Station Name` is the real station key (39 in Delhi).
2. The file is in wide format: one row per (station, date) with 24 hourly
   columns, which must be melted before aggregation.

Meteorology and traffic are **synthetic diurnal templates**, not measurements.
They are deterministic functions of hour-of-day plus noise, which is why the
input ablation measures the value of a diurnal encoding rather than the value
of weather or traffic information.

---

## Lag indexing

Lags are indexed from **prediction** time, not target time. At row `t` the model
knows the series up to and including `series[t]` and predicts `series[t+h]`, so
`aqi_lag1 = series[t]`.

This matters more than it sounds. Under this convention `aqi_lag1` is exactly
the persistence forecast at h=1, which lets the naive baselines be validated
against the learned models on identical inputs. Shifting by one more step
silently withholds the most recent observation from the tabular models while
the windowed sequence models still see it, and the comparison stops being fair.
The correctness check is in `tests/test_smoke.py`; a quick manual version is
that `persistence` must equal the `aqi_lag1` column at h=1.

---

## Dispersion closure

Eq. 1 writes dispersive removal as `β·exp(-κh)`, which *decreases* as the mixing
layer deepens. Boundary-layer physics runs the other way. `envs.py` implements
both that closure (`closure="exp"`) and `β·(1 - exp(-κh))` (`closure="saturating"`),
and `scripts/05_dispersion_sensitivity.py` quantifies the difference:

| Closure | Uncontrolled PM2.5 | Reduction at u=0.5 | Reduction at u=1 | Trough hour |
|---|---|---|---|---|
| `exp` | 67.75 | 39.2% | 63.4% | 04:00 |
| `saturating` | 65.17 | 38.9% | 63.0% | 15:00 |

The policy comparison is insensitive to the choice; the diurnal phase is not.
Delhi's observed minimum is in the afternoon, which the `saturating` closure
reproduces and `exp` does not.

---

## Reproducing the reported results

```bash
make data
make forecast        # tables 1-5
make control         # table 6
make sensitivity     # reward-regime and dispersion-closure studies
```

Every script writes a JSON to `results/` containing the full config, per-seed
metrics and wall-clock timings, so a run is auditable after the fact.
`results/RUNLOG.md` records the verification run that produced the bundled
checkpoints.

The CPCB parquet (15 MB) and the three derived city series are bundled under
`data/` so the repository replicates offline. If you push to GitHub, either
track `data/raw/*.parquet` with Git LFS or drop it and let
`scripts/01_build_dataset.py` fetch it on first run.

---

## Citation

```bibtex
@article{saicharan2026urbanclimaterl,
  title  = {Forecast horizon governs which air-quality models are useful
            for urban pollution control},
  author = {Sai Charan, A and Harshitha, Sunkara and Debnath, Ramit},
  year   = {2026}
}
```

Code released under MIT; see `LICENSE` for the data terms.

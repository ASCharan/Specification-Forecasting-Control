# Changelog

## 2026-08-08

Changes made while resolving the pre-submission audit of the manuscript. Every
number below was produced by running the pipeline in this repository; nothing
was estimated or carried over from an earlier draft.

### Added

- **Transformer at the 24-hour horizon, all three cities, five seeds.** This
  model had been evaluated at `h=1` only; Table 2 of the manuscript was
  incomplete. Run with the standard protocol (`--lag-offset 1`,
  `--align-test-rows`, 100-epoch cap, seeds 0-4):

  | City | RMSE | R² | per-seed RMSE |
  |---|---|---|---|
  | Delhi | 71.55 ± 0.96 | 0.345 | 71.60, 73.36, 71.16, 70.70, 70.90 |
  | Mumbai | 22.14 ± 0.84 | 0.781 | 22.57, 20.62, 23.07, 21.95, 22.50 |
  | Kolkata | 55.92 ± 2.19 | 0.540 | 54.18, 54.24, 56.09, 60.07, 55.01 |

  Written to `results/transformer_h24.json` and merged into
  `results/aggregated_rerun.json`. Checkpoints in
  `checkpoints/{city}/h24/transformer_seed{0..4}.pt`.

  Consequences for the manuscript: no city winner changes (Delhi seasonal
  ARIMA, Mumbai C-Mod, Kolkata MLP-64), but the stability result does. The
  Transformer is the least reproducible model at `h=24` in Delhi and Kolkata
  and the LSTM in Mumbai, so the claim is now that the two *sequence* models
  are the least stable, not the LSTM alone.

- `scripts/12_manuscript_figures.py` regenerates the two manuscript comparison
  figures (`fig_model_comparison.png`, `fig_seed_stability.png`) from
  `results/aggregated_rerun.json`. Neither figure was previously reproducible
  from this repository. The script plots whatever models are present, so
  re-running a model updates the figure without editing the script.

- `scripts/10_traffic_informativeness.py` now also records
  `rho_to_exceed_seed_noise_interp`, the informativeness threshold obtained by
  linear interpolation between swept values of rho. The manuscript quotes the
  interpolated value (0.09 at `h=1`, 0.11 at `h=24`); the file previously
  stored only the coarser grid value (0.1 and 0.2), so the reported number
  could not be traced back to a released artefact.

### Removed

- `urbanclimate/vision.py`, `scripts/11_vision_validation.py`,
  `results/vision_validation.json`, `requirements-vision.txt`, and the `vision`
  Makefile target. There is no computer-vision component in this study: the
  traffic input is a synthetic diurnal template throughout, no reported result
  depended on the module, and it appeared in neither the architecture figure
  nor the component-utilisation audit. The corresponding manuscript passages
  were deleted at the same time.

### Verified, not changed

- **Reproducibility.** Re-running C-Mod on Delhi `h=1` reproduces Table 1
  exactly (C-Mod 25.18, ridge 26.93, persistence 28.76, climatology 91.60,
  seasonal-naive 80.14). The 8 smoke tests pass.

- **Two uncontrolled PM2.5 baselines.** `results/control.json` (Table 7) uses
  `eval_episodes: 50` and gives 67.72; `scripts/05_dispersion_sensitivity.py`
  defaults to 20 evaluation episodes and gives 67.75. Different episode counts
  on the same environment and coefficients. Neither value is wrong; the
  manuscript now states the episode basis for each.

- **Seed noise in the informativeness table.** The value is C-Mod's across-seed
  standard deviation in RMSE units *with the rho=0 random covariate in place of
  the traffic proxy* (0.20 at `h=1`, 0.61 at `h=24`), which is a different input
  configuration from Tables 1 and 2 (0.10 and 0.48). Not a discrepancy; the
  table caption now says which configuration it refers to.

- **Inert w2 reward term.** Because the weather fields are exogenous and matched
  across policies, the term contributes an identical constant
  0.3 x 0.142637 = 0.0428 to every policy's mean per-step reward. Setting
  `w2 = 0` would shift the whole average-reward column of Table 7 by that amount
  and change no ranking and no difference between policies, so no retraining is
  needed. Left in place for consistency with the released artefacts.

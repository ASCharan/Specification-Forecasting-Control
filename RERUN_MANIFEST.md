# Re-run manifest

**STATUS: this re-run has been executed in full.** All 139 tasks completed;
every number in the manuscript comes from it. `results/aggregated_rerun.json`
holds the final values and `results/rerun_provenance.jsonl` the per-task log.
The commands below reproduce it from scratch. The order
matters: figures and bootstrap comparisons read files written by earlier steps.

Single-core wall-clock estimates in brackets. On a multi-core machine, run the
six forecasting commands in parallel.

## 0. Housekeeping (do this first)

```bash
rm -f NOTES_FOR_AUTHORS.md        # it says "do not distribute" on line 3
rm -f results/RUNLOG.md           # its control-arm claim is incorrect
rm -rf results/*.json checkpoints/*/ figures/
python -m pytest tests/ -q        # [~10 s]
```

## 1. Dataset

```bash
python scripts/01_build_dataset.py        # [~5 s, downloads 15 MB once]
```

Expected: Delhi 19,071 h, Mumbai 28,209 h, Kolkata 12,348 h.
If these differ, stop: the upstream mirror has changed and every downstream
number moves with it.

## 2. Forecasting, all six city-horizon combinations

Every command uses the default `--lag-offset 1` and the default
`--align-test-rows`. **Do not vary either between commands.** That is the
failure documented in Section S1 of `CRITICAL_ANALYSIS.md`.

```bash
python scripts/02_run_forecasting.py --city Delhi   --horizon 1  --seeds 0 1 2       # [~6 min]
python scripts/02_run_forecasting.py --city Delhi   --horizon 24 --seeds 0 1 2 3 4   # [~4 min]
python scripts/02_run_forecasting.py --city Mumbai  --horizon 1  --seeds 0 1 2       # [~12 min]
python scripts/02_run_forecasting.py --city Mumbai  --horizon 24 --seeds 0 1 2 3 4   # [~6 min]
python scripts/02_run_forecasting.py --city Kolkata --horizon 1  --seeds 0 1 2       # [~4 min]
python scripts/02_run_forecasting.py --city Kolkata --horizon 24 --seeds 0 1 2 3 4   # [~3 min]
```

Measured costs on one core: SARIMA is cheap (11-44 s) because the state-space
model is fitted once and rolled with `refit=False`. The Transformer at h=1 is
the slowest component, 106-292 s per seed.

Fills: Table 1, Table 3, Table 4, Table 5 (`tab:forecast`, `tab:h24`,
`tab:mumbai`, `tab:kolkata`).

## 3. The lag-convention diagnostic (Table 2)

```bash
python scripts/02_run_forecasting.py --city Delhi --horizon 1 --seeds 0 1 2 \
    --lag-offset 0 --skip-sarima --tag diagnostic      # [~6 min]
```

This prints a loud warning by design. Its output belongs only in
`tab:convention` and must never be merged with default-convention results.

## 4. Input ablation (Table 6)

```bash
python scripts/06_ablation.py --city Delhi --horizon 1 --seeds 0 1 2   # [~25 min]
```

Fills `tab:ablation`.

## 5. Control arm

```bash
python scripts/03_train_ppo.py --seeds 0 1 2 --episodes 300   # [~4 min]
python scripts/04_reward_sensitivity.py                       # [~3 min]
python scripts/05_dispersion_sensitivity.py                   # [~2 min]
python scripts/09_reward_diagnostics.py                       # [~10 s]
```

`03_train_ppo.py` now asserts that training and evaluation episode seeds are
disjoint. If that assertion fires, do not work around it.

Fills `tab:rl`, Section "The reward weighting, not the algorithm, selects the
regime", and the dispersion-closure numbers in Methods.

## 6. Statistical comparisons

```bash
python scripts/07_paired_bootstrap.py --city Delhi   --horizon 24 --all
python scripts/07_paired_bootstrap.py --city Delhi   --horizon 1  --all
python scripts/07_paired_bootstrap.py --city Mumbai  --horizon 24 --all
python scripts/07_paired_bootstrap.py --city Kolkata --horizon 24 --all
```

Reads saved predictions, so it never retrains. [~1 min each]

## 7. Figures

```bash
python scripts/08_make_figures.py --outdir figures
cp figures/*.png ../manuscript/
```

Regenerates all six. **The three control figures in the current Overleaf
project are from a superseded code version and must be replaced**: the shipped
`ppo_learning_curve.png` plateaus near −230 per episode, the patched code near
−102.

## 8. Manuscript

1. Transcribe results into `main.tex`, replacing every `\rr` flag.
2. Update the *n* in each table caption from the `n_scored` field of the
   corresponding results file.
3. Update the rolling-origin forecast counts in Methods §"Forecasting
   baselines" from the `sarima.n_scored` field.
4. `grep -c '\\rr' main.tex` must return 0. Then delete the `\newcommand{\rr}`
   line.
5. Reconcile the duplicate `guttikunda2023what` / `guttikunda2019apna`
   bibliography entries.
6. Compile twice (cross-references) and check for undefined citations.

## 9. Resolved

The published h=24 LSTM figure (75.56 ± 0.39) does not reproduce from this
repository under either lag convention; the full five-seed re-run gives
72.49 ± 0.73 on Delhi. The manuscript now reports the re-run value. The
original number appears to have come from a code state that is not in this
repository; it is superseded and should not be cited.

One earlier diagnosis was itself wrong and is corrected here: sequence models
are NOT insensitive to the lag convention. Windows are built from the feature
matrix, so all lags shift together and the window never sees y_t under
convention B. Measured degradation under B is uniform across model families
(persistence +56%, ridge +51%, C-Mod +49%, LSTM +48%).

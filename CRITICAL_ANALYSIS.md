# UrbanClimate-RL: technical audit

Scope: the 1 August 2026 Overleaf draft (`main.tex`, 866 lines) checked line by
line against the `urbanclimate-rl` repository, with every disputed number
re-run. Findings are ranked by what they do to the paper, not by how hard they
are to fix.

Everything below was verified by execution unless marked otherwise. Raw
evidence is in `verified_results.json`.

---

## Summary

Three findings are severe enough that the manuscript cannot be submitted in its
1 August form:

1. **Table 1 mixed two lag conventions**, so the headline comparison was not
   like-for-like. Correcting it removes the paper's central claim.
2. **The central claim is not merely unsupported but false on Delhi.** Under a
   uniform feature construction the LSTM beats C-Mod at 24 hours, and ridge
   regression beats both.
3. **The computer-vision arm does not exist.** No detector, no tracker, no
   video, no results file, while Code availability declares the pipeline
   deposited.

Two further findings would independently trigger an integrity query: the
control table and its learning curve do not reproduce from the released code,
and a three-seed standard deviation is reported from a single run.

---

## S1. Table 1 mixed two lag conventions (verified)

`urbanclimate/data.py::make_features` carries a `lag_offset` switch. With
`lag_offset=1`, lag *k* is `series[t-k+1]`, so lag 1 is the most recent
observation. With `lag_offset=0`, lag 1 is `series[t-1]` and the most recent
observation is withheld. Neither leaks. The published Table 1 drew its rows
from both.

Delhi, *h*=1, three seeds, RMSE:

| Model | `lag_offset=1` | `lag_offset=0` | Published Table 1 | Row came from |
|---|---|---|---|---|
| Persistence | **28.75** | 44.83 | **28.75** | offset 1 |
| Ridge | 26.89 | **40.69** | **41.54** | offset 0 |
| MLP-64 | 25.13 | **38.00** | **37.80** | offset 0 |
| MLP-128 | 25.02 | **37.70** | **37.65** | offset 0 |
| C-Mod | 25.13 | **37.59** | **37.75** | offset 0 |
| LSTM | **25.36** | 37.56 | **25.77** | offset 1 |
| Transformer | (not re-run) | | 26.74 | offset 1 |

The tabular models were scored one hour further from the target than the
sequence models. Every number reproduces individually. The table as a whole is
invalid.

Consequence: the claim that *"at this horizon the binding constraint is access
to sequence context, not the continuous-depth parameterisation"* is an artefact.
The tabular models clustered at R² ≈ 0.78–0.82 because they were denied *y_t*,
not because they lack sequence context.

**Fixed in the revised code**: `LAG_OFFSET` is now a module-level constant,
validated on entry, recorded in every results file, and printed as a loud
warning when a run deviates from it.

---

## S2. The central claim is falsified on Delhi (verified)

Recomputed under one convention and one set of scoring rows (*n* = 3,787 at
*h*=1, *n* = 3,783 at *h*=24):

**Delhi, h = 1** (three seeds)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| MLP-128 | **25.07 ± 0.12** | 14.51 | **0.920** |
| Transformer | 25.16 (1 seed) | 14.49 | 0.919 |
| C-Mod | 25.18 ± 0.10 | 14.47 | 0.919 |
| MLP-64 | 25.18 ± 0.16 | 14.47 | 0.919 |
| LSTM | 25.36 ± 0.16 | **14.38** | 0.918 |
| Ridge | 26.93 | 15.35 | 0.908 |
| Persistence | 28.76 | 16.92 | 0.895 |

Five architectures inside a 1.2 % band. Paired bootstrap, LSTM − C-Mod:
**+0.45, 95 % CI [−0.26, +1.27]**, brackets zero.

**Delhi, h = 24** (five seeds)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | **71.50** | **51.62** | **0.346** |
| LSTM | 72.49 ± 0.73 | 52.94 | 0.328 |
| MLP-128 | 72.59 ± 0.30 | 52.96 | 0.326 |
| C-Mod | 72.98 ± 0.48 | 52.72 | 0.319 |
| Persistence / seasonal | 80.18 | 55.74 | 0.178 |

Paired bootstrap, LSTM − C-Mod: **−1.21, 95 % CI [−2.24, −0.20]**, excludes
zero in the direction opposite to the published claim. Five-seed ranges overlap
almost completely (C-Mod 72.25–73.73, LSTM 71.70–73.53).

Three manuscript sentences do not survive:

- *"C-Mod is never outperformed by the LSTM at 24 hours"* — false on Delhi.
- *"overtakes it with no overlap across five seeds (C-Mod 71.8–72.9, LSTM
  75.1–76.1)"* — the actual ranges overlap on four of five seeds.
- *"the continuous-time forecaster, unremarkable at one hour"* — it is not
  unremarkable at one hour once fed fairly, so there is no reversal to narrate.

Note also that the published *h*=24 LSTM (75.56 ± 0.39) does not reproduce from
this repository under **either** convention (72.49 ± 0.73 / not run). That
discrepancy is separate from the lag issue and remains unexplained. It needs
checking against whatever code produced the original run.

---

## S3. The computer-vision arm did not exist (now built, scoped as engineering)

**Status: resolved.** A real detector plus tracker is now implemented in
`urbanclimate/vision.py`, its counting arithmetic verified exactly against
synthetic ground truth (zero error), and its throughput measured at 92 ms per
frame on 1080p input. It is released as a deployment component and produces no
reported result. Separately, `scripts/10_traffic_informativeness.py` now bounds
what a real traffic series would have to be worth: a covariate must correlate
at rho ~ 0.1 with the unexplained AQI variation before it beats retraining
noise, and rho = 0.3 buys more than the entire architecture spread. The
original finding is preserved below for the record.

### Original finding

No `ultralytics`, `cv2`, `torchvision`, ByteTrack import, video file, frame
dump, annotation, detection script or CV results file appears anywhere in the
archive. `requirements.txt` contains no vision dependency. The only string match
for "track" in the repository is a README line about Git LFS.

The manuscript nonetheless asserts: a 5.7 s 1080p clip at 171 frames, 2.91
vehicles per frame, 16 unique identities, 104 ms median per frame on CPU,
Ultralytics 8.4.92, plus the density formula in Methods §4.8 and a CV block in
Fig. 1. **Code availability declares the pipeline deposited.** Nature-family
journals check that statement against the repository.

§2.8 already concedes that no result depends on the module. Removing it costs
the paper nothing and removes an editorial-desk rejection risk.

**Action taken in the revision**: §2.8 and Methods §4.8 deleted, Fig. 1 redrawn
without the CV block, Code availability corrected.

---

## S4. Table 5 and Fig. 6 do not reproduce from the released code (verified)

| Quantity | Manuscript | Released code (verified) |
|---|---|---|
| No control, reward | −1.721 | −1.016 |
| Random, PM₂.₅ | 42.37 | 41.22 |
| Fixed 0.5, PM₂.₅ | 42.34 | 41.19 |
| Fixed max, PM₂.₅ / reduction | 25.99 / 61.7 % | 24.82 / 63.4 % |
| PPO, reward | −1.372 ± 0.008 | −0.598 (seed 0) |

The reward column is offset by roughly 1.7× throughout.
`ppo_learning_curve.png` plateaus near −230 per episode (−1.37 per step,
matching Table 5); the released `curves` array plateaus at −106 (−0.63 per
step). Figure and table agree with each other; neither agrees with the code.

`results/RUNLOG.md` asserts the control arm "reproduces as published". That
assertion is itself incorrect and should be removed.

---

## S5. Three-seed PPO statistics from a single run

`results/control.json` records `seeds: [0]`. `ppo_std` is exactly zero on all
three metrics, `curves` holds one key, `checkpoints/ppo/` holds one file. The
manuscript reports −1.372 ± 0.008, 58.3 % ± 1.7 %, "three training seeds" in
the Table 5 caption, "three independent seeds" in Methods §4.11, and
"Controllers are trained from three independent seeds" in §2.7.

A standard deviation cannot be computed from one run.

**Action taken**: the control arm was re-run from patched code with three
genuinely independent, uncontaminated seeds. Verified replacement values:

| Policy | Reward | Cost ‖u‖₂ | PM₂.₅ | Reduction |
|---|---|---|---|---|
| No control | −1.016 | 0.000 | 67.72 | — |
| Random | −0.710 | 1.265 | 41.22 | 39.1 % |
| Fixed 0.5 | −0.689 | 1.118 | 41.19 | 39.2 % |
| Fixed max | −0.625 | 2.236 | 24.82 | 63.4 % |
| **PPO (3 seeds)** | **−0.605 ± 0.002** | 1.899 ± 0.063 | 27.10 ± 0.53 | 60.0 % ± 0.8 % |

PPO attains the best reward at 15 % lower control effort than the always-maximal
rule. The qualitative conclusion of §2.7 survives; the numbers change.

---

## S6. PPO seed 1 trained on the evaluation episodes (verified)

`train_ppo` reset with `seed*10_000 + ep`, so run seed 1 swept 10,000–10,299.
`evaluate_policy` used `base_seed=10_000` through 10,049. Measured overlap:
**50 of 50 evaluation episodes**. One of the three reported controllers was
scored on episodes it had trained on.

**Fixed**: training and evaluation now occupy disjoint seed namespaces
(`TRAIN_SEED_BASE = 1_000_000`, `EVAL_SEED_BASE = 10_000`), with
`assert_disjoint_seeds()` failing loudly on any future collision.

---

## S7. Three different test sets inside one table (verified)

| City, horizon | Tabular models | Sequence models | SARIMA path |
|---|---|---|---|
| Delhi, h=1 | 3,810 | 3,787 | 3,815 |
| Delhi, h=24 | 3,806 | 3,783 | 3,792 |
| Mumbai, h=1 | 5,637 | 5,614 | 5,642 |
| Mumbai, h=24 | 5,633 | 5,610 | 5,619 |
| Kolkata, h=1 | 2,466 | 2,443 | 2,470 |
| Kolkata, h=24 | 2,461 | 2,438 | 2,447 |

`make_windows` costs the sequence models the first 23 rows of each split;
`sarima_forecast` splits the raw series 80/20 rather than the feature matrix.
Effects are small but non-zero, and Table 1 compared eleven models across three
row sets.

**Fixed**: `--align-test-rows` (default on) scores every model on the common
intersection. The manuscript now states *n* explicitly in every table caption.

Related: the stated rolling-origin counts are wrong. Delhi *h*=24 states 3,781
against 3,792 computed; Kolkata states 2,436 against 2,447. Both off by 11.

---

## S8. The reward's forecast-error term is inert (verified)

The weather fields are exogenous, so ε^W cannot depend on the action. Measured
‖ε^W‖ over 168 hours:

| u | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| mean ‖ε^W‖ | 0.142637 | 0.142637 | 0.142637 | 0.142637 | 0.142637 |

Spread across action levels: exactly 0. The *w₂* term contributes a constant
offset to the return and cannot influence the learned policy. It occupies a
symbol in Eq. (5), a coefficient in the weights tuple and a clause in the
Methods while doing nothing.

**Action**: retained for consistency with the released artefacts, but now
reported as inert in the Methods, with `scripts/09_reward_diagnostics.py`
demonstrating it. Coupling it to a controller-in-the-loop forecaster would make
it active and is the natural next experiment.

---

## S9. What could not be reproduced from the released scripts

| Manuscript element | Status before revision |
|---|---|
| Table 2 (ablation) | `use_met` / `use_traffic` exist in `make_features`, no script exposed them |
| Every bootstrap CI in the text | `paired_bootstrap` implemented, unit-tested, never called |
| Tables 3, 4 (Mumbai, Kolkata) | No result files in `results/` at all |
| All six figures | No plotting code anywhere |
| Methods §4.9 "torchdiffeq" | Released default is a built-in RK4; torchdiffeq commented out of requirements |
| Methods "five seeds at h=24" | Released h=24 file has three |

**Fixed**: `06_ablation.py`, `07_paired_bootstrap.py`, `08_make_figures.py`,
`09_reward_diagnostics.py` added; `02_run_forecasting.py` now saves predictions
so the bootstrap runs without retraining; Methods §4.9 corrected to describe
the solver actually used.

---

## S10. Statistical rigor

- **Bootstrap scope.** The CIs compare seed-0 models only, then support claims
  about seed means. The Methods did note this; the revision additionally quotes
  seed ranges wherever a comparison is close.
- **No multiple-comparison correction** across 3 cities × 2 horizons × ~10
  models. Now stated explicitly in the evaluation protocol.
- **Climatology as a floor.** Climatology attains R² = −0.069, meaning it fails
  to beat the test-split mean. "Clearing the climatology floor" was offered as
  evidence of skill; it is not. Persistence at R² = 0.895 is the only
  informative baseline. Corrected in the revision.
- **Post-hoc rationalisation.** *"the smallest of the three series penalises the
  recurrent model most"* explained away an inconvenient Kolkata number without
  testing it. Removed pending the re-run.

---

## S11. Physical and modelling points

- **Dispersion closure has the wrong sign** relative to box-model physics. The
  authors acknowledged this and the sensitivity analysis is the correct
  response. The revision states the sign error in the Methods rather than
  implying it is merely "effective".
- **Quadratic congestion term is leading-order**, not a correction: 49 % of
  PM₂.₅ emission at the diurnal traffic peak, 36 % at the daily mean, 20 % at
  the overnight minimum (measured, `09_reward_diagnostics.py`). The Discussion
  leaned on this term without quantifying it.
- **Eq. (1) omits gas-phase chemistry and any regional inflow term.** For Delhi
  the regional contribution is substantial. A reviewer will raise it. Now named
  as a limitation.
- **Wind template phase error propagates into the simulator.** The Methods
  acknowledged the early-morning wind maximum for the forecasting inputs but
  not that the same template drives advective removal in Eq. (1). Now stated.
- **τ_safe = 15 µg/m³ is the WHO 24-hour guideline applied as an hourly
  threshold.** Now flagged in the Methods.
- **Unit label.** α₁ carried "conc. h⁻¹ [u]⁻¹" across three pollutants with
  different units. Now defined in the table caption.

---

## S12. Component audit: what the architecture figure claimed versus what runs

The original Figure 1 drew a single state-fusion node fed by all three inputs
and feeding C-Mod, PPO and the pollutant-dynamics simulator alike. That implies
an integration that does not exist. Audited against the code:

| Component | Forecasting arm | Control arm | Measured role |
|---|---|---|---|
| AQI monitors (real CPCB) | target and lags | **never used** | lags alone 26.26 vs 25.18 full |
| Meteorology W(t) | template, 3 of 8 features | separate template, drives advection | removing it costs +3.5% |
| Traffic T(t) | template, 1 of 8 features | separate template, **actuated** | −0.2% as feature; 51.5% of reduction as actuated state |
| State fusion | 8-column feature matrix | 13-dim observation | two constructions, two data sources |
| C-Mod | used | **never used** | does not enter the control loop |
| Pollutant dynamics Eq. (1) | **never used** | the environment | coefficients assumed |
| PPO agent | not used | used | recovers near-saturation optimum |
| u1–u3 (signals, speed) | n/a | scales T(t) | **51.5% reduction in isolation** |
| u4–u5 (zone enforcement) | n/a | direct removal via Delta | 27.0% in isolation |
| Forecast feedback | not exercised | not exercised | design path only |

Three defects followed from the single-fusion drawing:

1. **No observational data enters the control arm at all.** The simulator
   generates its own W, T and P. The figure implied real AQI drives the
   controller and the simulator. It does not, so neither arm validates the
   other.

2. **There are two traffic series, not one, and they play opposite roles.**
   Both use the same template, `0.25 + 0.75(g_8.5 + 0.95 g_18.5)`, but the
   forecasting feature adds observation noise and a fixed amplitude while the
   simulator's is noiseless and actuated. As a predictor it is inert, because
   hour-of-day is already recoverable from the AQI lags. As an actuated state
   it carries most of the control benefit. Drawing one box for both was the
   source of the user's question and it was a fair question.

3. **The action labels are nominal, and the split between channels is an
   assumption.** Mechanically u1–u3 are a traffic multiplier and u4–u5 are two
   columns of Delta with larger coefficients. The 51.5/27.0 split follows from
   the assumed 35% traffic ceiling and the Delta entries, not from evidence.

**Resolved**: Figure 1 redrawn as two explicitly separated panels with the
data source of each labelled and a marked boundary that no observational data
crosses; new Methods section "Which components are exercised, and by what data"
carrying the three points that matter for reading the figure, with the full
audit table moved to Supplementary Table S1 so the Methods do not read as
defensive; new Results paragraph giving the action-channel decomposition; §2.5
wording tightened so the two traffic series are never conflated.

## S13. Citation numbering

`thebibliography` assigns numbers by `\bibitem` order, not by order of first
citation, so in-text numbers ran 1, 2, 3, 7, 17, 18, 19, 20, 21, 22, 23, 8.
37 of 42 positions disagreed with first-appearance order. Separately, five
references (Mnih 2015, Sutton & Barto, IntelliLight, PressLight, Chen 2020)
had lost their in-text citations during restructuring and were sitting in the
bibliography uncited.

**Resolved**: the five RL and traffic-signal-control references are re-cited
where they belong, in the control-arm introduction and the PPO methods, and the
bibliography is reordered by first citation. In-text numbering is now strictly
1..47 in order of appearance, with every reference cited exactly once in the
list and none orphaned.

## S14. Narrative and presentation

- **Fig. 1 promised an integration the experiments do not deliver.** The
  forecaster is not in the control loop, the CV pipeline fed nothing, the
  exogenous inputs were synthetic. The paper then apologised for this in five
  separate passages. Revision: figure moved to Methods, unexercised paths
  labelled as such, caveats consolidated to one statement each.
- **Title overclaimed.** *"Forecast horizon governs which air-quality models are
  useful for urban pollution control"* requires showing that forecaster choice
  changes control outcomes. No benchmarked forecaster ever entered the
  controller. Retitled; two alternatives are in a comment block at the top of
  `main.tex`.
- **The best result was buried.** §2.9 (reward weighting selects the regime) is
  the most novel and most transferable finding and ran to six lines near the
  end. Promoted to a full section and given the closing argument.
- **Redundancy.** The "no one-hour leader keeps its edge at 24 hours" formula
  appeared five times; the "synthetic inputs" caveat five times. Each now
  appears once.
- **Sentence architecture.** 22 sentences exceeded 45 words, the longest 82.
  Broken up. Abstract was 212 words (acceptable for Nature Communications, over
  the limit for Nature).
- **Bibliography.** Clean: 52 keys used, 52 defined, no dangling cross-
  references. One duplicate pair worth checking, `guttikunda2023what` and
  `guttikunda2019apna`, which currently carry identical bibliographic content
  under different keys.

---

## Can measured traffic rescue the architecture result?

No, and the question is now answered quantitatively rather than by assertion.

The concern is legitimate: the ablation tests a *synthetic* proxy, and vehicular
sources genuinely dominate urban PM2.5 in Indian cities. But three facts bound
what a camera could contribute.

1. **The threshold is known.** Sweeping traffic covariates of controlled
   informativeness gives: rho = 0.05 -> 0.2% gain (invisible), rho = 0.1 ->
   0.6-0.9% (at the noise floor), rho = 0.2 -> 2.5-3.0%, rho = 0.3 -> 5.5-6.4%,
   rho = 0.5 -> 15-17%. The seed-noise floor sits at rho ~ 0.09 (h=1) and
   rho ~ 0.11 (h=24).

2. **The achieved value cannot be established from this dataset.** The target is
   the hourly cross-station median AQI over a metropolitan network. One
   intersection's count is not a city-scale density, and no traffic record
   co-located with the CPCB network spans 2020-2023. Running a detector on an
   arbitrary clip would produce a number with no defined relationship to the
   forecasting target: it would be the same category of error as the original
   Table 1.

3. **Even a successful outcome would not restore the architecture claim.** A
   covariate at rho = 0.3 lifts *every* model by 5-6%. It changes the level, not
   the ordering. Nothing about it would make the Neural ODE beat ridge
   regression, because the covariate enters all of them identically.

The constructive reading is the paper's own thesis arriving from the opposite
direction: a covariate at rho = 0.3 buys more than the entire spread between
architectures, so a measurement programme would matter more than any
model-selection decision examined here. That is worth stating, and it is now
Section 2.6.

## What the corrected evidence supports

The honest paper is a different and, I would argue, stronger one:

1. **Architecture is not identifiable on this task.** Five architectures agree
   to 1.2 % at one hour; a linear model wins at 24 hours. A clean negative
   result on a question the field keeps answering positively.
2. **A one-position change in lag indexing manufactures a 50 % error swing**
   and, applied inconsistently, a spurious 12-point deep-learning advantage.
   The failure is silent: no warning, no leakage, no implausible score, survives
   seed averaging and ablation, invisible in a results table. This is worth
   reporting on its own.
3. **The reward weighting, not the algorithm, selects the control regime.**
   Untouched by any of the above and the most transferable finding in the study.
4. **The value of better inputs can be bounded before they are collected.**
   The informativeness sweep costs minutes and yields a procurement-relevant
   threshold. This is the constructive counterpart to the negative results.

Together these make one argument: specification dominates architecture in both
arms. That is what the revised manuscript now says.

---

## Before submission

1. Re-run all six city-horizon combinations under the uniform convention
   (`RERUN_MANIFEST.md`).
2. Regenerate all six figures (`08_make_figures.py`). The three control figures
   are currently from a superseded code version.
3. Delete `NOTES_FOR_AUTHORS.md`. It says so itself.
4. Delete `results/RUNLOG.md` or correct its control-arm claim.
5. Resolve the *h*=24 LSTM discrepancy against the original training code.
6. Clear every `\rr` flag in `main.tex`, then delete the macro.
7. Reconcile the duplicate `guttikunda` bibliography entries.

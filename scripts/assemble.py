#!/usr/bin/env python3
"""Collect every number the response letter needs, from the result files only."""
import json
from pathlib import Path

import numpy as np

R = Path("results")


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- data audit
a = load("r1_data_audit.json")
if a:
    sec("R1.2  DATA COMPLETENESS AUDIT")
    print(f"{'city':<9}{'hours':>7}{'interp':>8}{'%':>7}{'gaps 1/2/3':>13}"
          f"{'stations min/med/max':>23}{'acf1 full':>11}{'acf1 obs':>10}")
    for c, d in a.items():
        g = d["gap_length_hist"]
        gaps = "%d/%d/%d" % (g.get("1", 0), g.get("2", 0), g.get("3", 0))
        st = "%d/%.0f/%d" % (d["stations_min"], d["stations_median"],
                             d["stations_max"])
        print(f"{c:<9}{d['hours']:>7}{d['n_interpolated']:>8}"
              f"{d['pct_interpolated']:>7}{gaps:>13}{st:>23}"
              f"{d['acf1_full']:>11}{d['acf1_observed_only']:>10}")
    for c, d in a.items():
        print(f"  {c}: balanced panel {d['n_stable_stations_90pct']} stations, "
              f"{d['balanced_hours']} h, mean {d['balanced_mean_aqi']}, "
              f"acf1 {d['balanced_acf1']}")
        print(f"     monthly median station count ranges "
              f"{d['monthly_station_median_min']} to "
              f"{d['monthly_station_median_max']}; lowest monthly minimum "
              f"{d['monthly_station_min_overall']}")

# ------------------------------------------------------------------- control
for f, label in [("r1_control_rerun.json", "primary run"),
                 ("r1_control_box.json", "box run")]:
    c = load(f)
    if not c:
        continue
    for cl in ("saturating", "exp", "box"):
        if cl not in c:
            continue
        d = c[cl]
        sec(f"R1.3  CONTROL ARM, closure = {cl}  ({label})")
        pt = d["policy_table"]
        print(f"{'policy':<12}{'reward':>10}{'cost':>8}{'PM2.5':>8}{'reduction':>11}")
        for k in ("none", "random", "half", "max", "ppo"):
            r = pt[k]
            extra = (f"  +/- {pt['ppo']['avg_reward_sd']:.4f} reward, "
                     f"{pt['ppo']['cost_sd']:.3f} cost, "
                     f"{pt['ppo']['mean_pm25_sd']:.2f} pm25"
                     if k == "ppo" else "")
            print(f"{k:<12}{r['avg_reward']:>+10.4f}{r['cost']:>8.3f}"
                  f"{r['mean_pm25']:>8.2f}{r['reduction_pct']:>10.1f}%{extra}")
        print(f"diurnal: minimum {d['diurnal']['hour_of_min']:02d}:00, "
              f"maximum {d['diurnal']['hour_of_max']:02d}:00, "
              f"range {d['diurnal']['min']}-{d['diurnal']['max']}")
        ch = d["channels"]
        print(f"channels: traffic {ch['traffic_only']:.2f} "
              f"({ch['traffic_only_reduction_pct']}%), enforcement "
              f"{ch['enforcement_only']:.2f} "
              f"({ch['enforcement_only_reduction_pct']}%), both "
              f"{ch['both']:.2f} ({ch['both_reduction_pct']}%)")
        print(f"72-point map: u* minimum {d['u_star_min']}, "
              f"inaction optimal in {d['n_inaction_optimal']} of 72 cells, "
              f"gain {d['gain_range'][0]:.4f} to {d['gain_range'][1]:.4f}")
        print(f"return sd over the last third of training: "
              f"{d['return_sd_mean']:.3f} "
              f"(per seed {[round(x,3) for x in d['return_sd_per_seed']]})")
        print(f"{'w1':>5}{'w3':>6}{'u*':>6}{'avail':>9}{'cost':>8}"
              f"{'recovered':>11}{'ret sd':>8}{'SNR':>8}")
        for r in d["value_recovered"]:
            snr = r["value_available"] * 168 / max(r["return_sd"], 1e-9)
            print(f"{r['w1']:>5}{r['w3']:>6}{r['u_star']:>6}"
                  f"{r['value_available']:>9.4f}{r['ppo_cost']:>8.3f}"
                  f"{r['recovered_pct']:>10.1f}%{r['return_sd']:>8.3f}{snr:>8.1f}")

# ------------------------------------------------------------- decomposition
d = load("r1_decomposition.json")
if d:
    sec("R1.1  PAIRED LAG-CONVENTION PENALTY")
    print(f"{'city':<9}{'h':>4}{'model':<12}{'n':>3}{'penalty':>10}"
          f"{'paired sd':>11}{'|t|':>8}{'marg sd':>9}{'pen/marg':>10}")
    for r in d["paired_convention"]:
        if r["n_seeds"] == 0:
            print(f"{r['city']:<9}{r['horizon']:>4}{r['model']:<12}"
                  f"{'-':>3}{r['penalty_pct']:>9.2f}%{'exact':>11}")
        else:
            print(f"{r['city']:<9}{r['horizon']:>4}{r['model']:<12}"
                  f"{r['n_seeds']:>3}{r['penalty_pct']:>9.2f}%"
                  f"{r['penalty_sd_pct']:>11.3f}{r['ratio_paired']:>8.1f}"
                  f"{r['marginal_sd_pct']:>9.3f}{r['ratio_marginal']:>10.2f}")

    sec("R1.1  ARCHITECTURE RANGE")
    print(f"{'city':<9}{'h':>4}{'all':>9}{'-tfm':>9}{'-seq':>9}{'IQR':>9}"
          f"   worse than persistence")
    for r in d["architecture_range"]:
        print(f"{r['city']:<9}{r['horizon']:>4}{r['range_all_pct']:>8.2f}%"
              f"{r['range_no_transformer_pct']:>8.2f}%"
              f"{r['range_no_sequence_pct']:>8.2f}%{r['iqr_pct']:>8.2f}%   "
              f"{r['worse_than_persistence'] or 'none'}")

    sec("R1.1  VARIANCE DECOMPOSITION (sd, percent of persistence RMSE)")
    print(f"{'city':<9}{'h':>4}{'convention':>12}{'architecture':>14}"
          f"{'interaction':>13}{'seed':>8}{'models':>8}{'seeds':>7}")
    for r in d["decomposition"]:
        print(f"{r['city']:<9}{r['horizon']:>4}{r['sd_convention_pct']:>12.3f}"
              f"{r['sd_architecture_pct']:>14.3f}"
              f"{r['sd_interaction_pct']:>13.3f}{r['sd_seed_pct']:>8.3f}"
              f"{r['n_models']:>8}{r['n_seeds']:>7}")

# ----------------------------------------------------------------- interpfree
d = load("r1_interpfree.json")
if d:
    sec("R1.2  RESCORING ON INTERPOLATION-FREE ROWS")
    print(f"{'cell':<14}{'rows kept':>12}{'dropped':>9}{'max |dRMSE|':>13}"
          f"{'ranking':>12}")
    for k, v in d.items():
        mx = max(abs(x["delta_pct"]) for x in v["by_model"].values())
        print(f"{k:<14}{v['n_interp_free']:>6}/{v['n_full']:<5}"
              f"{v['pct_dropped']:>8}%{mx:>12.3f}%"
              f"{'unchanged' if v['ranking_unchanged'] else 'CHANGED':>12}")

# ------------------------------------------------------------------ inference
d = load("r1_inference.json")
if d:
    sec("R1.4  TIME-SERIES-AWARE INFERENCE")
    print(f"{'comparison':<30}{'n':>6}{'acf1(d)':>9}{'VIF':>7}{'b':>5}"
          f"{'SE ratio':>10}{'p (DM)':>9}  {'procedure: gap':>16}{'p':>9}")
    for k, v in d.items():
        if k.startswith("_"):
            continue
        pr = v.get("procedure", {})
        print(f"{k:<30}{v['n']:>6}{v['acf1_loss_diff']:>9.3f}{v['vif']:>7.2f}"
              f"{v['block_length_pw']:>5}"
              f"{v['se_ratio_block_over_iid']:>10.2f}{v['dm_p']:>9.4f}  "
              f"{pr.get('rmse_gap_seed_mean', float('nan')):>16.3f}"
              f"{pr.get('dm_p', float('nan')):>9.4f}")
    for k, v in d.items():
        if k.startswith("_"):
            continue
        print(f"  {k}: gap(seed0) {v['rmse_gap_seed0']:+.3f}  "
              f"CI iid [{v['ci_iid'][0]:+.4f},{v['ci_iid'][1]:+.4f}]  "
              f"block [{v['ci_moving_block'][0]:+.4f},"
              f"{v['ci_moving_block'][1]:+.4f}]")
        pr = v.get("procedure")
        if pr:
            print(f"      procedure: seeds {pr['seeds']}, gap "
                  f"{pr['rmse_gap_seed_mean']:+.3f} +/- {pr['rmse_gap_seed_sd']:.3f}, "
                  f"two-way CI [{pr['ci_two_way'][0]:+.4f},"
                  f"{pr['ci_two_way'][1]:+.4f}], sign flips "
                  f"{pr['sign_flips_across_seeds']}/{len(pr['seeds'])}")
    if "_fdr" in d:
        print(f"  BH-FDR: {sum(d['_fdr']['significant'].values())}"
              f"/{d['_fdr']['family_size']} significant, "
              f"threshold p <= {d['_fdr']['bh_threshold']:.4f}")

# ------------------------------------------------------------------ templates
d = load("r2_templates.json")
if d:
    sec("R2.3  TEMPLATE STRUCTURE AND THE FOURIER SUBSTITUTION")
    an = d["analytic"]
    print(f"closed form matches implementation to "
          f"{an['closed_form_max_deviation']:.2e}")
    print(f"correlations with noise: {an['corr_with_noise']}")
    print(f"correlations noiseless:  {an['corr_noiseless']}")
    print(f"noiseless singular values {an['singular_values_noiseless']}, "
          f"rank {an['numerical_rank_noiseless']}")
    print(f"traffic template R^2 on the diurnal phasor: "
          f"{an['traffic_r2_on_diurnal_phasor']}")
    for h in (1, 24):
        if f"h{h}" not in d:
            continue
        print(f"  h={h}:")
        for mode in ("templates", "fourier", "lags_only"):
            m = d[f"h{h}"].get(mode)
            if not m:
                continue
            print(f"    {mode:<11} features {m['n_features']}  "
                  f"cmod {m['cmod']['rmse_mean']:.3f} +/- "
                  f"{m['cmod']['rmse_std']:.3f}  "
                  f"ridge {m['ridge']['rmse']:.3f}  "
                  f"delta {m.get('delta_vs_templates_pct', 0.0):+.2f}%")

print()

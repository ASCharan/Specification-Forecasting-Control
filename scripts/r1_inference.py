#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CITIES = ["delhi", "mumbai", "kolkata"]
HORIZONS = [1, 24]
COMPARISONS = [("cmod", "lstm"), ("cmod", "ridge")]


# ----------------------------------------------------------------- utilities
def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    den = np.dot(x, x)
    return np.array([np.dot(x[:n - k], x[k:]) / den for k in range(nlags + 1)])


def vif(d, nlags=None):
    """Variance inflation factor 1 + 2 sum (1 - k/n) rho_k for the sample mean."""
    n = len(d)
    if nlags is None:
        nlags = int(min(n // 4, np.floor(10 * np.log10(n))))
    r = acf(d, nlags)[1:]
    k = np.arange(1, nlags + 1)
    return float(1.0 + 2.0 * np.sum((1 - k / n) * r)), float(r[0]), nlags


def politis_white_block(x):
    """Automatic block length for the stationary and circular bootstraps."""
    x = np.asarray(x, float)
    n = len(x)
    Kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    max_lag = min(n - 1, int(np.ceil(np.sqrt(n))) + Kn)
    r = acf(x, max_lag)
    crit = 2.0 * np.sqrt(np.log10(n) / n)
    m = 1
    for i in range(1, max_lag - Kn + 1):
        if np.all(np.abs(r[i:i + Kn]) < crit):
            m = i - 1
            break
    else:
        m = max_lag // 2
    M = max(1, min(2 * max(m, 1), max_lag))

    var = np.var(x)
    R = r[:M + 1] * var                                   # autocovariances
    lam = lambda t: 1.0 if abs(t) <= 0.5 else (2 * (1 - abs(t)) if abs(t) <= 1 else 0.0)
    ks = np.arange(-M, M + 1)
    w = np.array([lam(k / M) for k in ks])
    Rk = np.array([R[abs(k)] for k in ks])
    G = float(np.sum(w * np.abs(ks) * Rk))
    g0 = float(np.sum(w * Rk))
    if g0 <= 0:
        return max(2, int(round(n ** (1 / 3))))
    b_sb = (2 * G ** 2 / (2 * g0 ** 2)) ** (1 / 3) * n ** (1 / 3)
    cap = min(3 * np.sqrt(n), n / 3.0)
    return int(max(2, min(cap, round(b_sb))))


def block_indices(n, block, rng):
    nb = int(np.ceil(n / block))
    s = rng.integers(0, n - block + 1, nb)
    return (s[:, None] + np.arange(block)[None, :]).ravel()[:n]


def moving_block_bootstrap(d, block, n_boot, rng, ea=None, eb=None):
    """Bootstrap distribution of the RMSE gap (or of mean(d) if no errors).

    The manuscript reports the gap in RMSE units, so the statistic resampled
    here is sqrt(mean(eb)) - sqrt(mean(ea)) on the same block-resampled rows.
    """
    n = len(d)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_indices(n, block, rng)
        if ea is None:
            out[i] = d[idx].mean()
        else:
            out[i] = np.sqrt(eb[idx].mean()) - np.sqrt(ea[idx].mean())
    return out


def stationary_bootstrap(d, mean_block, n_boot, rng, ea=None, eb=None):
    n = len(d)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        j = rng.integers(0, n)
        for t in range(n):
            idx[t] = j
            if rng.random() < p:
                j = rng.integers(0, n)
            else:
                j = (j + 1) % n
        out[i] = d[idx].mean()
    return out


def newey_west_var(d, lags):
    d = np.asarray(d, float)
    n = len(d)
    e = d - d.mean()
    g0 = np.dot(e, e) / n
    v = g0
    for k in range(1, lags + 1):
        gk = np.dot(e[:-k], e[k:]) / n
        v += 2.0 * (1 - k / (lags + 1)) * gk
    return float(max(v, 1e-12))


def diebold_mariano(d, horizon):
    """DM statistic with Newey-West HAC variance and the HLN correction."""
    n = len(d)
    auto = int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    lags = max(horizon - 1, auto)
    v = newey_west_var(d, lags)
    dm = d.mean() / np.sqrt(v / n)
    corr = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm), float(dm_hln), float(p), int(lags)


def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    if k:
        rej[order[:k]] = True
    return rej, float(thresh[k - 1]) if k else 0.0


def rmse_gap(y, pa, pb):
    """RMSE(b) - RMSE(a): positive means b is worse than a."""
    return float(np.sqrt(np.mean((y - pb) ** 2)) - np.sqrt(np.mean((y - pa) ** 2)))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="results/predictions_crossing")
    ap.add_argument("--variant", default="main")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="results/r1_inference.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    res, pvals, keys = {}, [], []

    for city in CITIES:
        for h in HORIZONS:
            f = Path(args.pred_dir) / f"{city}_h{h}_off1_{args.variant}.npz"
            if not f.exists():
                print(f"missing {f}, skipping", flush=True)
                continue
            z = np.load(f)
            y = z["y_true"].astype(float)
            seeds_of = lambda m: sorted(
                int(k.split("_s")[1]) for k in z.files if k.startswith(m + "_s"))

            for a, b in COMPARISONS:                    # b is compared to a
                key = f"{city}_h{h}_{b}_vs_{a}"
                # ---- instance level: seed 0 (or the single available column)
                pa = z[f"{a}_s0"] if f"{a}_s0" in z.files else z[a]
                pb = z[f"{b}_s0"] if f"{b}_s0" in z.files else z[b]
                pa, pb = pa.astype(float), pb.astype(float)
                d0 = (y - pb) ** 2 - (y - pa) ** 2       # >0 means b worse

                v, rho1, nl = vif(d0)
                blk = politis_white_block(d0)
                iid = rng.standard_normal(0)             # keep rng deterministic
                # i.i.d. bootstrap for comparison with the submitted intervals
                n = len(d0)
                ea0 = (y - pa) ** 2
                eb0 = (y - pb) ** 2
                idx = rng.integers(0, n, (args.n_boot, n))
                iid_draws = (np.sqrt(eb0[idx].mean(axis=1))
                             - np.sqrt(ea0[idx].mean(axis=1)))
                mbb = moving_block_bootstrap(d0, blk, args.n_boot, rng, ea0, eb0)
                sb = stationary_bootstrap(d0, blk, min(args.n_boot, 1500), rng,
                                          ea0, eb0)
                dm, dm_hln, p_dm, lags = diebold_mariano(d0, h)

                sens = {int(bl): [float(x) for x in
                                  np.percentile(
                                      moving_block_bootstrap(d0, bl, 1500, rng,
                                                             ea0, eb0),
                                      [2.5, 97.5])]
                        for bl in (6, 12, 24, 48, 72) if bl < n // 3}

                entry = {
                    "n": int(n),
                    "rmse_gap_seed0": rmse_gap(y, pa, pb),
                    "acf1_loss_diff": round(rho1, 4),
                    "vif": round(v, 3), "vif_lags": nl,
                    "se_ratio_block_over_iid": round(
                        float(mbb.std() / iid_draws.std()), 3),
                    "block_length_pw": blk,
                    "ci_iid": [float(x) for x in
                               np.percentile(iid_draws, [2.5, 97.5])],
                    "ci_moving_block": [float(x) for x in
                                        np.percentile(mbb, [2.5, 97.5])],
                    "ci_stationary": [float(x) for x in
                                      np.percentile(sb, [2.5, 97.5])],
                    "ci_block_sensitivity": sens,
                    "dm": round(dm, 3), "dm_hln": round(dm_hln, 3),
                    "dm_p": float(p_dm), "hac_lags": lags,
                    "mean_loss_diff": float(d0.mean()),
                }

                # ---- procedure level: marginalise over initialisation ------
                sa, sb_ = seeds_of(a), seeds_of(b)
                # a deterministic model (ridge, persistence) has no seeds: hold
                # it fixed and marginalise over the seeds of the other model
                if sa and sb_:
                    common = sorted(set(sa) & set(sb_))
                elif sb_:
                    common = sb_
                elif sa:
                    common = sa
                else:
                    common = []
                if True:
                    get = lambda m, s: (z[f"{m}_s{s}"] if f"{m}_s{s}" in z.files
                                        else z[m]).astype(float)
                    if len(common) >= 2:
                        D = np.stack([(y - get(b, s)) ** 2 - (y - get(a, s)) ** 2
                                      for s in common])          # (S, n)
                        dbar = D.mean(axis=0)
                        vb, rb, _ = vif(dbar)
                        blkb = politis_white_block(dbar)
                        # two-way: resample time blocks and seeds together
                        EA = np.stack([(y - get(a, s)) ** 2 for s in common])
                        EB = np.stack([(y - get(b, s)) ** 2 for s in common])
                        draws = np.empty(args.n_boot)
                        for i in range(args.n_boot):
                            s_idx = rng.integers(0, len(common), len(common))
                            t_idx = block_indices(n, blkb, rng)
                            draws[i] = (
                                np.sqrt(EB[np.ix_(s_idx, t_idx)].mean())
                                - np.sqrt(EA[np.ix_(s_idx, t_idx)].mean()))
                        dm2, dm2h, p2, _ = diebold_mariano(dbar, h)
                        gaps = [rmse_gap(y, get(a, s), get(b, s))
                                for s in common]
                        entry["procedure"] = {
                            "seeds": common,
                            "rmse_gap_seed_mean": float(np.mean(gaps)),
                            "rmse_gap_seed_sd": float(np.std(gaps)),
                            "acf1_loss_diff": round(rb, 4),
                            "vif": round(vb, 3),
                            "block_length_pw": blkb,
                            "ci_two_way": [float(x) for x in
                                           np.percentile(draws, [2.5, 97.5])],
                            "mean_loss_diff": float(dbar.mean()),
                            "dm_hln": round(dm2h, 3), "dm_p": float(p2),
                            "sign_flips_across_seeds": int(
                                np.sum(np.sign(gaps) != np.sign(np.mean(gaps)))),
                        }
                        pvals.append(p2)
                        keys.append(key)
                res[key] = entry
                pr = entry.get("procedure", {})
                print(f"{key:<34} gap0 {entry['rmse_gap_seed0']:+.3f} "
                      f"acf1(d) {entry['acf1_loss_diff']:+.3f} "
                      f"VIF {entry['vif']:.2f} b={blk} "
                      f"SE x{entry['se_ratio_block_over_iid']:.2f} "
                      f"p_DM {entry['dm_p']:.4f} | proc gap "
                      f"{pr.get('rmse_gap_seed_mean', float('nan')):+.3f} "
                      f"p {pr.get('dm_p', float('nan')):.4f}", flush=True)

    if pvals:
        rej, thr = bh_fdr(pvals, 0.05)
        res["_fdr"] = {"family_size": len(pvals), "alpha": 0.05,
                       "bh_threshold": thr,
                       "significant": {k: bool(r) for k, r in zip(keys, rej)}}
        print("\nBenjamini-Hochberg at 5%: "
              f"{int(rej.sum())}/{len(pvals)} comparisons significant "
              f"(threshold p <= {thr:.4f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

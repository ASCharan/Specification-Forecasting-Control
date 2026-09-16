#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate.box_closure import BoxModelEnv, make_env
from urbanclimate.envs import REWARD_WEIGHTS, fixed_policy
from urbanclimate.ppo import (EVAL_SEED_BASE, assert_disjoint_seeds,
                              evaluate_policy, torch_policy_fn, train_ppo)

torch.set_num_threads(1)

BOX_EMISSION_SCALE = 1.22          # restores Delhi-plausible uncontrolled level
SWEEP_SEED_BASE = 90_000           # disjoint from training and from Table 6 eval
ACTION_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)
W1_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
W3_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
REGIME_POINTS = [(0.2, 0.20), (0.4, 0.20), (0.4, 0.10), (0.4, 0.05), (0.8, 0.05)]


def env_factory(closure, weights=REWARD_WEIGHTS, episode_hours=168):
    def build(seed):
        kw = dict(episode_hours=episode_hours, weights=weights, seed=seed)
        if closure == "box":
            return BoxModelEnv(emission_scale=BOX_EMISSION_SCALE, **kw)
        return make_env(closure, **kw)
    return build


def score_policy(build, policy_fn, n_episodes, base_seed=EVAL_SEED_BASE):
    env = build(base_seed)
    return evaluate_policy(env, policy_fn, n_episodes=n_episodes,
                           base_seed=base_seed)


def diurnal_profile(build, n_episodes=30, warmup=48):
    tot, cnt = np.zeros(24), np.zeros(24)
    for e in range(n_episodes):
        env = build(EVAL_SEED_BASE + e)
        env.reset(seed=EVAL_SEED_BASE + e)
        done, t = False, 0
        while not done:
            _, _, done, info = env.step(np.zeros(5))
            if t >= warmup:
                tot[t % 24] += info["pm25"]
                cnt[t % 24] += 1
            t += 1
    prof = tot / np.maximum(cnt, 1)
    return prof, int(np.argmin(prof)), int(np.argmax(prof))


def channel_split(build, n_episodes=30):
    """PM2.5 with one action group at maximum and the other at zero."""
    def const(vec):
        v = np.asarray(vec, float)
        return lambda obs: v
    groups = {
        "none": np.zeros(5),
        "traffic_only": np.array([1, 1, 1, 0, 0], float),
        "enforcement_only": np.array([0, 0, 0, 1, 1], float),
        "both": np.ones(5),
    }
    out = {}
    for k, v in groups.items():
        out[k] = score_policy(build, const(v), n_episodes)["mean_pm25"]
    base = out["none"]
    for k in ("traffic_only", "enforcement_only", "both"):
        out[k + "_reduction_pct"] = round(100 * (1 - out[k] / base), 1)
    return out


def mean_reward_constant(closure, weights, u_const, n_episodes, episode_hours=168):
    build = env_factory(closure, weights, episode_hours)
    tot, n = 0.0, 0
    for e in range(n_episodes):
        env = build(SWEEP_SEED_BASE + e)
        env.reset(seed=SWEEP_SEED_BASE + e)
        u = np.full(env.act_dim, u_const, float)
        done = False
        while not done:
            _, r, done, _ = env.step(u)
            tot += r
            n += 1
    return tot / n


def best_constant(closure, weights, n_episodes):
    rew = {float(u): mean_reward_constant(closure, weights, float(u), n_episodes)
           for u in ACTION_GRID}
    u_star = max(rew, key=rew.get)
    return u_star, rew[u_star], rew[0.0]


def run_closure(closure, episodes, sweep_episodes, seeds, eval_episodes):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\nCLOSURE: {closure}\n{'='*70}", flush=True)
    build = env_factory(closure)
    out = {"closure": closure}

    # ---- Table 6: policy comparison -----------------------------------
    assert_disjoint_seeds(len(seeds), episodes, eval_episodes)
    rows = {}
    for kind in ("none", "random", "half", "max"):
        rows[kind] = score_policy(build, fixed_policy(kind), eval_episodes)
        print(f"  {kind:<8} reward {rows[kind]['avg_reward']:+.4f}  "
              f"cost {rows[kind]['cost']:.3f}  "
              f"pm25 {rows[kind]['mean_pm25']:.2f}", flush=True)

    ppo_rows, ret_sd = [], []
    for s in seeds:
        env = build(s)
        policy, returns, diag = train_ppo(env, episodes=episodes, seed=s,
                                          log_every=0)
        ppo_rows.append(score_policy(build, torch_policy_fn(policy),
                                     eval_episodes))
        # per-update return spread over the last third of training: the scale
        # a policy gradient must beat to detect an advantage
        tail = np.asarray(returns[-episodes // 3:], float)
        ret_sd.append(float(tail.std()))
        print(f"  ppo seed {s}: reward {ppo_rows[-1]['avg_reward']:+.4f}  "
              f"cost {ppo_rows[-1]['cost']:.3f}  "
              f"pm25 {ppo_rows[-1]['mean_pm25']:.2f}  "
              f"return sd {ret_sd[-1]:.2f}  [{diag['train_seconds']}s]",
              flush=True)

    base_pm = rows["none"]["mean_pm25"]
    agg = lambda k: (float(np.mean([r[k] for r in ppo_rows])),
                     float(np.std([r[k] for r in ppo_rows])))
    rows["ppo"] = {
        "avg_reward": agg("avg_reward")[0], "avg_reward_sd": agg("avg_reward")[1],
        "cost": agg("cost")[0], "cost_sd": agg("cost")[1],
        "mean_pm25": agg("mean_pm25")[0], "mean_pm25_sd": agg("mean_pm25")[1],
        "per_seed": ppo_rows,
    }
    for k, r in rows.items():
        r["reduction_pct"] = round(100 * (1 - r["mean_pm25"] / base_pm), 1)
    rows["ppo"]["reduction_pct_sd"] = round(
        100 * float(np.std([r["mean_pm25"] for r in ppo_rows])) / base_pm, 1)
    out["policy_table"] = rows
    out["return_sd_mean"] = float(np.mean(ret_sd))
    out["return_sd_per_seed"] = ret_sd

    # ---- diurnal profile and channel split ----------------------------
    prof, hmin, hmax = diurnal_profile(build)
    out["diurnal"] = {"profile": [round(float(x), 2) for x in prof],
                      "hour_of_min": hmin, "hour_of_max": hmax,
                      "min": round(float(prof.min()), 2),
                      "max": round(float(prof.max()), 2)}
    print(f"  diurnal minimum at {hmin:02d}:00, maximum at {hmax:02d}:00",
          flush=True)
    out["channels"] = channel_split(build)
    print(f"  channels: traffic {out['channels']['traffic_only_reduction_pct']}%, "
          f"enforcement {out['channels']['enforcement_only_reduction_pct']}%, "
          f"both {out['channels']['both_reduction_pct']}%", flush=True)

    # ---- 72-point weighting map ---------------------------------------
    _, w2, _, w4, w5 = REWARD_WEIGHTS
    grid = {}
    print("  weighting map:", flush=True)
    for w1 in W1_GRID:
        row = []
        for w3 in W3_GRID:
            w = (w1, w2, w3, w4, w5)
            u_star, r_star, r_zero = best_constant(closure, w, sweep_episodes)
            grid[f"{w1:.2f}|{w3:.2f}"] = {
                "w1": w1, "w3": w3, "u_star": u_star,
                "gain_over_inaction": r_star - r_zero,
                "regime": "inaction" if u_star == 0.0 else "active"}
            row.append(f"{u_star:.2f}")
        print(f"    w1={w1:.2f}  u* = {' '.join(row)}", flush=True)
    out["weighting_map"] = grid
    gains = [g["gain_over_inaction"] for g in grid.values()]
    out["gain_range"] = [float(min(gains)), float(max(gains))]
    out["u_star_min"] = float(min(g["u_star"] for g in grid.values()))
    out["n_inaction_optimal"] = sum(1 for g in grid.values()
                                    if g["regime"] == "inaction")

    # ---- value recovered at five weightings ---------------------------
    regime = []
    for w1, w3 in REGIME_POINTS:
        w = (w1, w2, w3, w4, w5)
        u_star, r_star, r_zero = best_constant(closure, w, sweep_episodes)
        avail = r_star - r_zero
        env = env_factory(closure, w)(0)
        policy, returns, _ = train_ppo(env, episodes=episodes, seed=0,
                                       log_every=0)
        ev = score_policy(env_factory(closure, w), torch_policy_fn(policy),
                          20, base_seed=SWEEP_SEED_BASE + 5_000)
        got = ev["avg_reward"] - r_zero
        rec = 100 * got / avail if abs(avail) > 1e-9 else float("nan")
        tail = np.asarray(returns[-episodes // 3:], float)
        regime.append({"w1": w1, "w3": w3, "u_star": u_star,
                       "value_available": round(float(avail), 4),
                       "ppo_cost": round(float(ev["cost"]), 4),
                       "ppo_reward": round(float(ev["avg_reward"]), 4),
                       "recovered_pct": round(float(rec), 1),
                       "return_sd": round(float(tail.std()), 3)})
        print(f"    w1={w1} w3={w3}: u*={u_star:.2f} avail={avail:.4f} "
              f"cost={ev['cost']:.3f} recovered={rec:.1f}%", flush=True)
    out["value_recovered"] = regime
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closures", nargs="+",
                    default=["saturating", "exp", "box"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--sweep-episodes", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--out", default="results/r1_control_rerun.json")
    args = ap.parse_args()

    res = {"config": vars(args), "box_emission_scale": BOX_EMISSION_SCALE}
    for c in args.closures:
        res[c] = run_closure(c, args.episodes, args.sweep_episodes,
                             args.seeds, args.eval_episodes)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

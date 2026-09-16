#!/usr/bin/env python3
"""Train the PPO-GAE controller and compare it with the fixed policies.

Usage:
  python scripts/03_train_ppo.py --seeds 0 1 2 --episodes 300
  python scripts/03_train_ppo.py --smoke
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate.envs import PollutantDynamicsEnv, fixed_policy
from urbanclimate.ppo import (assert_disjoint_seeds, evaluate_policy,
                              torch_policy_fn, train_ppo)
from urbanclimate.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--episode-hours", type=int, default=168)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--closure", default="exp", choices=["exp", "saturating"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt-dir", default="checkpoints/ppo")
    args = ap.parse_args()

    if args.smoke:
        args.episodes, args.seeds, args.eval_episodes = 10, [0], 5

    # Fails loudly if any training episode seed collides with an evaluation
    # seed. Under the pre-August-2026 scheme, training seed 1 covered every
    # evaluation episode.
    assert_disjoint_seeds(max(args.seeds) + 1, args.episodes,
                          args.eval_episodes)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    env = PollutantDynamicsEnv(episode_hours=args.episode_hours,
                               closure=args.closure)

    table, curves = {}, {}
    for kind, label in [("none", "no_control"), ("random", "random"),
                        ("half", "fixed_0.5"), ("max", "fixed_max")]:
        table[label] = evaluate_policy(
            env, fixed_policy(kind, rng=np.random.default_rng(0)),
            n_episodes=args.eval_episodes,
        )
        print(f"{label:<12} {table[label]}", flush=True)

    ppo_rows, diags = [], []
    for seed in args.seeds:
        print(f"\ntraining PPO seed {seed} "
              f"({args.episodes} episodes x {args.episode_hours} simulated hours "
              f"= {args.episodes * args.episode_hours} env steps)", flush=True)
        policy, returns, diag = train_ppo(env, episodes=args.episodes, seed=seed)
        row = evaluate_policy(env, torch_policy_fn(policy),
                              n_episodes=args.eval_episodes)
        ppo_rows.append(row)
        diags.append(diag)
        curves[f"seed{seed}"] = returns
        ck = Path(args.ckpt_dir) / f"ppo_seed{seed}.pt"
        torch.save({"state_dict": policy.state_dict(), "seed": seed,
                    "config": vars(args), "eval": row, "diagnostics": diag,
                    "returns": returns}, ck)
        print(f"  saved {ck}  eval={row}  {diag['train_seconds']}s", flush=True)

    agg = {k: (float(np.mean([r[k] for r in ppo_rows])),
               float(np.std([r[k] for r in ppo_rows])))
           for k in ppo_rows[0]}
    table["ppo"] = {k: v[0] for k, v in agg.items()}
    table["ppo_std"] = {k: v[1] for k, v in agg.items()}

    base = table["no_control"]["mean_pm25"]
    for k in ("random", "fixed_0.5", "fixed_max", "ppo"):
        table[k]["reduction_pct"] = round(
            100.0 * (base - table[k]["mean_pm25"]) / base, 1)

    save_json({"config": vars(args), "table": table, "curves": curves,
               "diagnostics": diags},
              f"results/control{'_smoke' if args.smoke else ''}.json")

    print("\n" + "=" * 70)
    print(f"{'policy':<14}{'reward':>10}{'cost':>9}{'PM2.5':>10}{'reduction':>12}")
    print("-" * 70)
    for k in ("no_control", "random", "fixed_0.5", "fixed_max", "ppo"):
        v = table[k]
        print(f"{k:<14}{v['avg_reward']:>10.3f}{v['cost']:>9.3f}"
              f"{v['mean_pm25']:>10.2f}{v.get('reduction_pct', 0):>11.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

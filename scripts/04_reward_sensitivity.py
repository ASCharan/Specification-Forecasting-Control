#!/usr/bin/env python3
"""Reproduce the reward-design result: the benefit-cost weighting selects the
control regime. Under (w1, w3) = (0.4, 0.2) the optimum is the zero-action
policy; under (0.8, 0.05) intervention carries positive net value.

Usage:  python scripts/04_reward_sensitivity.py [--episodes 300]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate.envs import PollutantDynamicsEnv
from urbanclimate.ppo import evaluate_policy, torch_policy_fn, train_ppo
from urbanclimate.utils import save_json

SETTINGS = {"initial (0.4, 0.20)": (0.4, 0.3, 0.20, 0.1, 0.15),
            "final   (0.8, 0.05)": (0.8, 0.3, 0.05, 0.1, 0.15)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = {}
    for label, weights in SETTINGS.items():
        env = PollutantDynamicsEnv(weights=weights)
        policy, _, diag = train_ppo(env, episodes=args.episodes, seed=args.seed,
                                    log_every=0)
        row = evaluate_policy(env, torch_policy_fn(policy),
                              n_episodes=args.eval_episodes)
        row["train_seconds"] = diag["train_seconds"]
        out[label] = row
        print(f"{label}: cost={row['cost']:.3f}  PM2.5={row['mean_pm25']:.2f}  "
              f"reward={row['avg_reward']:.3f}", flush=True)

    save_json(out, "results/reward_sensitivity.json")
    print("\nA cost near 0.000 in the first row is the zero-action policy. "
          "It is NOT the optimum of that reward: see scripts/13_regime_diagram.py, "
          "which finds a best constant action of u=0.25 worth 0.050 reward units "
          "per step more than inaction at this weighting. PPO recovers 1.7 per "
          "cent of that, so the collapse is an optimisation failure.")


if __name__ == "__main__":
    main()

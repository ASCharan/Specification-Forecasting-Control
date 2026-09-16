#!/usr/bin/env python3
"""Sensitivity of the policy comparison to the dispersion closure of Eq. 1.

Compares beta*exp(-kappa*h) against beta*(1 - exp(-kappa*h)), which reverses
the mixing-height dependence so that removal increases with layer depth.

Usage:  python scripts/05_dispersion_sensitivity.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate.envs import PollutantDynamicsEnv, fixed_policy
from urbanclimate.ppo import evaluate_policy
from urbanclimate.utils import save_json


def diurnal_profile(closure, u, episodes=8, hours=168):
    env = PollutantDynamicsEnv(episode_hours=hours, closure=closure)
    by_hour = {h: [] for h in range(24)}
    for k in range(episodes):
        obs = env.reset(seed=500 + k)
        done, t = False, 0
        while not done:
            obs, _, done, info = env.step(u)
            t += 1
            if t > 24:                       # discard spin-up
                by_hour[int(t % 24)].append(info["pm25"])
    return np.array([np.mean(by_hour[h]) for h in range(24)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    out = {}
    for closure in ("exp", "saturating"):
        env = PollutantDynamicsEnv(closure=closure)
        base = evaluate_policy(env, fixed_policy("none"),
                               n_episodes=args.eval_episodes)["mean_pm25"]
        row = {"uncontrolled_pm25": round(base, 2)}
        for kind, label in [("half", "u=0.5"), ("max", "u=1")]:
            m = evaluate_policy(env, fixed_policy(kind),
                                n_episodes=args.eval_episodes)["mean_pm25"]
            row[f"reduction_{label}"] = round(100 * (base - m) / base, 1)
        prof = diurnal_profile(closure, np.zeros(5))
        row["peak_hour"] = int(np.argmax(prof))
        row["trough_hour"] = int(np.argmin(prof))
        out[closure] = row
        print(f"{closure:<11} {row}", flush=True)

    save_json(out, "results/dispersion_sensitivity.json")
    print("\nThe reduction percentages should agree to within a fraction of a "
          "point; the trough hour should not.")


if __name__ == "__main__":
    main()

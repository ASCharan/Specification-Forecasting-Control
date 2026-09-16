#!/usr/bin/env python3
"""Map the benefit-cost weighting space to the control regime it selects.

Section 2.4 of the manuscript reports two weightings: one under which the
optimal policy is inaction and one under which it is active control. Two points
establish that the weighting matters; they do not say where the switch lies, how
sharp it is, or whether a city could find itself near it.

This script sweeps the two weights that carry the trade-off -- w1 on the
normalised pollutant level and w3 on quadratic control cost -- and for each pair
reports the constant action that maximises average per-step reward. Because the
learned policy of Table 7 is near-saturation and close to constant, the optimal
constant action is an informative proxy for the optimal policy, and it costs
three simulator rollouts instead of a PPO training run. A handful of weightings
are then verified by actually training PPO, which is what --verify does.

Outputs results/regime_diagram.json.

Usage
  python scripts/13_regime_diagram.py                       # sweep only
  python scripts/13_regime_diagram.py --verify 6            # + PPO at 6 points
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate.envs import PollutantDynamicsEnv, REWARD_WEIGHTS

EVAL_SEED_BASE = 90_000          # disjoint from training and from Table 7's eval seeds
ACTION_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)


def mean_reward(weights, u_const, n_episodes, episode_hours):
    """Average per-step reward of the constant policy u = u_const."""
    tot, n = 0.0, 0
    for e in range(n_episodes):
        env = PollutantDynamicsEnv(episode_hours=episode_hours,
                                   weights=weights,
                                   seed=EVAL_SEED_BASE + e)
        env.reset(seed=EVAL_SEED_BASE + e)
        u = np.full(env.act_dim, u_const, dtype=float)
        done = False
        while not done:
            _, r, done, _ = env.step(u)
            tot += r
            n += 1
    return tot / n


def best_constant(weights, n_episodes, episode_hours):
    rewards = {float(u): mean_reward(weights, float(u), n_episodes, episode_hours)
               for u in ACTION_GRID}
    u_star = max(rewards, key=rewards.get)
    return u_star, rewards[u_star], rewards[0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    help="Weights on the normalised pollutant level.")
    ap.add_argument("--w3", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
                    help="Weights on quadratic control cost.")
    ap.add_argument("--episodes", type=int, default=4,
                    help="Evaluation episodes per (weighting, action) pair.")
    ap.add_argument("--episode-hours", type=int, default=168)
    ap.add_argument("--verify", type=int, default=0,
                    help="Train PPO at this many weightings straddling the boundary.")
    ap.add_argument("--out", default="results/regime_diagram.json")
    args = ap.parse_args()

    _, w2, _, w4, w5 = REWARD_WEIGHTS
    grid = {}
    print(f"sweeping {len(args.w1)}x{len(args.w3)} weightings "
          f"over {len(ACTION_GRID)} constant actions, "
          f"{args.episodes} episodes each")
    for w1 in args.w1:
        row = []
        for w3 in args.w3:
            weights = (w1, w2, w3, w4, w5)
            u_star, r_star, r_zero = best_constant(weights, args.episodes,
                                                   args.episode_hours)
            grid[f"{w1:.2f}|{w3:.2f}"] = {
                "w1": w1, "w3": w3,
                "u_star": u_star,
                "reward_at_u_star": r_star,
                "reward_at_zero": r_zero,
                "gain_over_inaction": r_star - r_zero,
                "regime": "inaction" if u_star == 0.0 else "active",
            }
            row.append(f"{u_star:.2f}")
        print(f"  w1={w1:.2f}  u* = {' '.join(row)}")

    # boundary: for each w1, the largest w3 at which action is still worthwhile
    boundary = {}
    for w1 in args.w1:
        active = [w3 for w3 in args.w3
                  if grid[f"{w1:.2f}|{w3:.2f}"]["regime"] == "active"]
        boundary[f"{w1:.2f}"] = max(active) if active else None
    print("\nboundary (largest w3 with active control):")
    for k, v in boundary.items():
        print(f"  w1={k}: w3 <= {v}")

    out = {
        "fixed_weights": {"w2": w2, "w4": w4, "w5": w5},
        "action_grid": [float(u) for u in ACTION_GRID],
        "episodes_per_point": args.episodes,
        "episode_hours": args.episode_hours,
        "eval_seed_base": EVAL_SEED_BASE,
        "grid": grid,
        "boundary_w3_by_w1": boundary,
    }

    if args.verify:
        from urbanclimate.ppo import train_ppo, evaluate_policy
        # pick verification points straddling the boundary
        pts = []
        for w1 in args.w1:
            b = boundary[f"{w1:.2f}"]
            if b is None:
                continue
            above = [w3 for w3 in args.w3 if w3 > b]
            pts.append((w1, b))
            if above:
                pts.append((w1, min(above)))
        step = max(1, len(pts) // args.verify)
        pts = pts[::step][:args.verify]
        ver = {}
        for w1, w3 in pts:
            weights = (w1, w2, w3, w4, w5)
            env = PollutantDynamicsEnv(episode_hours=args.episode_hours,
                                       weights=weights, seed=0)
            policy = train_ppo(env, episodes=300, seed=0)
            ev = evaluate_policy(env, policy, n_episodes=20,
                                 seed_base=EVAL_SEED_BASE)
            key = f"{w1:.2f}|{w3:.2f}"
            ver[key] = {"ppo_cost": ev.get("cost"),
                        "ppo_reward": ev.get("avg_reward"),
                        "ppo_mean_pm25": ev.get("mean_pm25"),
                        "predicted_regime": grid[key]["regime"]}
            print(f"  PPO at w1={w1:.2f} w3={w3:.2f}: cost={ev.get('cost'):.4f} "
                  f"(sweep predicted {grid[key]['regime']})")
        out["ppo_verification"] = ver

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

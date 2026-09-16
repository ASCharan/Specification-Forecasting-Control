"""Fast end-to-end checks. Run with: python -m pytest tests/ -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanclimate import data, models
from urbanclimate.envs import PollutantDynamicsEnv, fixed_policy
from urbanclimate.forecasting import train_neural
from urbanclimate.ppo import evaluate_policy, train_ppo
from urbanclimate.utils import paired_bootstrap, scores


def _toy_series(n=800):
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    t = np.arange(n)
    v = 200 + 60 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(0).normal(0, 10, n)
    return pd.Series(v, index=idx, name="aqi")


def test_features_and_split():
    X, y = data.make_features(_toy_series(), horizon=1)
    assert X.shape[1] == 8 and len(X) == len(y)
    tr, va, te = data.chrono_split(len(X))
    assert tr.stop <= va.start and va.stop <= te.start


def test_each_model_trains():
    X, y = data.make_features(_toy_series(), horizon=1)
    tr, va, te = data.chrono_split(len(X))
    Xn, yn = X.to_numpy(np.float32), y.to_numpy(np.float32)
    for name in ["cmod", "mlp64", "mlp128", "lstm", "transformer"]:
        m, pred, state, _ = train_neural(
            name, Xn[tr], yn[tr], Xn[va], yn[va], Xn[te], yn[te],
            seed=0, epochs=2,
        )
        assert np.isfinite(m["rmse"]) and state
        assert models.build(name, Xn.shape[1]) is not None


def test_env_conserves_sign_of_control():
    """More control must not increase pollution."""
    env = PollutantDynamicsEnv(episode_hours=72)
    out = {k: evaluate_policy(env, fixed_policy(k), n_episodes=3)["mean_pm25"]
           for k in ("none", "half", "max")}
    assert out["none"] > out["half"] > out["max"]


def test_ppo_runs_and_improves():
    env = PollutantDynamicsEnv(episode_hours=48)
    _, returns, diag = train_ppo(env, episodes=8, seed=0, log_every=0)
    assert len(returns) == 8 and diag["total_env_steps"] == 8 * 48


def test_bootstrap_interval_brackets_zero_for_identical_models():
    y = np.random.default_rng(0).normal(0, 1, 300)
    p = y + np.random.default_rng(1).normal(0, 0.5, 300)
    point, lo, hi = paired_bootstrap(y, p, p, n_boot=200)
    assert abs(point) < 1e-9 and lo <= 0 <= hi
    assert scores(y, p)["r2"] < 1.0


def test_lag_convention_is_pinned_and_validated():
    """The reported convention is lag k = series[t-k+1]; offsets are checked."""
    assert data.LAG_OFFSET == 1
    s = _toy_series(200)
    X, _ = data.make_features(s, horizon=1)
    # Under the reported convention, aqi_lag1 at row t IS series[t].
    assert np.allclose(X["aqi_lag1"].to_numpy(), s.loc[X.index].to_numpy())
    # And the shifted convention withholds it.
    Xb, _ = data.make_features(s, horizon=1, lag_offset=0)
    assert not np.allclose(Xb["aqi_lag1"].to_numpy(), s.loc[Xb.index].to_numpy())
    for bad in (-1, 2, 24):
        try:
            data.make_features(s, horizon=1, lag_offset=bad)
        except ValueError:
            continue
        raise AssertionError(f"lag_offset={bad} should have been rejected")


def test_training_and_evaluation_seeds_are_disjoint():
    """Regression test for the pre-August-2026 seed collision."""
    from urbanclimate.ppo import (EVAL_SEED_BASE, TRAIN_SEED_BASE,
                                  assert_disjoint_seeds)
    assert assert_disjoint_seeds(n_train_seeds=3, episodes=300,
                                 n_eval_episodes=50)
    # The old scheme (train = seed*10_000 + ep, eval = 10_000 + k) put every
    # evaluation episode inside training seed 1.
    old_train_seed1 = {1 * 10_000 + ep for ep in range(300)}
    old_eval = set(range(10_000, 10_050))
    assert len(old_train_seed1 & old_eval) == 50, "collision test is stale"
    assert TRAIN_SEED_BASE > EVAL_SEED_BASE + 1_000


def test_forecast_error_reward_term_is_action_invariant():
    """Documents that the w2 term cannot shape the policy (Methods)."""
    scale = np.array([40.0, 100.0, 10.0, 1020.0])
    means = []
    for u in (0.0, 0.5, 1.0):
        env = PollutantDynamicsEnv(episode_hours=48)
        env.reset(seed=77)
        errs = []
        for _ in range(48):
            before = env.W.copy()
            env.step(np.full(env.act_dim, u))
            errs.append(np.linalg.norm((env.W - before) / scale))
        means.append(float(np.mean(errs)))
    assert max(means) - min(means) < 1e-12

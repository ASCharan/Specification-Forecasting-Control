"""Proximal Policy Optimisation with Generalised Advantage Estimation.

Implemented from scratch (no stable-baselines dependency) so that every
hyperparameter reported in the manuscript is visible in one file. Defaults
match the Methods: clip 0.2, gamma 0.99, lambda 0.95, value coefficient 0.5,
entropy bonus 0.01, gradient-norm clip 0.5, Adam 3e-4, minibatch 64, 10 epochs
per update, one update per episode.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .utils import set_seed

# Episode-seed namespaces. Training and evaluation MUST NOT overlap: with the
# original scheme (train = seed*10_000 + ep, eval = 10_000 + k) training seed 1
# swept 10_000-10_299 and therefore covered every one of the 50 evaluation
# episodes. The two bases below are separated by more than any realistic
# episode budget, and `assert_disjoint_seeds` fails loudly if that is violated.
TRAIN_SEED_BASE = 1_000_000
EVAL_SEED_BASE = 10_000


def assert_disjoint_seeds(n_train_seeds, episodes, n_eval_episodes):
    """Raise if any training episode seed collides with an evaluation seed."""
    train = {TRAIN_SEED_BASE + s * 100_000 + e
             for s in range(n_train_seeds) for e in range(episodes)}
    evalu = set(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval_episodes))
    clash = train & evalu
    if clash:
        raise ValueError(
            f"{len(clash)} training episode seeds collide with evaluation seeds")
    return True


class ActorCritic(nn.Module):
    """Gaussian policy with sigmoid mean head, plus a separate value head."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim), nn.Sigmoid(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def distribution(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs):
        dist = self.distribution(obs)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        return raw.clamp(0.0, 1.0), raw, logp, self.critic(obs).squeeze(-1)

    def evaluate(self, obs, raw):
        dist = self.distribution(obs)
        return (
            dist.log_prob(raw).sum(-1),
            dist.entropy().sum(-1),
            self.critic(obs).squeeze(-1),
        )


def gae(rewards, values, last_value, gamma: float = 0.99, lam: float = 0.95):
    """Generalised advantage estimation over one episode."""
    adv = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + np.asarray(values, np.float32)


def train_ppo(
    env,
    episodes: int = 300,
    seed: int = 0,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    minibatch: int = 64,
    epochs_per_update: int = 10,
    device: str = "cpu",
    log_every: int = 25,
):
    """Train one controller. Returns (policy, returns, diagnostics)."""
    set_seed(seed)
    policy = ActorCritic(env.obs_dim, env.act_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    returns_log, t0 = [], time.perf_counter()
    for ep in range(episodes):
        obs = env.reset(seed=TRAIN_SEED_BASE + seed * 100_000 + ep)
        obs_buf, raw_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, raw, logp, value = policy.act(obs_t)
            nxt, reward, done, _ = env.step(action.cpu().numpy())
            obs_buf.append(obs)
            raw_buf.append(raw.cpu().numpy())
            logp_buf.append(float(logp))
            rew_buf.append(reward)
            val_buf.append(float(value))
            obs = nxt

        with torch.no_grad():
            last_value = float(
                policy.critic(torch.tensor(obs, dtype=torch.float32, device=device))
            )
        adv, ret = gae(rew_buf, val_buf, last_value, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        raw_t = torch.tensor(np.array(raw_buf), dtype=torch.float32, device=device)
        logp_old = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, device=device)
        ret_t = torch.tensor(ret, device=device)

        n = len(obs_buf)
        for _ in range(epochs_per_update):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, minibatch):
                idx = perm[start : start + minibatch]
                logp, entropy, value = policy.evaluate(obs_t[idx], raw_t[idx])
                ratio = (logp - logp_old[idx]).exp()
                surr = torch.min(
                    ratio * adv_t[idx],
                    ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[idx],
                )
                loss = (
                    -surr.mean()
                    + vf_coef * ((value - ret_t[idx]) ** 2).mean()
                    - ent_coef * entropy.mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt.step()

        returns_log.append(float(np.sum(rew_buf)))
        if log_every and (ep + 1) % log_every == 0:
            recent = np.mean(returns_log[-log_every:])
            print(f"  ep {ep+1:4d}/{episodes}  return(mean last {log_every}) "
                  f"{recent:8.2f}", flush=True)

    diagnostics = {
        "episodes": episodes,
        "train_seed_base": TRAIN_SEED_BASE,
        "steps_per_episode": env.episode_hours,
        "total_env_steps": episodes * env.episode_hours,
        "train_seconds": round(time.perf_counter() - t0, 2),
        "n_params": sum(p.numel() for p in policy.parameters()),
    }
    return policy, returns_log, diagnostics


@torch.no_grad()
def evaluate_policy(env, policy_fn, n_episodes: int = 50,
                    base_seed: int = EVAL_SEED_BASE):
    """Score a policy over matched environment seeds."""
    rewards, costs, pm = [], [], []
    for k in range(n_episodes):
        obs = env.reset(seed=base_seed + k)
        done = False
        ep_r, ep_c, ep_p = [], [], []
        while not done:
            obs, reward, done, info = env.step(policy_fn(obs))
            ep_r.append(reward)
            ep_c.append(info["cost"])
            ep_p.append(info["pm25"])
        rewards.append(np.mean(ep_r))
        costs.append(np.mean(ep_c))
        pm.append(np.mean(ep_p))
    return {
        "avg_reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
        "mean_pm25": float(np.mean(pm)),
    }


def torch_policy_fn(policy, device="cpu", deterministic=True):
    """Wrap a trained ActorCritic as a plain obs -> action callable."""
    def fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            if deterministic:
                return policy.actor(obs_t).cpu().numpy()
            action, *_ = policy.act(obs_t)
            return action.cpu().numpy()
    return fn

"""Coupled traffic-pollution simulator (Eq. 1 of the manuscript).

State: P = [PM2.5 (ug/m3), NOx (ppb), CO (ppm)].
Action: u in [0, 1]^5 -- u1, u2 signal timing; u3 speed limit; u4, u5 emission
zone enforcement. Signal and speed actions additionally reduce traffic by up
to 35 percent.

Two dispersion closures are provided. `exp` is the form used in the manuscript;
`saturating` reverses the mixing-height dependence so that removal increases
with layer depth, matching box-model behaviour. See the sensitivity analysis in
the Methods.
"""
from __future__ import annotations

import numpy as np

ALPHA1 = np.array([0.30, 0.12, 0.006])
ALPHA2 = np.array([4e-3, 1.6e-3, 8e-5])
BETA = np.array([0.55, 0.60, 0.45])
KAPPA = 1.0e-3
LAMBDA0 = 0.020
DELTA = np.array([
    [0.02, 0.02, 0.010, 0.06, 0.06],
    [0.02, 0.02, 0.015, 0.06, 0.06],
    [0.01, 0.01, 0.010, 0.04, 0.04],
])
P_REF = np.array([100.0, 40.0, 2.0])
TAU_SAFE = 15.0            # WHO 24-hour PM2.5 guideline, ug/m3
REWARD_WEIGHTS = (0.8, 0.3, 0.05, 0.1, 0.15)


class PollutantDynamicsEnv:
    """Gym-style environment integrating Eq. 1 with a forward Euler step."""

    def __init__(
        self,
        episode_hours: int = 168,
        dt: float = 1.0,
        seed: int = 0,
        closure: str = "exp",
        weights=REWARD_WEIGHTS,
        traffic_amplitude: float = 73.0,
    ):
        self.episode_hours = episode_hours
        self.dt = dt
        self.closure = closure
        self.weights = tuple(weights)
        self.traffic_amplitude = traffic_amplitude
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.act_dim = 13, 5
        self.reset()

    # -- exogenous drivers -------------------------------------------------
    def _traffic(self, t):
        h = t % 24
        morning = np.exp(-0.5 * ((h - 8.5) / 2.0) ** 2)
        evening = np.exp(-0.5 * ((h - 18.5) / 2.5) ** 2)
        return self.traffic_amplitude * (0.25 + 0.75 * (morning + 0.95 * evening))

    def _mixing_height(self, t):
        return 750.0 + 450.0 * np.sin(2 * np.pi * ((t % 24) - 6) / 24)

    def _weather(self, t):
        h = t % 24
        return np.array([
            28 + 8 * np.sin(2 * np.pi * (h - 6) / 24),
            60 - 15 * np.sin(2 * np.pi * (h - 12) / 24),
            3.5 + 1.5 * np.sin(2 * np.pi * h / 24),
            1010.0,
        ]) + self.rng.normal(0, [1.5, 5.0, 0.8, 1.0])

    def _removal(self, h):
        if self.closure == "exp":
            return BETA * np.exp(-KAPPA * h)
        if self.closure == "saturating":
            return BETA * (1.0 - np.exp(-KAPPA * h))
        raise ValueError(f"unknown closure {self.closure!r}")

    # -- gym API -----------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.P = np.array([60.0, 30.0, 1.0])
        self.u_prev = np.zeros(self.act_dim)
        self.W = self._weather(self.t)
        return self._obs()

    def _obs(self):
        return np.concatenate([
            self.W / np.array([40.0, 100.0, 10.0, 1020.0]),
            [self._traffic(self.t) / 100.0],
            self.P / P_REF,
            self.u_prev,
        ]).astype(np.float32)

    def step(self, u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        traffic = self._traffic(self.t) * (1.0 - 0.35 * u[:3].mean())
        h = self._mixing_height(self.t)
        W_next = self._weather(self.t + self.dt)

        emission = ALPHA1 * traffic + ALPHA2 * traffic ** 2
        dP = (
            emission
            - self._removal(h) * self.P
            - LAMBDA0 * self.W[2] * self.P
            - (DELTA @ u) * self.P
        )
        P_old = self.P.copy()
        self.P = np.maximum(self.P + self.dt * dP, 0.0)

        w1, w2, w3, w4, w5 = self.weights
        norm_old = np.linalg.norm(P_old / P_REF)
        norm_new = np.linalg.norm(self.P / P_REF)
        forecast_err = np.linalg.norm(
            (W_next - self.W) / np.array([40.0, 100.0, 10.0, 1020.0])
        )
        delta_poll = (norm_old - norm_new) / max(norm_old, 1e-8)

        reward = (
            -w1 * norm_new
            - w2 * forecast_err
            - w3 * float(np.sum(u ** 2))
            + w4 * delta_poll
            + w5 * float(self.P[0] < TAU_SAFE)
        )

        self.W, self.u_prev = W_next, u
        self.t += self.dt
        done = self.t >= self.episode_hours
        info = {"pm25": float(self.P[0]), "cost": float(np.linalg.norm(u))}
        return self._obs(), float(reward), done, info


def fixed_policy(kind: str, act_dim: int = 5, rng=None):
    """Non-learned reference policies used in the comparison table."""
    if kind == "none":
        return lambda obs: np.zeros(act_dim)
    if kind == "half":
        return lambda obs: np.full(act_dim, 0.5)
    if kind == "max":
        return lambda obs: np.ones(act_dim)
    if kind == "random":
        rng = rng or np.random.default_rng(0)
        return lambda obs: rng.uniform(0, 1, act_dim)
    raise ValueError(f"unknown policy {kind!r}")

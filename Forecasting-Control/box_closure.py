
from __future__ import annotations

import numpy as np

from .envs import (ALPHA1, ALPHA2, DELTA, P_REF, TAU_SAFE, PollutantDynamicsEnv)

H_REF = 750.0            # m, mean mixing height of the prescribed profile
INV_L = 0.12             # h^-1 per (m/s): ventilation over a ~30 km cell
K_DEP = np.array([0.010, 0.030, 0.002])   # h^-1, deposition / chemical loss
EMISSION_SCALE = 1.0     # set by calibrate(); see scripts/r1_control_rerun.py


class BoxModelEnv(PollutantDynamicsEnv):
    """PollutantDynamicsEnv with the well-mixed box closure."""

    def __init__(self, *args, emission_scale: float = EMISSION_SCALE,
                 inv_L: float = INV_L, k_dep=K_DEP, **kwargs):
        self.emission_scale = float(emission_scale)
        self.inv_L = float(inv_L)
        self.k_dep = np.asarray(k_dep, float)
        kwargs.setdefault("closure", "box")
        super().__init__(*args, **kwargs)

    def _removal(self, h):          # not used by step(); kept for interface
        return self.k_dep + self.inv_L * self.W[2]

    def _dhdt(self, t):
        """Analytic derivative of the prescribed mixing-height profile, m/h."""
        return 450.0 * (2 * np.pi / 24.0) * np.cos(2 * np.pi * ((t % 24) - 6) / 24)

    def step(self, u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        traffic = self._traffic(self.t) * (1.0 - 0.35 * u[:3].mean())
        h = self._mixing_height(self.t)
        dh = self._dhdt(self.t)
        W_next = self._weather(self.t + self.dt)

        emission = (ALPHA1 * traffic + ALPHA2 * traffic ** 2)
        emission = emission * self.emission_scale * (H_REF / h)

        entrainment = max(dh, 0.0) / h
        ventilation = self.inv_L * self.W[2]

        dP = (
            emission
            - entrainment * self.P
            - ventilation * self.P
            - self.k_dep * self.P
            - (DELTA @ u) * self.P
        )
        P_old = self.P.copy()
        self.P = np.maximum(self.P + self.dt * dP, 0.0)

        w1, w2, w3, w4, w5 = self.weights
        norm_old = np.linalg.norm(P_old / P_REF)
        norm_new = np.linalg.norm(self.P / P_REF)
        forecast_err = np.linalg.norm(
            (W_next - self.W) / np.array([40.0, 100.0, 10.0, 1020.0]))
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


def make_env(closure: str, **kwargs):
    """Factory over the three closures used in the revision."""
    if closure == "box":
        return BoxModelEnv(**kwargs)
    return PollutantDynamicsEnv(closure=closure, **kwargs)

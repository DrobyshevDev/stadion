"""Admission control: which jobs to let into a finite buffer.

Each arriving job shows its value before you decide. Taking it captures that
value but occupies a slot and adds to a congestion charge that is paid every
step until the server works it off. The classical answer is a fixed value
threshold — admit anything worth more than theta — and the threshold is chosen
by search on tuning seeds.

The optimum here is a threshold too, but one that moves with the queue and with
the time remaining: near the deadline a full buffer is a liability rather than
an asset. Note the timing, which the dynamic program has to respect and an agent
has to notice: the decision is made before the server's coin flip is revealed,
so what is being chosen is a bet on whether a slot will free up.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from decisionrl.baselines import best_value_threshold, value_threshold
from decisionrl.core.env import Env
from decisionrl.envs import QueueAdmissionControl

from stadion.core.agent import Agent, PolicyAgent
from stadion.core.task import Brief, Choice, Instance, Task, View
from stadion.solvers.dp import QueueSolution, solve_queue

__all__ = ["Queueing"]


class _Optimal(Agent):
    name = "optimum"

    def __init__(self, solution: QueueSolution) -> None:
        self._solution = solution

    def act(self, view: View) -> int:
        queue = int(round(float(view.obs[0]) * view.env.buffer_size))
        job_value = float(view.obs[1])
        return int(self._solution.admits(view.step, queue, job_value))


class Queueing(Task):
    name = "queueing"
    summary = (
        "Admit or reject each arriving job, balancing the value it carries "
        "against the congestion it causes."
    )
    baseline_name = "best fixed value threshold"

    def sample_params(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "buffer_size": float(rng.choice([8, 10, 12])),
            "service_prob": float(rng.uniform(0.35, 0.65)),
            "holding_cost": float(rng.uniform(0.02, 0.10)),
            "horizon": 100.0,
        }

    def build(self, params: Mapping[str, float]) -> Env:
        return QueueAdmissionControl(
            buffer_size=int(params["buffer_size"]),
            service_prob=params["service_prob"],
            holding_cost=params["holding_cost"],
            horizon=int(params["horizon"]),
        )

    def brief(self, inst: Instance) -> Brief:
        p = inst.params
        text = (
            f"You control admission to a server with {int(p['buffer_size'])} buffer slots, "
            f"for {int(p['horizon'])} steps.\n"
            f"Each step happens in this order:\n"
            f"  1. If the buffer is not empty, the server finishes one job with "
            f"probability {p['service_prob']:.2f}.\n"
            f"  2. One job arrives. You already know its value, drawn uniformly from "
            f"[0, 1). You admit or reject it.\n"
            f"  3. You pay {p['holding_cost']:.2f} for every job sitting in the buffer.\n"
            f"Admitting captures the job's value. If the buffer is full after step 1, "
            f"admitting does nothing.\n"
            f"You decide before seeing whether the server finished, so admitting into a "
            f"nearly full buffer is a bet on a slot opening up.\n"
            f"Maximise total value captured minus congestion paid."
        )
        return Brief(
            task=self.name,
            seed=inst.seed,
            horizon=int(p["horizon"]),
            text=text,
            params=p,
        )

    def view(self, env: Env, obs: np.ndarray, step: int, earned: float) -> View:
        queue = int(round(float(obs[0]) * env.buffer_size))
        job_value = float(obs[1])
        text = (
            f"Step {step + 1} of {env.horizon}. "
            f"Buffer: {queue} of {env.buffer_size}. "
            f"Incoming job is worth {job_value:.3f}. "
            f"Earned so far: {earned:.2f}."
        )
        choices = (
            Choice(value=0, label="reject", detail="capture nothing, add no congestion"),
            Choice(
                value=1,
                label="admit",
                detail=f"capture {job_value:.3f}, occupy a slot until served",
            ),
        )
        return View(step=step, text=text, choices=choices, earned=earned, obs=obs, env=env)

    def baseline(self, inst: Instance) -> Agent:
        key = ("baseline", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            theta, _ = best_value_threshold(
                inst.env, episodes=self.tuning_episodes, seed=self.tuning_seed
            )
            cached = float(theta)
            self._memo[key] = cached
        return PolicyAgent(
            value_threshold(cached), name=f"{self.baseline_name} {cached:.2f}"
        )

    def _solve(self, inst: Instance) -> QueueSolution:
        key = ("dp", inst.seed)
        cached = self._memo.get(key)
        if cached is None:
            p = inst.params
            cached = solve_queue(
                buffer_size=int(p["buffer_size"]),
                service_prob=p["service_prob"],
                holding_cost=p["holding_cost"],
                horizon=int(p["horizon"]),
            )
            self._memo[key] = cached
        return cached

    def optimal(self, inst: Instance) -> Agent:
        return _Optimal(self._solve(inst))

    def optimal_value(self, inst: Instance) -> float:
        return self._solve(inst).value

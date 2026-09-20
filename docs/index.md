# stadion

**A proving ground for operational decisions.** An agent is scored against two
references it cannot argue with: the classical operations-research method for the
problem, and the exact optimum. The result is a normalised score with a
confidence interval — and *"indistinguishable from the classical method"* is a
first-class outcome, not a rounding error.

```bash
pip install stadion-rl
```

The import package is `stadion`; the distribution carries the `-rl` suffix
because PyPI's `stadion` belongs to an unrelated causal-modelling package.

## Why another gym

The environments an agent can be trained and measured on today are mostly code,
browsers and SaaS workflows. They share a problem: **nobody knows what the right
answer was.** A score of 61% on such a benchmark tells you an agent beat other
agents, not whether it did well.

στάδιον is both the racecourse and a unit of length. These tasks are picked so
that they can be both: each is a decision a business makes thousands of times a
day, each has a textbook method a practitioner would reach for, and each is small
enough that the true optimum can be *computed* by backward induction rather than
approximated. So every run puts three numbers on one scale.

## What a run looks like

`examples/scarcity.py` prices higher when stock is scarce relative to the time
left, lower when it is piling up. The right shape, the rule most people write
first, and obviously better than a fixed price:

```
pricing / scarcity
  instances        40
  agent               22.503
  classical           24.490
  optimum             26.006
  score               -1.310   (0 = classical, 1 = optimum;
                             one point = 1.516, 6.2% of the classical result)
  agent - classical: -1.987 [-3.038, -1.027] (-8.1%) -> worse
  agent - optimum: -3.503 [-4.403, -2.670] (-13.5%) -> worse
  optimum - classical: +1.516 [+1.265, +1.761] (+6.2%) -> better
```

It loses to a plain fixed price by 8.1%, and the interval does not touch zero. On
a leaderboard against other adaptive agents it might have looked fine.

That is the whole argument for the project: the scale has a top, so a result can
be wrong rather than merely mid-table.

## Getting started

```python
import stadion

task = stadion.get("pricing")
report = stadion.evaluate(task, my_agent, instances=30, episodes=20)
print(report.summary())
```

From the shell:

```bash
stadion brief pricing --seed 3
stadion run inventory --agent optimum --instances 30
```

Next: [the tasks](tasks.md) and how much room each one actually has, or
[writing an agent](agents.md).

`Python 3.10+` · [PyPI](https://pypi.org/project/stadion-rl/) ·
[repository](https://github.com/DrobyshevDev/stadion) · MIT

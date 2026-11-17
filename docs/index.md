# stadion

A benchmark for agents on operational decisions. An agent is compared with two
references: the classical operations-research method for the problem and the
exact optimum. The result is a normalised score with a confidence interval, and
"no measurable difference from the classical method" is one of the possible
results.

```bash
pip install stadion-rl
```

The import package is `stadion`; the distribution carries the `-rl` suffix
because PyPI's `stadion` belongs to an unrelated causal-modelling package.

## Motivation

Most environments for training and evaluating agents today are about code,
browsers and SaaS workflows. In those tasks the correct answer is usually
unknown, so a score of 61% shows how an agent compares with other agents, but
not how good its decisions are.

The tasks here are chosen so that each is a decision a business makes many times
a day, each has a standard textbook method, and each is small enough for the
exact optimum to be computed by backward induction. Every run therefore reports
three numbers on one scale: the agent, the classical method and the optimum. The
name στάδιον means both a racecourse and a unit of length.

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

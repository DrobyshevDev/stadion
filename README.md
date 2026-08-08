# stadion

**A proving ground for operational decisions.** An agent is scored against two
references it cannot argue with: the classical operations-research method for
the problem, and the exact optimum. The result is a normalised score with a
confidence interval — and *"indistinguishable from the classical method"* is a
first-class outcome, not a rounding error.

[Русская версия](README.ru.md)

---

## Why another gym

The environments an agent can be trained and measured on today are mostly code,
browsers and SaaS workflows. They share a problem: **nobody knows what the right
answer was.** A score of 61% on such a benchmark tells you an agent beat other
agents, not whether it did well.

στάδιον is both the racecourse and a unit of length. These tasks are picked so
that they can be both: each is a decision a business makes thousands of times a
day, each has a textbook method a practitioner would reach for, and each is
small enough that the true optimum can be *computed* by backward induction
rather than approximated. So every run puts three numbers on one scale.

Here is `examples/scarcity.py` — price higher when stock is scarce relative to
the time left, lower when it is piling up. The right shape, the rule most people
write first, and obviously better than a fixed price:

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

It loses to a plain fixed price by 8.1%, and the interval does not touch zero.
On a leaderboard against other adaptive agents it might have looked fine.

## Install

```bash
pip install stadion-rl
```

The import package is `stadion`; the distribution carries the `-rl` suffix
because PyPI's `stadion` belongs to an unrelated causal-modelling package.

## Use

```python
import stadion

task = stadion.get("pricing")
report = stadion.evaluate(task, my_agent, instances=30, episodes=20)
print(report.summary())

report.score              # normalised: 0 = classical method, 1 = optimum
report.vs_baseline.ci     # bootstrap interval on the paired difference
report.degenerate         # True when the classical method is already optimal
```

An agent implements one method:

```python
class MyAgent(stadion.Agent):
    name = "my-agent"

    def act(self, view: stadion.View) -> int:
        # view.text and view.choices  — what a language model reads
        # view.obs and view.env       — what a numeric policy reads
        return view.choices[0].value
```

Both surfaces are always present, so an RL policy and an LLM are scored on the
same task without either being translated through the other's interface. For a
language model, `stadion.llm.LLMAgent` takes any `prompt -> reply` callable —
no client library, no provider.

From the shell:

```bash
stadion brief pricing --seed 3
```

```bash
stadion run inventory --agent optimum --instances 30
```

## The tasks

| Task | The decision | Classical method | Optimum from |
|---|---|---|---|
| `inventory` | how much stock to order each day against Poisson demand | analytic base-stock (newsvendor critical fractile) | backward induction over on-hand stock |
| `pricing` | what price to post each period for a perishable stock with a deadline | the strongest fixed price, tuned by search | backward induction over remaining stock |
| `queueing` | admit or reject each arriving job into a finite buffer | the strongest fixed value threshold, tuned by search | backward induction with the job value integrated in closed form |

Environments and the classical policies come from
[decisionrl](https://github.com/DrobyshevDev/decisionrl) unmodified, so the
opponent is the same code that library ships and tests, not a re-implementation
written to lose.

### How much room is actually in each one

Measured over 40 instances × 20 episodes; reproduce with
`stadion run <task> --agent classical --instances 40 --episodes 20`.

| Task | Classical | Optimum | Headroom | 95% interval |
|---|---:|---:|---:|---|
| `inventory` | 204.141 | 204.890 | **+0.4%** | [+0.561, +1.031] |
| `pricing` | 24.490 | 26.006 | +6.2% | [+1.265, +1.761] |
| `queueing` | 21.911 | 25.611 | +16.9% | [+3.418, +3.988] |

The spread is the point. On `queueing` the textbook rule leaves a sixth of the
achievable value on the table. On `inventory` the newsvendor formula is within
half a percent of the exact optimum — there is almost nothing to win, and an
agent that reports a large improvement there has a bug, not a policy. A
benchmark whose tasks all have generous headroom has quietly selected for
problems where the classical answer is bad.

## The protocol

**Instances are generated, not stored.** `task.instance(seed)` draws the demand
level, the cost structure and the horizon from a documented distribution. The
numbers an agent is asked about did not exist before the run, so they cannot
have been memorised from a public dataset.

**The agent is told everything the baseline is told.** The brief states the
instance's full parameters, because the classical rule is built from those same
parameters — withholding them would not make the comparison harder, it would
make it dishonest.

**What the agent does not get is the tuning budget.** Where the classical rule
has a free parameter, it is fitted by search over practice episodes on seeds
that never appear in the evaluation set. The agent reads the brief once and
plays. A draw against a tuned classical rule is therefore a real result.

**Everything is paired.** Agent, baseline and optimum see the same instances and
the same episode seeds, and the interval is a bootstrap over instances. One
caveat is worth stating plainly: NumPy's Poisson sampler consumes a variable
amount of the random stream, so once two policies diverge their demand paths
diverge too. Pairing removes between-instance variance, which is the large term,
but not within-episode noise — which is why each instance is averaged over
several episodes before the arms are compared.

**Sometimes the ceiling is a draw.** On some instance families the textbook rule
is already indistinguishable from the optimum. There the normalised score has no
denominator, and the report says so rather than dividing by a small number and
reporting a dramatic figure. A benchmark that hides this is selling a race that
cannot be won.

## Checking the harness against itself

Every score here is measured *against* the optimum, so nothing in an ordinary
run would catch a wrong recurrence. One command does:

```bash
stadion verify
```

It computes each dynamic program's analytic value and, separately, simulates the
policy that same program emits. Two independent routes to one number; they have
to agree within Monte Carlo error. A dynamic program that quietly disagrees with
its own policy is the failure this is built to catch, and it runs in CI.

## Status

v0.1 — three tasks, exact optima, the scoring protocol. The remaining applied
environments in `decisionrl` (energy, supply chain, joint pricing-and-inventory)
follow once the protocol has settled. Not yet on PyPI.

Issues and pull requests welcome.

`Python 3.10+` · MIT

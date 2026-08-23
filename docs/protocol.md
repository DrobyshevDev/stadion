# The protocol

What the agent is given, what it is not, and how the interval is built.

## Instances are generated, not stored

`task.instance(seed)` draws the demand level, the cost structure and the horizon
from a documented distribution. The numbers an agent is asked about did not exist
before the run, so they cannot have been memorised from a public dataset.

## The agent is told everything the baseline is told

The brief states the instance's full parameters, because the classical rule is
built from those same parameters. Withholding them would not make the comparison
harder — it would make it dishonest.

## What the agent does not get is the tuning budget

Where the classical rule has a free parameter, it is fitted by search over
practice episodes on seeds that never appear in the evaluation set. The agent
reads the brief once and plays.

A draw against a tuned classical rule is therefore a real result, not a
consolation.

## Everything is paired

Agent, baseline and optimum see the same instances and the same episode seeds,
and the interval is a bootstrap over instances.

One caveat is worth stating plainly: NumPy's Poisson sampler consumes a variable
amount of the random stream, so once two policies diverge their demand paths
diverge too. Pairing removes between-instance variance, which is the large term,
but not within-episode noise — which is why each instance is averaged over
several episodes before the arms are compared.

## Sometimes the ceiling is a draw

On some instance families the textbook rule is already indistinguishable from the
optimum. There the normalised score has no denominator, and the report says so
rather than dividing by a small number and reporting a dramatic figure.

```python
report.degenerate   # True when the classical method is already optimal
```

A benchmark that hides this is selling a race that cannot be won.

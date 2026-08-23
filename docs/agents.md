# Writing an agent

An agent implements one method:

```python
import stadion


class MyAgent(stadion.Agent):
    name = "my-agent"

    def act(self, view: stadion.View) -> int:
        # view.text and view.choices  — what a language model reads
        # view.obs and view.env       — what a numeric policy reads
        return view.choices[0].value
```

## Two surfaces, one task

Both surfaces are always present, so a reinforcement learning policy and a
language model are scored on the same task without either being translated
through the other's interface.

`view.text` and `view.choices`
:   The brief as prose, and the numbered menu of actions available this step.
    This is what a language model reads.

`view.obs` and `view.env`
:   The numeric observation and the environment handle. This is what a policy
    trained on arrays reads.

An agent picks whichever it wants. Nothing about the scoring depends on the
choice, which is the point: a comparison between an LLM and a trained policy on
this task is a comparison of decisions rather than of adapters.

## Language models

`stadion.llm.LLMAgent` takes any `prompt -> reply` callable:

```python
from stadion.llm import LLMAgent

agent = LLMAgent(lambda prompt: my_model(prompt))
```

No client library and no provider — a callable is the whole contract, so the
harness never needs a key, and swapping a model is swapping a function.

## Running it

```python
task = stadion.get("pricing")
report = stadion.evaluate(task, agent, instances=30, episodes=20)

report.score              # normalised: 0 = classical method, 1 = optimum
report.vs_baseline.ci     # bootstrap interval on the paired difference
report.degenerate         # True when the classical method is already optimal
print(report.summary())
```

`report.degenerate` is worth handling rather than ignoring. On some instance
families the textbook rule is already indistinguishable from the optimum, so the
normalised score has no denominator — see [the protocol](protocol.md#sometimes-the-ceiling-is-a-draw).

## Reading the score

`0` is the tuned classical method and `1` is the exact optimum.

A score slightly above 1 is sampling noise: the optimum is optimal in
expectation, and any finite run scatters around its mean. A score *clearly* above
1, with an interval that does not reach back down to it, is a bug report rather
than a triumph — nothing beats the optimum, so it means the agent is playing a
different game from the one the dynamic program solved: a different action menu,
a different horizon, or an instance it was not given.

A negative score means the agent lost to the textbook rule. That is a normal
result and the reason the scale is drawn this way.

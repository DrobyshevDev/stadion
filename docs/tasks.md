# The tasks

Six decisions, each with a textbook method a practitioner would reach for and an
optimum small enough to compute rather than approximate.

| Task | The decision | Classical method | Optimum from |
|---|---|---|---|
| `inventory` | how much stock to order each day against Poisson demand | analytic base-stock (newsvendor critical fractile) | backward induction over on-hand stock |
| `pricing` | what price to post each period for a perishable stock with a deadline | the strongest fixed price, tuned by search | backward induction over remaining stock |
| `queueing` | admit or reject each arriving job into a finite buffer | the strongest fixed value threshold, tuned by search | backward induction with the job value integrated in closed form |
| `energy` | when to charge and discharge a battery against a daily price cycle | the strongest fixed price threshold, tuned by search | backward induction over the charge lattice |
| `supply-chain` | how much to order at two echelons, a period before it can help | per-echelon base-stock, tuned by search | backward induction over the collapsed two-dimensional state |
| `joint-pricing` | what to charge and how much to restock, decided together | the strongest static price-and-target pair, tuned jointly | backward induction over on-hand stock |

Environments and the classical policies come from
[decisionrl](https://github.com/DrobyshevDev/decisionrl) unmodified, so the
opponent is the same code that library ships and tests, not a re-implementation
written to lose.

The last two have a continuous action space, which every player here meets as the
same numbered menu — nine power settings for the battery, twenty-five order pairs
for the chain. The classical rule's real-valued action is snapped to that menu,
and its free parameter is tuned *through* the snap, so it is optimised for the
game it actually plays rather than for a continuous relaxation of it.

## How much room is actually in each one

Measured over 40 instances × 20 episodes. Reproduce with:

```bash
stadion run <task> --agent classical --instances 40 --episodes 20
```

| Task | Classical | Optimum | Headroom | 95% interval |
|---|---:|---:|---:|---|
| `inventory` | 204.141 | 204.890 | **+0.4%** | [+0.561, +1.031] |
| `joint-pricing` | 110.099 | 114.812 | +4.3% | [+3.794, +5.656] |
| `pricing` | 24.490 | 26.006 | +6.2% | [+1.265, +1.761] |
| `queueing` | 21.911 | 25.611 | +16.9% | [+3.418, +3.988] |
| `supply-chain` | −37.532 | −31.027 | +17.3% | [+5.505, +7.493] |
| `energy` | 16.778 | 21.234 | **+26.6%** | [+4.221, +4.690] |

The spread is the point. A price threshold with no forecast leaves a quarter of
the battery's value unclaimed, because it cannot decide to arrive at the evening
peak full. At the other end the newsvendor formula is within half a percent of
the exact optimum — there is almost nothing to win on `inventory`, and an agent
that reports a large improvement there has a bug, not a policy.

!!! note "What a wide benchmark would hide"

    A benchmark whose tasks all have generous headroom has quietly selected for
    problems where the classical answer is bad. Publishing the spread is what
    makes that visible.

## joint-pricing, and the claim it corrected

`joint-pricing` is the case that changed our mind about something. Its
environment is built around a coupling — the right price depends on how much
stock is on the shelf, so no fixed price can be right — and that is true. Priced
out, letting the price answer to the stock is worth 4.3%. Real, measurable, and a
good deal smaller than "no static rule is right" suggests.

The number is sensitive to how fine the price menu is — 2.2% over six prices,
3.2% over eight, 3.6% over twelve — which is why the menu is set where that has
mostly stopped moving rather than where the headline looks best.

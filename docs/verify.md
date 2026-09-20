# Checking the harness

Every score here is measured *against* the optimum. That makes the optimum the
one number nothing in an ordinary run would catch being wrong: a dynamic program
with a mistaken recurrence moves the whole scale, and every task still reports a
plausible-looking result.

One command exists for that:

```bash
stadion verify
```

It computes each dynamic program's analytic value and, separately, simulates the
policy that same program emits. Two independent routes to one number; they have
to agree within Monte Carlo error.

A dynamic program that quietly disagrees with its own policy is the failure this
is built to catch, and it runs in CI on every push.

## Why this is the check that matters

The rest of the suite tests behaviour that a wrong answer would visibly break —
an environment that rejects an illegal action, a report that carries an interval.
The optimum is different: it is the definition of the top of the scale, so
nothing above it exists to compare against.

Two derivations that have to meet is the only way to check a number with no
external reference. It is the same reason a published benchmark figure here is
reproduced by a command in the repository rather than quoted from a run someone
did once.

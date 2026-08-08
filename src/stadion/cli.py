"""Command line: look at a task, score an agent, check the harness against itself."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from stadion.core.agent import Agent, RandomAgent, ReferenceAgent
from stadion.core.runner import evaluate, play_episode
from stadion.core.task import Task, get, names, registry

__all__ = ["main"]

BUILTIN_AGENTS = ("random", "classical", "optimum")


def _agent(task: Task, which: str) -> Agent:
    if which == "random":
        return RandomAgent()
    return ReferenceAgent(task, which)


def _cmd_tasks(_: argparse.Namespace) -> int:
    for task in registry().values():
        print(f"{task.name}\n  {task.summary}\n  baseline: {task.baseline_name}")
    return 0


def _cmd_brief(args: argparse.Namespace) -> int:
    task = get(args.task)
    inst = task.instance(args.seed)
    brief = task.brief(inst)
    env = inst.env()
    obs, _ = env.reset(seed=0)
    view = task.view(env, obs, 0, 0.0)
    print(f"# {task.name}, instance seed {args.seed}\n")
    print(brief.text)
    print(f"\n{view.text}\n")
    for choice in view.choices:
        detail = f"  ({choice.detail})" if choice.detail else ""
        print(f"  {choice.value} = {choice.label}{detail}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    task = get(args.task)
    report = evaluate(
        task,
        _agent(task, args.agent),
        instances=args.instances,
        episodes=args.episodes,
    )
    print(report.summary())
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Check the dynamic programs against the policies they emit.

    The analytic value and a simulation of the optimal policy are two
    independent routes to the same number. Where they disagree beyond Monte
    Carlo error, the recurrence is wrong — and this is the only check in the
    library that can catch that, because every other number is measured against
    the optimum rather than at it.
    """
    chosen = [get(args.task)] if args.task else list(registry().values())
    failures = 0
    for task in chosen:
        for seed in range(args.instances):
            inst = task.instance(seed)
            analytic = task.optimal_value(inst)
            # Each instance gets its own block of episode seeds. Sharing one block
            # across instances correlates their sampling error, which makes a run
            # of same-signed deviations look like a systematic bug when it is not.
            base = args.episode_seed + 1_000_000 * (seed + 1)
            returns = np.array(
                [
                    play_episode(task, task.optimal(inst), inst, base + j)
                    for j in range(args.episodes)
                ]
            )
            error = float(returns.std(ddof=1) / np.sqrt(returns.size))
            z = (float(returns.mean()) - analytic) / error if error > 0 else 0.0
            ok = abs(z) <= args.max_z
            failures += not ok
            print(
                f"{'ok ' if ok else 'BAD'} {task.name:<10} seed {seed:<3} "
                f"analytic {analytic:9.3f}  simulated {returns.mean():9.3f} "
                f"+- {error:5.3f}   z = {z:+5.2f}"
            )
    if failures:
        print(
            f"\n{failures} instance(s) disagree with their own dynamic program by more "
            f"than {args.max_z} standard errors",
            file=sys.stderr,
        )
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stadion",
        description="Score an agent on operational decisions against the classical "
        "method and the exact optimum.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tasks = sub.add_parser("tasks", help="list the built-in tasks")
    p_tasks.set_defaults(func=_cmd_tasks)

    p_brief = sub.add_parser("brief", help="print what an agent reads for one instance")
    p_brief.add_argument("task", choices=names())
    p_brief.add_argument("--seed", type=int, default=0)
    p_brief.set_defaults(func=_cmd_brief)

    p_run = sub.add_parser("run", help="score an agent on a task")
    p_run.add_argument("task", choices=names())
    p_run.add_argument("--agent", choices=BUILTIN_AGENTS, default="classical")
    p_run.add_argument("--instances", type=int, default=30)
    p_run.add_argument("--episodes", type=int, default=20)
    p_run.set_defaults(func=_cmd_run)

    p_verify = sub.add_parser("verify", help="check each dynamic program against its own policy")
    p_verify.add_argument("task", nargs="?", choices=names(), default=None)
    p_verify.add_argument("--instances", type=int, default=3)
    p_verify.add_argument("--episodes", type=int, default=2000)
    p_verify.add_argument("--episode-seed", type=int, default=0)
    p_verify.add_argument(
        "--max-z",
        type=float,
        default=4.0,
        help="how many standard errors the simulation may sit from the analytic value",
    )
    p_verify.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

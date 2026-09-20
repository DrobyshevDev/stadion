#!/usr/bin/env python3
"""The headroom table has to be the table the benchmark actually produces.

Six tasks, four figures each, written out twice -- once per language. That is
forty-eight numbers, none of which recomputed itself, and they are the most
specific claims this organisation publishes anywhere: "+0.4% on inventory" is
the sentence the next paragraph leans on when it says an agent reporting a
large improvement there has a bug rather than a policy. A benchmark whose own
headline numbers are typed by hand is asking to be taken on trust, which is the
one thing it exists not to do.

Two tiers, because the six tasks do not cost the same:

  here          the task list, the size of each numbered action menu, the two
                language tables agreeing with each other, and `inventory`
                recomputed in full -- seconds, so it runs on every push

  --all         every row recomputed: 855s on an ordinary laptop, of which
                `joint-pricing` is 626 and `inventory` is 7. Too slow to sit in
                front of every push given how often a branch here has to be
                brought up to date, and run weekly instead. The README offers
                one command for all six without mentioning that one of them is
                ninety times the others, which is worth knowing before starting
                it and wondering whether it has hung

Reproduces with exactly the command the README tells a reader to use. If that
command stops producing the table, the README is wrong about how to reproduce
it, and that is worth failing over too.

    python scripts/check_readme.py
    python scripts/check_readme.py --all
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Iterator
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
READMES = ("README.md", "README.ru.md")

#: The documented instance and episode counts. Read from the README rather than
#: hard-coded: the sentence above the table promises these exact numbers, so if
#: somebody changes one, the recomputation has to follow it there or the check
#: would be measuring something the reader was never told about.
COMMAND = re.compile(r"--agent classical --instances (\d+) --episodes (\d+)")

#: One row of the headroom table. The minus signs are two different characters:
#: `supply-chain` is negative and the table uses a typographic minus.
ROW = re.compile(
    r"^\| `([a-z-]+)` \| (−?-?[\d.]+) \| (−?-?[\d.]+) \| \*{0,2}\+?([\d.]+)%\*{0,2} \| "
    r"\[([+-][\d.]+), ([+-][\d.]+)\] \|$",
    re.M,
)

#: Menu sizes the prose names in words: "nine power settings for the battery,
#: twenty-five order pairs for the chain".
MENUS = {"energy": 9, "supply-chain": 25}

#: How far a recomputed figure may sit from the published one.
#:
#: Not zero. The table is printed to three decimals, and the same run on
#: another machine can differ in the last of them through nothing worse than
#: float summation order. decisionrl already has one test that asserts a single
#: exact figure and fails on about half of its runs for exactly that reason;
#: repeating it here would produce a check people learn to re-run rather than
#: read. A drift large enough to matter is orders of magnitude bigger than this.
TOLERANCE = 0.002
PERCENT_TOLERANCE = 0.05


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


#: The five figures in a row, in the order the table prints them.
FIGURES = ("classical", "optimum", "headroom", "low", "high")


def published(name: str) -> dict[str, tuple[float, ...]]:
    """The headroom table as written in one README."""
    table: dict[str, tuple[float, ...]] = {}
    for task, classical, optimum, headroom, low, high in ROW.findall(read(name)):
        table[task] = tuple(
            float(value.replace("−", "-")) for value in (classical, optimum, headroom, low, high)
        )
    return table


def documented_run() -> tuple[int, int] | None:
    found = COMMAND.search(read("README.md"))
    return (int(found.group(1)), int(found.group(2))) if found else None


def measure(task: str, instances: int, episodes: int) -> dict[str, float] | str:
    """Run one task exactly as the README says to, and read the figures back."""
    from stadion.cli import main as cli

    argv = ["run", task, "--agent", "classical", "--instances", str(instances),
            "--episodes", str(episodes)]
    import contextlib
    import io

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            cli(argv)
    except SystemExit as stop:
        if stop.code not in (0, None):
            return f"`stadion run {task}` exited {stop.code}"
    except Exception as problem:  # noqa: BLE001 - reported, not swallowed
        return f"`stadion run {task}` raised {type(problem).__name__}: {problem}"

    text = captured.getvalue()
    figures = {}
    for label, pattern in (
        ("classical", r"^\s*classical\s+(-?[\d.]+)\s*$"),
        ("optimum", r"^\s*optimum\s+(-?[\d.]+)\s*$"),
    ):
        found = re.search(pattern, text, re.M)
        if not found:
            return f"`stadion run {task}` printed no {label} figure"
        figures[label] = float(found.group(1))

    gap = re.search(
        r"^\s*optimum - classical: \+?(-?[\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\] \(\+?(-?[\d.]+)%\)",
        text, re.M,
    )
    if not gap:
        return f"`stadion run {task}` printed no optimum-minus-classical line"
    figures["low"] = float(gap.group(2))
    figures["high"] = float(gap.group(3))
    figures["headroom"] = float(gap.group(4))
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="recompute every row, not just inventory (minutes)")
    args = parser.parse_args()

    problems: list[str] = []

    from stadion.tasks import all_tasks

    tasks = {task.name: task for task in all_tasks()}

    # ---- what the prose counts ---------------------------------------------
    for name in READMES:
        text = read(name)
        word = "six" if name == "README.md" else "шесть"
        if not re.search(rf"(?i){word} (?:tasks|задач)", text):
            problems.append(f"{name}: no longer says there are {word} tasks")
    if len(tasks) != 6:
        problems.append(f"{len(tasks)} tasks are registered, both READMEs say six")

    for task, size in MENUS.items():
        if task not in tasks:
            problems.append(f"{task} is named in the prose but is not a registered task")
            continue
        instance = tasks[task].instance(0)
        menu = _menu_size(tasks[task], instance)
        if menu is None:
            problems.append(f"{task}: could not read its action menu from the tool schema")
        elif menu != size:
            problems.append(
                f"{task}: the prose says {size} actions, the tool schema offers {menu}"
            )

    # ---- the two tables ------------------------------------------------------
    tables = {name: published(name) for name in READMES}
    for name, table in tables.items():
        if set(table) != set(tasks):
            missing = sorted(set(tasks) - set(table))
            extra = sorted(set(table) - set(tasks))
            problems.append(
                f"{name}: the headroom table covers {sorted(table)}"
                + (f", missing {missing}" if missing else "")
                + (f", and lists {extra} which are not tasks" if extra else "")
            )
    if tables["README.md"] != tables["README.ru.md"]:
        for task in sorted(set(tables["README.md"]) | set(tables["README.ru.md"])):
            if tables["README.md"].get(task) != tables["README.ru.md"].get(task):
                problems.append(
                    f"{task}: the two READMEs disagree — "
                    f"en {tables['README.md'].get(task)}, ru {tables['README.ru.md'].get(task)}"
                )

    # ---- the numbers themselves ---------------------------------------------
    run = documented_run()
    if run is None:
        problems.append(
            "README.md: the line telling a reader how to reproduce the table is gone, "
            "so there is nothing to reproduce it with"
        )
    else:
        instances, episodes = run
        wanted = sorted(tables["README.md"]) if args.all else ["inventory"]
        for task in wanted:
            measured = measure(task, instances, episodes)
            if isinstance(measured, str):
                problems.append(measured)
                continue
            claimed = tables["README.md"][task]
            for index, key in enumerate(FIGURES):
                limit = PERCENT_TOLERANCE if key == "headroom" else TOLERANCE
                if abs(measured[key] - claimed[index]) > limit:
                    problems.append(
                        f"{task}: the table says {key} {claimed[index]}, "
                        f"the run gives {measured[key]}"
                    )

    for problem in problems:
        print(f"  {problem}")
    if problems:
        print(f"\n  discrepancies: {len(problems)}", file=sys.stderr)
        return 1

    scope = "every row" if args.all else "inventory"
    print(f"  {len(tasks)} tasks, both tables agree, and {scope} reproduces what is published.")
    return 0


def _menu_size(task: Any, instance: Any) -> int | None:
    """The length of the numbered menu the agent is offered, from the tool schema.

    None when the schema holds anything other than exactly one enum: a task with
    two menus is not one this check knows how to describe, and guessing which of
    them the prose meant is how a check starts being wrong quietly.
    """
    def walk(node: Any) -> Iterator[int]:
        if isinstance(node, dict):
            if "enum" in node:
                yield len(node["enum"])
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    sizes = list(walk(task.tool_schema(instance)))
    return sizes[0] if len(sizes) == 1 else None


if __name__ == "__main__":
    raise SystemExit(main())

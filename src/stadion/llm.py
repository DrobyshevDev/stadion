"""Playing a task with a language model.

Deliberately provider-agnostic: you hand :class:`LLMAgent` a function that maps
a prompt to a reply, and nothing here knows or cares what produced it. That
keeps the benchmark free of a client library that would date faster than the
tasks do, and it means a local model, a hosted one and a canned transcript all
enter through the same door.

The reply is parsed for the number of a legal choice. Refusing to guess when
none is present is intentional — an agent whose output cannot be read is a
failed run, and quietly substituting a default would put a number on the board
that no policy actually earned.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from stadion.core.agent import Agent
from stadion.core.task import Brief, View

__all__ = ["LLMAgent", "parse_choice"]

_INTEGER = re.compile(r"-?\d+")


def parse_choice(reply: str, view: View) -> int:
    """Pull a legal choice out of a model's reply.

    Takes the last legal integer in the text, which is what survives a model
    that reasons out loud before committing ("...so 3 is too many, I pick 2").
    """
    legal = {c.value for c in view.choices}
    found = [int(m) for m in _INTEGER.findall(reply)]
    for value in reversed(found):
        if value in legal:
            return value
    allowed = ", ".join(str(c.value) for c in sorted(view.choices, key=lambda c: c.value))
    raise ValueError(
        f"no legal choice in the model's reply at step {view.step}.\n"
        f"  legal values: {allowed}\n"
        f"  reply was: {reply.strip()[:400]!r}\n"
        f"If the model is reasoning at length, ask it to end with the number alone."
    )


class LLMAgent(Agent):
    """An agent backed by any ``prompt -> reply`` callable."""

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        name: str = "llm",
        history: int = 5,
    ) -> None:
        self._complete = complete
        self.name = name
        self._history = history
        self._brief: Brief | None = None
        self._log: list[str] = []

    def start(self, brief: Brief) -> None:
        self._brief = brief
        self._log = []

    def act(self, view: View) -> int:
        reply = self._complete(self.prompt(view))
        chosen = parse_choice(reply, view)
        self._log.append(f"  {view.text} -> {view.choice(chosen).label}")
        return chosen

    def prompt(self, view: View) -> str:
        """The full text sent to the model. Exposed so a run can be audited."""
        if self._brief is None:  # pragma: no cover - start() precedes act() in the runner
            raise RuntimeError("LLMAgent.start() must run before act(); no brief bound")
        menu = "\n".join(
            f"  {c.value} = {c.label}" + (f"  ({c.detail})" if c.detail else "")
            for c in view.choices
        )
        recent = ""
        if self._log:
            recent = "What you did recently:\n" + "\n".join(self._log[-self._history :]) + "\n\n"
        return (
            f"{self._brief.text}\n\n"
            f"{recent}"
            f"{view.text}\n\n"
            f"Choose one:\n{menu}\n\n"
            f"Answer with the number of your choice, and put it last."
        )

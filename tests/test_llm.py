"""Reading a choice out of a model's reply, including when there isn't one."""

from __future__ import annotations

import pytest

import stadion
from stadion.llm import LLMAgent, parse_choice


def _view(task_name: str = "queueing") -> stadion.View:
    task = stadion.get(task_name)
    inst = task.instance(0)
    env = inst.env()
    obs, _ = env.reset(seed=0)
    return task.view(env, obs, 0, 0.0)


def test_a_bare_number_is_read() -> None:
    assert parse_choice("1", _view()) == 1


def test_the_choice_is_taken_from_the_end_of_a_model_that_thinks_out_loud() -> None:
    """Reasoning mentions numbers it rejects; the commitment comes last."""
    reply = "The buffer is at 4 of 10 and the job is worth 0.21, so 1 would be greedy. 0"
    assert parse_choice(reply, _view()) == 0


def test_numbers_that_are_not_legal_choices_are_skipped() -> None:
    reply = "Step 37 of 100, value 0.812, buffer 9 — admit. 1"
    assert parse_choice(reply, _view()) == 1


def test_a_reply_with_no_legal_choice_raises_instead_of_defaulting() -> None:
    """Substituting a default would put a number on the board no policy earned."""
    with pytest.raises(ValueError, match="no legal choice"):
        parse_choice("I would rather not answer.", _view())


def test_the_error_quotes_the_reply_and_lists_what_was_allowed() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_choice("maybe later", _view())
    message = str(excinfo.value)
    assert "maybe later" in message
    assert "legal values: 0, 1" in message


def test_an_llm_agent_plays_a_whole_episode_through_its_completion_function() -> None:
    prompts: list[str] = []

    def always_reject(prompt: str) -> str:
        prompts.append(prompt)
        return "0"

    task = stadion.get("queueing")
    inst = task.instance(0)
    earned = stadion.play_episode(task, LLMAgent(always_reject, name="rejector"), inst, seed=0)

    assert len(prompts) == int(inst.params["horizon"])
    assert earned == pytest.approx(0.0)  # admitted nothing, so paid no congestion
    assert "Choose one:" in prompts[0]
    assert "What you did recently:" in prompts[-1]


def test_the_prompt_carries_the_brief_so_a_model_knows_the_cost_structure() -> None:
    task = stadion.get("inventory")
    inst = task.instance(0)
    agent = LLMAgent(lambda _: "0")
    agent.start(task.brief(inst))
    env = inst.env()
    obs, _ = env.reset(seed=0)
    prompt = agent.prompt(task.view(env, obs, 0, 0.0))
    assert f"{inst.params['demand_mean']:.2f}" in prompt
    assert "order 0" in prompt

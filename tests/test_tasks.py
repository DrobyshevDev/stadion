"""What a task family guarantees: reproducible instances, legal moves, honest briefs."""

from __future__ import annotations

import pytest

import stadion
from stadion.core.task import _salt

TASKS = stadion.names()


@pytest.mark.parametrize("task_name", TASKS)
def test_the_same_seed_draws_the_same_instance(task_name: str) -> None:
    task = stadion.get(task_name)
    assert dict(task.instance(11).params) == dict(task.instance(11).params)
    assert dict(task.instance(11).params) != dict(task.instance(12).params)


def test_the_instance_salt_is_stable_across_processes() -> None:
    """``hash()`` is randomised per process; instance parameters must not be.

    These constants are pinned rather than recomputed: if the salt changes, every
    published number in every README silently refers to different problems.
    """
    assert _salt("inventory") == 0x9067F6583BEC1104
    assert _salt("pricing") == 0xB89E47670523DB3E
    assert _salt("queueing") == 0x7415BA59A75C26E9
    assert _salt("energy") == 0x014E017EB39655BA
    assert _salt("supply-chain") == 0x27731B5B031D515F
    assert _salt("joint-pricing") == 0xC9B14D22C4A41B98


@pytest.mark.parametrize("task_name", TASKS)
def test_every_offered_choice_is_accepted_by_the_environment(task_name: str) -> None:
    """The menu an agent reads and the action space it plays into are one thing."""
    task = stadion.get(task_name)
    inst = task.instance(0)
    env = inst.env()
    obs, _ = env.reset(seed=0)
    view = task.view(env, obs, 0, 0.0)
    assert view.choices, "a task with no legal moves cannot be played"
    for choice in view.choices:
        probe = inst.env()
        probe.reset(seed=0)
        probe.step(choice.act)  # must not raise, must not be silently clipped
        assert env.action_space.contains(choice.act)


@pytest.mark.parametrize("task_name", TASKS)
def test_choice_values_are_distinct_and_dense(task_name: str) -> None:
    """An agent answering with a number must land on exactly one option."""
    task = stadion.get(task_name)
    inst = task.instance(0)
    env = inst.env()
    obs, _ = env.reset(seed=0)
    values = [c.value for c in task.view(env, obs, 0, 0.0).choices]
    assert values == sorted(set(values))
    assert values == list(range(len(values)))


@pytest.mark.parametrize("task_name", TASKS)
def test_an_illegal_choice_is_named_rather_than_clipped(task_name: str) -> None:
    task = stadion.get(task_name)
    inst = task.instance(0)
    env = inst.env()
    obs, _ = env.reset(seed=0)
    view = task.view(env, obs, 0, 0.0)
    with pytest.raises(ValueError, match="not a legal choice"):
        view.choice(9_999)


@pytest.mark.parametrize("task_name", TASKS)
def test_the_brief_states_the_horizon_the_episode_actually_runs(task_name: str) -> None:
    task = stadion.get(task_name)
    inst = task.instance(4)
    brief = task.brief(inst)
    assert str(brief.horizon) in brief.text

    env = inst.env()
    obs, _ = env.reset(seed=0)
    first = task.view(env, obs, 0, 0.0).choices[0].act
    steps, done = 0, False
    while not done:
        _, _, terminated, truncated, _ = env.step(first)
        steps += 1
        done = terminated or truncated
    # Taking the first option every step can end a task early — selling out ends
    # pricing — but never late.
    assert steps <= brief.horizon


@pytest.mark.parametrize("task_name", TASKS)
def test_baseline_tuning_seeds_are_far_from_evaluation_seeds(task_name: str) -> None:
    """A classical rule fitted on the episodes it is scored on is not a baseline."""
    task = stadion.get(task_name)
    evaluation_seeds = range(0, 10_000)
    tuning_seeds = range(task.tuning_seed, task.tuning_seed + task.tuning_episodes)
    assert not set(evaluation_seeds) & set(tuning_seeds)


@pytest.mark.parametrize("task_name", TASKS)
def test_the_tool_schema_offers_exactly_the_legal_choices(task_name: str) -> None:
    task = stadion.get(task_name)
    inst = task.instance(0)
    schema = task.tool_schema(inst)
    env = inst.env()
    obs, _ = env.reset(seed=0)
    offered = [c.value for c in task.view(env, obs, 0, 0.0).choices]
    assert schema["input_schema"]["properties"]["choice"]["enum"] == offered


def test_an_unknown_task_name_lists_the_known_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        stadion.get("warehouse")
    message = str(excinfo.value)
    assert "warehouse" in message
    for known in TASKS:
        assert known in message

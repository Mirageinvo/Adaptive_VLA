"""Dependency-free contract tests for the receding-horizon executor."""

from __future__ import annotations

import numpy as np
import pytest

from receding_horizon import (
    CALVIN_DT_SECONDS,
    assert_calvin_control_timestep,
    execute_receding_horizon,
)


class CountingPolicy:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict_action_chunk(self, observation: int) -> np.ndarray:
        self.calls.append(observation)
        chunk = np.zeros((30, 7), dtype=np.float32)
        chunk[:, 0] = 0.25 * len(self.calls)
        chunk[:, 6] = 1.0
        return chunk


class CountingEnv:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self, action: np.ndarray) -> tuple[int, float, bool, dict[str, int]]:
        self.step_count += 1
        return self.step_count, 0.0, False, {"step": self.step_count}


def test_replans_after_twelve_steps_and_discards_tail() -> None:
    policy = CountingPolicy()
    env = CountingEnv()
    trace = execute_receding_horizon(
        policy,
        env,
        initial_observation=0,
        env_dt_seconds=CALVIN_DT_SECONDS,
        max_env_steps=25,
    )
    assert policy.calls == [0, 12, 24]
    assert trace.plan_count == 3
    assert trace.actions.shape == (25, 7)
    assert np.all(trace.actions[:12, 0] == 0.25)
    assert np.all(trace.actions[12:24, 0] == 0.5)
    assert trace.actions[24, 0] == 0.75


def test_rejects_non_calvin_timestep() -> None:
    with pytest.raises(ValueError, match="1/30"):
        assert_calvin_control_timestep(1.0 / 20.0)


def test_stops_without_replanning_after_termination() -> None:
    policy = CountingPolicy()

    class TerminatingEnv(CountingEnv):
        def step(self, action: np.ndarray) -> tuple[int, float, bool, dict[str, int]]:
            result = super().step(action)
            return result[0], result[1], self.step_count == 5, result[3]

    trace = execute_receding_horizon(
        policy,
        TerminatingEnv(),
        initial_observation=0,
        env_dt_seconds=CALVIN_DT_SECONDS,
        max_env_steps=30,
    )
    assert trace.terminated
    assert len(trace.actions) == 5
    assert trace.plan_count == 1

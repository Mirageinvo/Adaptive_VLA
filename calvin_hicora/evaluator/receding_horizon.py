"""Fail-closed receding-horizon execution core for CALVIN policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


CALVIN_FREQUENCY_HZ = 30.0
CALVIN_DT_SECONDS = 1.0 / CALVIN_FREQUENCY_HZ
CALVIN_CHUNK_LENGTH = 30
CALVIN_EXECUTION_HORIZON = 12
CALVIN_ACTION_DIM = 7


class ChunkPolicy(Protocol):
    def predict_action_chunk(self, observation: Any) -> np.ndarray:
        """Return one native-scale CALVIN action chunk with shape (30, 7)."""


class StepEnvironment(Protocol):
    def step(self, action: np.ndarray) -> tuple[Any, ...]:
        """Advance the simulator by exactly one control step."""


@dataclass(frozen=True)
class ExecutionTrace:
    observations: list[Any]
    actions: np.ndarray
    plan_count: int
    terminated: bool
    truncated: bool
    infos: list[dict[str, Any]]


def assert_calvin_control_timestep(
    env_dt_seconds: float, *, absolute_tolerance: float = 1e-9
) -> None:
    """Assert that one env.step corresponds to one 30 Hz control tick."""
    if not np.isfinite(env_dt_seconds) or env_dt_seconds <= 0:
        raise ValueError(f"env_dt_seconds must be positive and finite, got {env_dt_seconds}")
    if not np.isclose(
        env_dt_seconds, CALVIN_DT_SECONDS, rtol=0.0, atol=absolute_tolerance
    ):
        raise ValueError(
            f"CALVIN control dt must be 1/30 s ({CALVIN_DT_SECONDS:.17g}), "
            f"got {env_dt_seconds:.17g}"
        )


def validate_action_chunk(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    expected = (CALVIN_CHUNK_LENGTH, CALVIN_ACTION_DIM)
    if chunk.shape != expected:
        raise ValueError(f"Policy action chunk must have shape {expected}, got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("Policy action chunk contains NaN or inf")
    if np.any(chunk < -1.0) or np.any(chunk > 1.0):
        raise ValueError("Policy action chunk must use native CALVIN range [-1, 1]")
    gripper = chunk[:, 6]
    if not np.isin(gripper, (-1.0, 1.0)).all():
        raise ValueError("Executed CALVIN gripper actions must be discretized to -1 or +1")
    return chunk


def _unpack_step(result: tuple[Any, ...]) -> tuple[Any, bool, bool, dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError(f"env.step must return a tuple, got {type(result)}")
    if len(result) == 4:
        observation, _reward, done, info = result
        return observation, bool(done), False, dict(info)
    if len(result) == 5:
        observation, _reward, terminated, truncated, info = result
        return observation, bool(terminated), bool(truncated), dict(info)
    raise ValueError(f"env.step must return a Gym 4- or 5-tuple, got length {len(result)}")


def execute_receding_horizon(
    policy: ChunkPolicy,
    env: StepEnvironment,
    initial_observation: Any,
    *,
    env_dt_seconds: float,
    max_env_steps: int,
) -> ExecutionTrace:
    """Plan 30 actions, execute 12, discard 18, then replan.

    There is deliberately no overlap averaging. The next policy call receives
    the observation produced by the twelfth executed action.
    """
    assert_calvin_control_timestep(env_dt_seconds)
    if max_env_steps <= 0:
        raise ValueError("max_env_steps must be positive")

    observation = initial_observation
    observations = [observation]
    actions: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    plan_count = 0
    terminated = False
    truncated = False

    while len(actions) < max_env_steps and not (terminated or truncated):
        chunk = validate_action_chunk(policy.predict_action_chunk(observation))
        plan_count += 1
        remaining = max_env_steps - len(actions)
        execute_count = min(CALVIN_EXECUTION_HORIZON, remaining)
        for action in chunk[:execute_count]:
            observation, terminated, truncated, info = _unpack_step(env.step(action))
            actions.append(action.copy())
            observations.append(observation)
            infos.append(info)
            if terminated or truncated:
                break

    action_array = np.asarray(actions, dtype=np.float32).reshape(-1, CALVIN_ACTION_DIM)
    return ExecutionTrace(
        observations=observations,
        actions=action_array,
        plan_count=plan_count,
        terminated=terminated,
        truncated=truncated,
        infos=infos,
    )

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Tuple

from .environment import OptimizationEnvironment


class OptimizationGymEnv(gym.Env):
    """
    Gymnasium wrapper for OptimizationEnvironment to enable Stable-Baselines3.

    This keeps the core environment untouched and only adapts the API to the
    Gym/Gymnasium interface expected by SB3.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        problem: Any,
        algorithms: list,
        max_evaluations: int = 10000,
        schedule_interval: int = 500,
        reward_type: str = "improvement_with_efficiency",
    ):
        super().__init__()

        # Use existing environment (no changes to core logic)
        self.env = OptimizationEnvironment(
            problem=problem,
            algorithms=algorithms,
            max_evaluations=max_evaluations,
            schedule_interval=schedule_interval,
            reward_type=reward_type,
        )

        # Observation/action spaces
        obs_dim = self.env.get_observation_space_size()
        action_dim = self.env.get_action_space_size()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)

        self._last_info: Dict[str, Any] = {}

    def reset(
        self, *, seed: int | None = None, options: Dict | None = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        state = self.env.reset().astype(np.float32)
        info = self.env.get_episode_info()
        self._last_info = info
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        next_state, reward, done, info = self.env.step(int(action))

        # Merge info with episode stats
        episode_info = self.env.get_episode_info()
        merged_info = {**info, **episode_info}
        self._last_info = merged_info

        terminated = done
        truncated = False  # no time-limit truncation here

        return (
            next_state.astype(np.float32),
            float(reward),
            terminated,
            truncated,
            merged_info,
        )

    def render(self, mode: str = "human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

    # Convenience helpers
    def get_episode_info(self) -> Dict[str, Any]:
        return self.env.get_episode_info()

    def get_algorithm_selection_counts(self) -> Dict[str, int]:
        return self.env.get_algorithm_selection_counts()


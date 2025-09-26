import gymnasium as gym
import gymnasium_robotics
import numpy as np
from typing import Tuple, Dict, Any
from gymnasium.wrappers import TimeLimit


class AdroitHandDoorEnvironment(gym.Env):
    """
    Wrapper for AdroitHandDoor-v1 with preprocessing for SAC training.
    """

    def __init__(self,
                 render_mode=None,
                 seed=None,
                 max_episode_steps=200,
                 reward_type='dense'):
        # Choose environment
        env_name = "AdroitHandDoor-v1" if reward_type == "dense" else "AdroitHandDoorSparse-v1"
        base_env = gym.make(env_name, render_mode=render_mode)

        # Wrap with TimeLimit to enforce max steps
        self.env = TimeLimit(base_env, max_episode_steps=max_episode_steps)

        # Set environment spaces for gym.Env compatibility
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type

        # Info
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        print(f"AdroitHandDoor Environment Initialized:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Action dimension: {self.action_dim}")
        print(f"  Reward type: {reward_type}")
        print(f"  Max episode steps: {max_episode_steps}")

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return self._preprocess_observation(obs), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._preprocess_observation(obs)
        reward = self._shape_reward(reward, obs, info)
        return obs, reward, terminated, truncated, info

    def _preprocess_observation(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def _shape_reward(self, reward, obs, info):
        if self.reward_type == "sparse":
            return reward
        return reward

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_action_bounds(self):
        return self.action_space.low, self.action_space.high



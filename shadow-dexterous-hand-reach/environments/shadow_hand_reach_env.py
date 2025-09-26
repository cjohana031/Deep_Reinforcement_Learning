import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class SuccessInfoWrapper(gym.Wrapper):
    """
    Wrapper that tracks success info and makes it available to episode recording.
    This ensures the 'is_success' flag gets properly tracked by SB3.
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_episode_success = False

    def reset(self, **kwargs):
        self.current_episode_success = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Track success for this step
        if info.get('is_success', False):
            self.current_episode_success = True

        # Add episode success to info when episode ends
        if terminated or truncated:
            info['episode'] = {'is_success': self.current_episode_success}

        return obs, reward, terminated, truncated, info


class ShadowHandReachEnvironment(gym.Env):
    """
    Wrapper for Shadow Dexterous Hand Reach environment with goal-conditioned setup
    """

    def __init__(self,
                 render_mode: Optional[str] = None,
                 seed: int = 42,
                 max_episode_steps: int = 50,
                 reward_type: str = "dense",
                 reward_scale: float = 1.0,
                 use_log_reward: bool = False,
                 log_epsilon: float = 1e-6):
        """
        Initialize Shadow Hand Reach environment

        Args:
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            seed: Random seed
            max_episode_steps: Maximum steps per episode
            reward_type: 'dense' or 'sparse' reward function
            reward_scale: Scaling factor for rewards (default 1.0 = no scaling)
            use_log_reward: Apply log transform to amplify near-zero learning signal
            log_epsilon: Small constant to avoid log(0) in reward transform
        """
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.use_log_reward = use_log_reward
        self.log_epsilon = log_epsilon
        self.current_step = 0

        # Create the environment with dense rewards
        env_id = 'HandReachDense-v3' if reward_type == 'dense' else 'HandReach-v3'

        self.env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps
        )

        # Set seed
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        # Cache environment properties
        self._action_dim = self.env.action_space.shape[0]  # 20 for Shadow Hand
        self._action_bounds = (self.env.action_space.low, self.env.action_space.high)

        # For goal-conditioned environment, we need to handle the dict observation space
        self._setup_observation_space()

        # Set up Gymnasium environment attributes
        super().__init__()

    def _setup_observation_space(self):
        """Setup observation space handling for goal-conditioned environment"""
        # The observation space is a Dict with 'observation', 'achieved_goal', 'desired_goal'
        obs_space = self.env.observation_space

        if isinstance(obs_space, gym.spaces.Dict):
            # Extract dimensions
            self.obs_dim = obs_space['observation'].shape[0]  # 63
            self.goal_dim = obs_space['achieved_goal'].shape[0]  # 15 (5 fingers * 3 coords)

            # Combined state dimension (observation + achieved_goal + desired_goal)
            self._state_dim = self.obs_dim + 2 * self.goal_dim  # 63 + 15 + 15 = 93

            # Create flattened observation space for Stable Baselines3
            self._observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._state_dim,),
                dtype=np.float32
            )
        else:
            # Fallback for non-dict observation spaces
            self._state_dim = obs_space.shape[0]
            self.obs_dim = self._state_dim
            self.goal_dim = 0
            self._observation_space = obs_space

        # Store the original action space
        self._action_space = self.env.action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial state"""
        self.current_step = 0

        obs, info = self.env.reset(seed=seed)

        # Convert dict observation to flat array
        state = self._obs_to_state(obs)

        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return next state, reward, done flags, and info"""
        self.current_step += 1

        # Execute action in environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store original reward for success calculation fallback
        original_reward = reward

        # Apply log transform if enabled (before scaling)
        if self.use_log_reward:
            reward = self._apply_log_transform(reward)

        # Apply reward scaling
        if self.reward_scale != 1.0:
            reward = reward * self.reward_scale

        # Convert dict observation to flat array
        state = self._obs_to_state(obs)

        # Add distance info and ensure success tracking is correct
        if isinstance(obs, dict) and 'achieved_goal' in obs and 'desired_goal' in obs:
            distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
            info['distance'] = distance

            # Always override is_success with our custom threshold
            info['is_success'] = distance < 0.02  # 2cm tolerance
        elif 'is_success' not in info:
            # Fallback for non-dict observations: use original reward for success check
            info['is_success'] = original_reward > -0.01 if self.reward_type == 'dense' else original_reward >= 0

        return state, reward, terminated, truncated, info

    def _apply_log_transform(self, reward: float) -> float:
        """
        Apply log transform to amplify near-zero learning signal.
        For negative rewards (distances): r' = -log(-r + epsilon)

        This amplifies small improvements near zero while keeping reward ordering.
        Example: -0.02 → 3.9, -0.01 → 4.6, -0.005 → 5.3
        """
        if reward >= 0:
            # For positive or zero rewards, return as-is or apply different transform
            return reward
        else:
            # For negative rewards (typical distance-based rewards)
            # Ensure we don't take log of zero or negative values
            negative_reward = -abs(reward)  # Ensure it's negative
            return -np.log(-negative_reward + self.log_epsilon)

    def _obs_to_state(self, obs) -> np.ndarray:
        """Convert observation dict to flat state array"""
        if isinstance(obs, dict):
            # Concatenate observation, achieved_goal, and desired_goal
            state_parts = [
                obs['observation'],
                obs['achieved_goal'],
                obs['desired_goal']
            ]
            return np.concatenate(state_parts)
        else:
            # If obs is already a flat array, return as is
            return obs

    def get_state_dim(self) -> int:
        """Get state dimension"""
        return self._state_dim

    def get_action_dim(self) -> int:
        """Get action dimension"""
        return self._action_dim

    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get action bounds (low, high)"""
        return self._action_bounds

    def get_success_rate(self) -> float:
        """Get success rate for current episode"""
        # This will be tracked by the training callback
        return 0.0

    def close(self):
        """Close the environment"""
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self):
        """Render the environment"""
        return self.env.render()

    @property
    def action_space(self):
        """Get action space"""
        return self._action_space

    @property
    def observation_space(self):
        """Get observation space"""
        return self._observation_space
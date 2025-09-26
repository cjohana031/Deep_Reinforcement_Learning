import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional


class HandReachEnv(gym.Env):
    """
    Wrapper for the Shadow Dexterous Hand Reach Dense environment.
    This environment is based on the FetchReachDense-v4 environment from gymnasium-robotics.

    The goal is for the fingertips of a Shadow Dexterous Hand to reach predefined target positions.
    The hand has 20 degrees of freedom and the task uses dense rewards.
    """

    def __init__(
        self,
        env_id: str = "FetchReachDense-v4",
        max_episode_steps: int = 50,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Hand Reach environment wrapper.

        Args:
            env_id: The gymnasium environment ID
            max_episode_steps: Maximum number of steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Create the environment
        try:
            self.env = gym.make(
                env_id,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode
            )
        except Exception as e:
            print(f"Error creating environment {env_id}: {e}")
            print("Make sure gymnasium-robotics is installed: pip install gymnasium-robotics")
            raise

        # Get spaces from the underlying environment
        self.action_space = self.env.action_space

        # The observation space is a Dict with 'observation', 'achieved_goal', 'desired_goal'
        # We'll flatten it for easier use with standard RL algorithms
        if isinstance(self.env.observation_space, spaces.Dict):
            obs_dim = (
                self.env.observation_space['observation'].shape[0] +
                self.env.observation_space['achieved_goal'].shape[0] +
                self.env.observation_space['desired_goal'].shape[0]
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
        else:
            self.observation_space = self.env.observation_space

        # Environment properties
        self.action_dim = self.action_space.shape[0]  # 20 DOF
        self.state_dim = self.observation_space.shape[0]  # Flattened observation

        print(f"Environment initialized: {env_id}")
        print(f"Action space: {self.action_space}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")

    def _flatten_observation(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten the dictionary observation into a single numpy array.

        Args:
            obs_dict: Dictionary observation with 'observation', 'achieved_goal', 'desired_goal'

        Returns:
            Flattened observation array
        """
        if isinstance(obs_dict, dict):
            return np.concatenate([
                obs_dict['observation'].flatten(),
                obs_dict['achieved_goal'].flatten(),
                obs_dict['desired_goal'].flatten()
            ])
        return obs_dict

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        obs_dict, info = self.env.reset(seed=seed)
        obs = self._flatten_observation(obs_dict)
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (20-dimensional for Shadow Hand)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        obs = self._flatten_observation(obs_dict)

        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def seed(self, seed: Optional[int] = None):
        """Set the random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        elif hasattr(self.env, 'reset'):
            return self.env.reset(seed=seed)
        else:
            np.random.seed(seed)
            return [seed]

    def get_success_rate(self, obs_dict: Dict[str, np.ndarray]) -> bool:
        """
        Check if the task is successfully completed.

        Args:
            obs_dict: Dictionary observation

        Returns:
            True if task is completed successfully
        """
        if isinstance(obs_dict, dict):
            achieved_goal = obs_dict['achieved_goal']
            desired_goal = obs_dict['desired_goal']

            # Success is when the 2-norm distance between achieved and desired goal is < 0.05
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return distance < 0.05

        return False


def make_hand_reach_env(render_mode: Optional[str] = None, max_episode_steps: int = 50) -> HandReachEnv:
    """
    Factory function to create a Hand Reach environment.

    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_episode_steps: Maximum number of steps per episode

    Returns:
        HandReachEnv instance
    """
    return HandReachEnv(
        env_id="FetchReachDense-v4",
        max_episode_steps=max_episode_steps,
        render_mode=render_mode
    )


if __name__ == "__main__":
    # Test the environment
    env = make_hand_reach_env(render_mode=None)

    print("Testing environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Take a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward = {reward:.4f}, Done = {terminated or truncated}")

        if terminated or truncated:
            obs, info = env.reset()
            print("Environment reset")

    env.close()
    print("Environment test completed!")
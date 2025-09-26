import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer


class ReplayBuffer:
    def __init__(self, capacity, device='cpu', observation_space=None, action_space=None):
        self.device = device

        # Create a simple observation/action space if not provided
        if observation_space is None:
            from gymnasium.spaces import Box
            # Default observation space - will be updated on first push
            observation_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        if action_space is None:
            from gymnasium.spaces import Box
            # Default action space - will be updated on first push
            action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.sb3_buffer = SB3ReplayBuffer(
            buffer_size=capacity,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=1
        )

        self._first_push = True

    def push(self, state, action, reward, next_state, done):
        # Convert to numpy arrays if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()

        # Ensure proper shapes
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        reward = np.array([reward], dtype=np.float32)
        done = np.array([done], dtype=np.float32)

        # Update spaces on first push if they were default
        if self._first_push:
            from gymnasium.spaces import Box
            # Only update if we started with default spaces
            if hasattr(self.sb3_buffer, 'observation_space') and self.sb3_buffer.observation_space.shape == (1,):
                self.sb3_buffer.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=state.shape, dtype=np.float32
                )
                # Need to update the observation dimension
                self.sb3_buffer.obs_shape = state.shape

            if hasattr(self.sb3_buffer, 'action_space') and self.sb3_buffer.action_space.shape == (1,):
                self.sb3_buffer.action_space = Box(
                    low=-1, high=1, shape=action.shape, dtype=np.float32
                )
                # Need to update the action dimension
                self.sb3_buffer.action_dim = action.shape[0]

            self._first_push = False

        # Add to buffer
        self.sb3_buffer.add(
            obs=state,
            next_obs=next_state,
            action=action,
            reward=reward,
            done=done,
            infos=[{}]
        )

    def sample(self, batch_size):
        # Sample from SB3 buffer
        batch = self.sb3_buffer.sample(batch_size)

        # Convert to tensors and move to device
        # Handle the case where batch data might already be on a device
        states = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards.squeeze(), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones.squeeze(), dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.sb3_buffer.size()
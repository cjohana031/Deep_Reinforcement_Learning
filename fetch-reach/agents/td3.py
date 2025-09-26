import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional

from models.td3.networks import ActorNetwork, CriticNetwork
from train.replay_buffer import ReplayBuffer


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.

    TD3 is an improvement over DDPG that addresses the overestimation bias
    through several key mechanisms:
    1. Twin critic networks (double Q-learning)
    2. Delayed policy updates
    3. Target policy smoothing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        model_dir: str = 'models/td3',
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        hidden_size: int = 256,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        observation_space=None,
        action_space=None
    ):
        """
        Initialize TD3 agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            model_dir: Directory to save models
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            tau: Soft update parameter
            policy_noise: Noise added to target policy during update
            noise_clip: Range to clip target policy noise
            policy_freq: Frequency of delayed policy updates
            hidden_size: Size of hidden layers
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            exploration_noise: Noise for exploration during training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # TD3 hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise

        # Device setup
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = device
        print(f"Using device: {device}")

        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_size, max_action).to(device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_size).to(device)

        # Target networks
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_size, max_action).to(device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_size).to(device)

        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, device, observation_space, action_space)

        # Training step counter for delayed policy updates
        self.total_it = 0

        print(f"TD3 Agent initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Max action: {max_action}")
        print(f"  Device: {device}")

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current state observation
            training: Whether in training mode (adds exploration noise)

        Returns:
            Action to take
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        # Add exploration noise during training
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = action + noise

        # Clip to action bounds
        action = np.clip(action, -self.max_action, self.max_action)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> dict:
        """
        Update TD3 networks.

        Returns:
            Dictionary of loss values for logging
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self.total_it += 1

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Update critic networks
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)

        losses = {'critic_loss': critic_loss}

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Update actor network
            actor_loss = self._update_actor(states)
            losses['actor_loss'] = actor_loss

            # Soft update target networks
            self._soft_update_targets()

        return losses

    def _update_critic(self, states, actions, rewards, next_states, dones) -> float:
        """
        Update critic networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Critic loss value
        """
        with torch.no_grad():
            # Target policy smoothing: add noise to target actions
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            next_actions = self.target_actor(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)

            # Compute target Q-values (take minimum to reduce overestimation bias)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Compute critic losses
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states) -> float:
        """
        Update actor network.

        Args:
            states: Batch of states

        Returns:
            Actor loss value
        """
        # Compute actor loss
        actor_loss = -self.critic.q1(states, self.actor(states)).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _soft_update_targets(self):
        """Soft update of target networks."""
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, episode: Optional[int] = None):
        """
        Save model checkpoint.

        Args:
            episode: Current episode number (optional)
        """
        filename = f"td3_episode_{episode}.pth" if episode else "td3_final.pth"
        filepath = self.model_dir / filename

        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode': episode,
            'total_it': self.total_it,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'max_action': self.max_action,
                'gamma': self.gamma,
                'tau': self.tau,
                'policy_noise': self.policy_noise,
                'noise_clip': self.noise_clip,
                'policy_freq': self.policy_freq,
                'exploration_noise': self.exploration_noise
            }
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

        # Save as best model if no episode specified
        if episode is None:
            best_path = self.model_dir / "td3_best.pth"
            torch.save(checkpoint, best_path)

    def load(self, filepath: Optional[str] = None) -> bool:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            True if loading successful, False otherwise
        """
        if filepath is None:
            filepath = self.model_dir / "td3_best.pth"
            if not filepath.exists():
                filepath = self.model_dir / "td3_final.pth"

        if not Path(filepath).exists():
            print(f"No model found at {filepath}")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            self.total_it = checkpoint.get('total_it', 0)

            print(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return False

    def set_exploration_noise(self, noise: float):
        """Set exploration noise level."""
        self.exploration_noise = noise

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'total_iterations': self.total_it,
            'buffer_size': len(self.replay_buffer),
            'exploration_noise': self.exploration_noise
        }


if __name__ == "__main__":
    # Test TD3 agent
    state_dim = 16  # FetchReachDense-v4 flattened observation dim (10+3+3)
    action_dim = 4   # FetchReachDense-v4 action dim
    max_action = 1.0

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )

    print("Testing TD3 Agent...")

    # Test action selection
    state = np.random.randn(state_dim)
    action = agent.act(state)
    print(f"Action shape: {action.shape}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

    # Test transition storage
    next_state = np.random.randn(state_dim)
    agent.store_transition(state, action, 0.1, next_state, False)
    print(f"Buffer size after 1 transition: {len(agent.replay_buffer)}")

    print("TD3 Agent test completed successfully!")
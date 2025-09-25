import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from typing import Tuple

from models.sac.networks import ActorNetwork, CriticNetwork
from train.replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dir='models/sac',
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        hidden_size=256,
        buffer_size=1000000,
        batch_size=256
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Device setup
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = device

        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(device)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_size).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_size).to(device)

        # Target networks
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_size).to(device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_size).to(device)

        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # Entropy temperature
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, device)

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if training:
            with torch.no_grad():
                action, _ = self.actor.sample(state_tensor)
        else:
            with torch.no_grad():
                action = self.actor.mean_action(state_tensor)

        return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> dict:
        """Update SAC networks"""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Update critic networks
        critic_losses = self._update_critics(states, actions, rewards, next_states, dones)

        # Update actor network
        actor_loss, entropy_loss = self._update_actor(states)

        # Update target networks
        self._soft_update_targets()

        losses = {
            'critic1_loss': critic_losses[0],
            'critic2_loss': critic_losses[1],
            'actor_loss': actor_loss,
        }

        if self.automatic_entropy_tuning:
            losses['entropy_loss'] = entropy_loss
            losses['alpha'] = self.alpha.item()

        return losses

    def _update_critics(self, states, actions, rewards, next_states, dones) -> Tuple[float, float]:
        """Update critic networks"""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Compute target Q values
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q

        # Current Q values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return critic1_loss.item(), critic2_loss.item()

    def _update_actor(self, states) -> Tuple[float, float]:
        """Update actor network and entropy temperature"""
        # Sample actions from current policy
        new_actions, log_probs = self.actor.sample(states)

        # Compute Q values for new actions
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update entropy temperature
        entropy_loss = 0.0
        if self.automatic_entropy_tuning:
            entropy_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            entropy_loss = entropy_loss.item()

        return actor_loss.item(), entropy_loss

    def _soft_update_targets(self):
        """Soft update of target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, episode=None):
        """Save model checkpoint"""
        filename = f"sac_episode_{episode}.pth" if episode else "sac_final.pth"
        filepath = self.model_dir / filename

        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'episode': episode,
            'alpha': self.alpha,
        }

        if self.automatic_entropy_tuning:
            checkpoint.update({
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'target_entropy': self.target_entropy,
            })

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

        if episode is None:
            best_path = self.model_dir / "sac_best.pth"
            torch.save(checkpoint, best_path)

    def load(self, filepath=None):
        """Load model checkpoint"""
        if filepath is None:
            filepath = self.model_dir / "sac_best.pth"
            if not filepath.exists():
                filepath = self.model_dir / "sac_final.pth"

        if not Path(filepath).exists():
            print(f"No model found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.target_entropy = checkpoint['target_entropy']

        print(f"Model loaded from {filepath}")
        return True
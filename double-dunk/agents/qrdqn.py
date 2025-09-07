import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path

from models.qrdqn.networks import QRDQNNetwork
from train.replay_buffer import PrioritizedReplayBuffer


class QRDQNAgent:
    """Quantile Regression DQN Agent"""
    
    def __init__(
            self,
            action_dim,
            model_dir='models/qrdqn',
            input_channels=4,
            num_quantiles=51
    ):
        self.action_dim = action_dim
        self.input_channels = input_channels
        self.num_quantiles = num_quantiles
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.epsilon = 1.0

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.device = device

        # Initialize networks
        self.q_network = QRDQNNetwork(input_channels, action_dim, num_quantiles).to(self.device)
        self.target_network = QRDQNNetwork(input_channels, action_dim, num_quantiles).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def act(self, state, training=True):
        """Select action using epsilon-greedy policy based on mean Q-values"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            # Convert state to tensor and ensure correct shape
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            q_values = self.q_network.get_q_values(state_tensor)
            return q_values.argmax().item()

    def get_quantiles(self, state):
        """Get full quantile distribution for a state"""
        with torch.no_grad():
            # Convert state to tensor and ensure correct shape
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
            quantiles = self.q_network(state_tensor)
            return quantiles.cpu().numpy()

    def save(self, episode=None):
        filename = f"qrdqn_episode_{episode}.pth" if episode else "qrdqn_final.pth"
        filepath = self.model_dir / filename

        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'episode': episode,
            'num_quantiles': self.num_quantiles
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

        if episode is None:
            best_path = self.model_dir / "qrdqn_best.pth"
            torch.save(checkpoint, best_path)

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.model_dir / "qrdqn_best.pth"
            if not filepath.exists():
                filepath = self.model_dir / "qrdqn_final.pth"

        if not Path(filepath).exists():
            print(f"No model found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])

        print(f"Model loaded from {filepath}")
        return True


class QRDQNUpdater:
    """QR-DQN Updater with Quantile Huber Loss"""
    
    def __init__(
            self,
            agent: QRDQNAgent,
            target_update_freq=10000,
            buffer_size=100_000,
            batch_size=32,
            lr=5e-5,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay_steps=1000000,
            kappa=1.0,  # Huber loss threshold
            device='cpu'
    ):
        self.agent = agent
        self.device = device
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_decay_steps
        self.kappa = kappa
        self.num_quantiles = agent.num_quantiles
        
        print(f"Epsilon decay per step: {self.epsilon_decay:.8f}")
        print(f"Number of quantiles: {self.num_quantiles}")

        self.optimizer = optim.Adam(self.agent.q_network.parameters(), lr=lr)
        self.update_counter = 0
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, self.device)

    def quantile_huber_loss(self, current_quantiles, target_quantiles, tau, weights=None):
        """
        Compute the quantile Huber loss with optional importance sampling weights
        
        Args:
            current_quantiles: [batch_size, num_quantiles]
            target_quantiles: [batch_size, num_quantiles]  
            tau: [num_quantiles] quantile fractions
            weights: [batch_size] importance sampling weights for prioritized replay
        """
        # Expand dimensions for broadcasting
        # current_quantiles: [batch_size, num_quantiles, 1]
        # target_quantiles: [batch_size, 1, num_quantiles]
        current_quantiles = current_quantiles.unsqueeze(2)
        target_quantiles = target_quantiles.unsqueeze(1)
        tau = tau.view(1, -1, 1)
        
        # Compute quantile regression loss
        u = target_quantiles - current_quantiles
        huber_loss = torch.where(u.abs() <= self.kappa, 
                                0.5 * u.pow(2), 
                                self.kappa * (u.abs() - 0.5 * self.kappa))
        quantile_loss = torch.abs(tau - (u < 0).float()) * huber_loss
        
        # Sum over quantile dimension and average over quantile pairs
        elementwise_loss = quantile_loss.sum(dim=(1, 2))  # [batch_size]
        
        if weights is not None:
            # Apply importance sampling weights
            weighted_loss = (elementwise_loss * weights).mean()
            return weighted_loss, elementwise_loss
        else:
            return elementwise_loss.mean(), elementwise_loss

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, weights, idxs = self.replay_buffer.sample(self.batch_size)

        # Get current quantiles for selected actions
        current_quantiles = self.agent.q_network(states)
        current_quantiles = current_quantiles.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.num_quantiles)).squeeze(1)

        with torch.no_grad():
            # Double DQN: Use main network to select actions, target network to evaluate
            next_q_values = self.agent.q_network.get_q_values(next_states)
            next_actions = next_q_values.argmax(1)
            
            # Get target quantiles
            next_quantiles = self.agent.target_network(next_states)
            next_quantiles = next_quantiles.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.num_quantiles)).squeeze(1)
            
            # Compute target quantiles
            target_quantiles = rewards.unsqueeze(1) + self.gamma * next_quantiles * (1 - dones.unsqueeze(1))

        # Compute quantile regression loss with importance sampling
        tau = self.agent.q_network.quantile_fractions
        weighted_loss, elementwise_loss = self.quantile_huber_loss(current_quantiles, target_quantiles, tau, weights)

        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Calculate TD errors for priority updates (use mean Q-values for priority)
        with torch.no_grad():
            current_q_values = self.agent.q_network.get_q_values(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = self.agent.q_network.get_q_values(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            td_errors = current_q_values - target_q_values
        
        # Update priorities based on TD errors
        td_errors_abs = td_errors.abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, td_errors_abs)

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())

        # Linear epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            self.agent.epsilon = self.epsilon

        return weighted_loss.item()

    def save(self, episode=None):
        checkpoint = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'episode': episode
        }

        filename = f"qrdqn_updater_{episode}.pth" if episode else "qrdqn_updater_final.pth"
        filepath = self.agent.model_dir / filename
        torch.save(checkpoint, filepath)

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.agent.model_dir / "qrdqn_updater_final.pth"

        if not Path(filepath).exists():
            print(f"No updater checkpoint found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_counter = checkpoint.get('update_counter', 0)

        print(f"Updater loaded from {filepath}")
        return True
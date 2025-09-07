import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
from pathlib import Path

from models.ddqn.networks import DDQNNetwork
from train.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DDQNAgent:
    def __init__(
            self,
            action_dim,
            model_dir='models/ddqn',
            input_channels=4
    ):
        self.action_dim = action_dim
        self.input_channels = input_channels
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

        self.q_network = DDQNNetwork(input_channels, action_dim).to(self.device)
        self.target_network = DDQNNetwork(input_channels, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def save(self, episode=None):
        filename = f"ddqn_episode_{episode}.pth" if episode else "ddqn_final.pth"
        filepath = self.model_dir / filename

        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'episode': episode
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

        if episode is None:
            best_path = self.model_dir / "ddqn_best.pth"
            torch.save(checkpoint, best_path)

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.model_dir / "ddqn_best.pth"
            if not filepath.exists():
                filepath = self.model_dir / "ddqn_final.pth"

        if not Path(filepath).exists():
            print(f"No model found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])

        print(f"Model loaded from {filepath}")
        return True


class DDQNUpdater:
    def __init__(
            self,
            agent: DDQNAgent,
            target_update_freq=10000,
            buffer_size=100_000,
            batch_size=64,
            lr=2.5e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay_steps=500000,
            warmup_steps=50000,
            device='cpu',
            use_prioritized_replay=True
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
        self.warmup_steps = warmup_steps
        self.use_prioritized_replay = use_prioritized_replay
        print(f"Epsilon decay per step: {self.epsilon_decay:.8f}")
        print(f"Warmup steps: {warmup_steps:,}")
        print(f"Using prioritized replay: {use_prioritized_replay}")

        self.optimizer = optim.Adam(self.agent.q_network.parameters(), lr=lr)
        # Learning rate scheduler - reduce LR after initial training
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.5)
        self.loss_fn = nn.SmoothL1Loss()

        self.update_counter = 0

        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, self.device)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, self.device)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        # Don't update during warmup period or if not enough samples
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, weights, idxs = self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights, idxs = None, None

        current_q_values = self.agent.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: Use main network to select actions, target network to evaluate
            next_actions = self.agent.q_network(next_states).argmax(1)
            next_q_values = self.agent.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        if self.use_prioritized_replay and weights is not None:
            # Compute element-wise loss for priority updates (need to use reduction='none')
            elementwise_loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            # Apply importance sampling weights
            loss = (elementwise_loss * weights).mean()
        else:
            loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities for prioritized replay
        if self.use_prioritized_replay and idxs is not None:
            with torch.no_grad():
                td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(idxs, td_errors)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.agent.target_network.load_state_dict(self.agent.q_network.state_dict())

        # Linear epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            self.agent.epsilon = self.epsilon
        
        # Update learning rate scheduler
        if self.update_counter % 1000 == 0:  # Update LR less frequently
            self.scheduler.step()

        return loss.item()

    def save(self, episode=None):
        checkpoint = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'episode': episode
        }

        filename = f"ddqn_updater_{episode}.pth" if episode else "ddqn_updater_final.pth"
        filepath = self.agent.model_dir / filename
        torch.save(checkpoint, filepath)

    def load(self, filepath=None):
        if filepath is None:
            filepath = self.agent.model_dir / "ddqn_updater_final.pth"

        if not Path(filepath).exists():
            print(f"No updater checkpoint found at {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_counter = checkpoint.get('update_counter', 0)

        print(f"Updater loaded from {filepath}")
        return True
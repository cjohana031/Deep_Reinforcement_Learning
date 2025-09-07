import numpy as np
import random
from collections import deque
import torch


class SumTree:
    """
    Binary tree data structure for efficient priority sampling.
    Used in Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.pending_idx + self.capacity - 1
        
        self.data[self.pending_idx] = data
        self.update(idx, p)
        
        self.pending_idx += 1
        if self.pending_idx >= self.capacity:
            self.pending_idx = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using TD error for prioritization.
    Based on Schaul et al. (2016) "Prioritized Experience Replay"
    """
    def __init__(self, capacity, device='cpu', alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon  # Small constant to prevent zero probabilities
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)
    
    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        
        # Calculate priority segments
        priority_segment = self.tree.total() / batch_size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        # Convert to tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack states and next_states to ensure consistent tensor shapes
        state = torch.FloatTensor(np.stack(states)).to(self.device)
        action = torch.LongTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_states)).to(self.device)
        done = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(is_weights).to(self.device)
        
        return state, action, reward, next_state, done, weights, idxs
    
    def update_priorities(self, idxs, errors):
        """
        Update priorities based on TD errors.
        
        Args:
            idxs: List of sample indices
            errors: TD errors for the samples
        """
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.tree.n_entries


class ReplayBuffer:
    """Original uniform replay buffer for compatibility"""
    def __init__(self, capacity, device='cpu'):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Stack states and next_states to ensure consistent tensor shapes
        state = torch.FloatTensor(np.stack(state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(np.stack(next_state)).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
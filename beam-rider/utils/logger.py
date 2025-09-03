import json
import os
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir='logs', tensorboard_dir='runs'):
        self.log_dir = log_dir
        self.history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'timestamps': []
        }
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize TensorBoard writer
        run_name = f"DDQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tensorboard_dir = os.path.join(tensorboard_dir, run_name)
        self.writer = SummaryWriter(self.tensorboard_dir)
    
    def log_episode(self, episode, reward, steps):
        self.history['episodes'].append(episode)
        self.history['rewards'].append(reward)
        self.history['steps'].append(steps)
        self.history['timestamps'].append(datetime.now().isoformat())
        
        # Log to TensorBoard
        self.writer.add_scalar('Episode/Reward', reward, episode)
        self.writer.add_scalar('Episode/Steps', steps, episode)
        
        # Log moving averages
        if len(self.history['rewards']) >= 10:
            avg_10 = np.mean(self.history['rewards'][-10:])
            self.writer.add_scalar('Average/Reward_10ep', avg_10, episode)
        
        if len(self.history['rewards']) >= 100:
            avg_100 = np.mean(self.history['rewards'][-100:])
            self.writer.add_scalar('Average/Reward_100ep', avg_100, episode)
    
    def save_history(self, filename=None):
        if filename is None:
            filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {filepath}")
    
    def get_history(self):
        return self.history
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, **self.history)
        print(f"Training history saved to {filepath}")
    
    def get_statistics(self, last_n=100):
        if len(self.history['rewards']) < last_n:
            last_n = len(self.history['rewards'])
        
        if last_n == 0:
            return {}
        
        recent_rewards = self.history['rewards'][-last_n:]
        recent_steps = self.history['steps'][-last_n:]
        
        stats = {
            'mean_reward': sum(recent_rewards) / len(recent_rewards),
            'max_reward': max(recent_rewards),
            'min_reward': min(recent_rewards),
            'mean_steps': sum(recent_steps) / len(recent_steps),
            'total_episodes': len(self.history['episodes'])
        }
        
        return stats
    
    def log_training_metrics(self, episode, loss=None, epsilon=None, eval_reward=None):
        """Log additional training metrics to TensorBoard"""
        if loss is not None:
            self.writer.add_scalar('Training/Loss', loss, episode)
        if epsilon is not None:
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)
        if eval_reward is not None:
            self.writer.add_scalar('Evaluation/Reward', eval_reward, episode)
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
        print(f"TensorBoard logs saved to {self.tensorboard_dir}")
        return self.tensorboard_dir
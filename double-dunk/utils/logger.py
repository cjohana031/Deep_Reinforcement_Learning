import json
import os
import numpy as np
import csv
from datetime import datetime


class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'timestamps': [],
            'td_losses': [],
            'epsilons': [],
            'network_losses': []
        }
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize CSV logging
        self.csv_path = os.path.join(log_dir, 'training_log.csv')
        self._init_csv()
    
    def log_episode(self, episode, reward, steps, td_loss=None, epsilon=None, network_loss=None):
        timestamp = datetime.now().isoformat()
        self.history['episodes'].append(episode)
        self.history['rewards'].append(reward)
        self.history['steps'].append(steps)
        self.history['timestamps'].append(timestamp)
        
        # Log to CSV
        self._log_to_csv(episode, reward, steps, timestamp, td_loss, epsilon, network_loss)
    
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
    
    def log_training_metrics(self, episode, loss=None, epsilon=None, eval_reward=None, td_loss=None):
        """Log additional training metrics"""
        if loss is not None:
            self.history['network_losses'].append(loss)
        if epsilon is not None:
            self.history['epsilons'].append(epsilon)
        if td_loss is not None:
            self.history['td_losses'].append(td_loss)
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'reward', 'steps', 'timestamp', 'td_loss', 'epsilon', 'network_loss', 'eval_reward'])
    
    def _log_to_csv(self, episode, reward, steps, timestamp, td_loss=None, epsilon=None, network_loss=None, eval_reward=None):
        """Append a row to the CSV file"""
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([episode, reward, steps, timestamp, td_loss, epsilon, network_loss, eval_reward])
    
    def close(self):
        """Close logger and save final data"""
        print(f"CSV training log saved to {self.csv_path}")
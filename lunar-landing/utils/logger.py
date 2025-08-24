import json
import os
import numpy as np
from datetime import datetime


class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'timestamps': []
        }
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def log_episode(self, episode, reward, steps):
        self.history['episodes'].append(episode)
        self.history['rewards'].append(reward)
        self.history['steps'].append(steps)
        self.history['timestamps'].append(datetime.now().isoformat())
    
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
import numpy as np


class RandomAgent:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.name = "RandomAgent"
        
    def select_action(self, observation):
        return np.random.randint(0, self.action_space_size)
    
    def learn(self, observation, action, reward, next_observation, done):
        pass
    
    def save(self, filepath):
        print(f"Random agent doesn't need to save weights")
        
    def load(self, filepath):
        print(f"Random agent doesn't need to load weights")
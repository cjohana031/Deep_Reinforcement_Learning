import gymnasium as gym
import numpy as np


class LunarEnvironment:
    def __init__(self, render_mode=None, seed=None):
        self.env = gym.make('LunarLander-v3', render_mode=render_mode)
        self.seed = seed
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self):
        observation, info = self.env.reset(seed=self.seed)
        return observation, info
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def get_action_space_size(self):
        return self.action_space.n
    
    def get_observation_space_shape(self):
        return self.observation_space.shape
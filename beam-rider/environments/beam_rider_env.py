import gymnasium as gym
import numpy as np
from collections import deque
import cv2
import ale_py

gym.register_envs(ale_py)


class BeamRiderEnvironment:
    def __init__(self, render_mode=None, seed=None, frame_skip=4, frame_stack=4):
        self.env = gym.make('ALE/BeamRider-v5', render_mode=render_mode, frameskip=1)
        self.seed = seed
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        
        # Frame buffer for max pooling
        self.frame_buffer = deque(maxlen=2)
        
        # Stack of frames for temporal information
        self.frame_stack_buffer = deque(maxlen=frame_stack)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(frame_stack, 84, 84), 
            dtype=np.uint8
        )
        self.action_space = self.env.action_space
        
    def reset(self):
        observation, info = self.env.reset(seed=self.seed)
        
        # Process initial frame
        processed_frame = self._preprocess_frame(observation)
        
        # Fill frame stack with initial frame
        for _ in range(self.frame_stack):
            self.frame_stack_buffer.append(processed_frame)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        total_reward = 0
        done = False
        info = {}
        
        # Frame skipping with max pooling
        for i in range(self.frame_skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Store last two frames for max pooling
            self.frame_buffer.append(observation)
            
            if done:
                break
        
        # Max pooling over last two frames
        if len(self.frame_buffer) == 2:
            max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        else:
            max_frame = self.frame_buffer[0]
        
        # Preprocess and add to frame stack
        processed_frame = self._preprocess_frame(max_frame)
        self.frame_stack_buffer.append(processed_frame)
        
        return self._get_stacked_frames(), total_reward, terminated, truncated, info
    
    def _preprocess_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        return resized.astype(np.uint8)
    
    def _get_stacked_frames(self):
        return np.array(list(self.frame_stack_buffer), dtype=np.uint8)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def get_action_space_size(self):
        return self.action_space.n
    
    def get_observation_space_shape(self):
        return self.observation_space.shape

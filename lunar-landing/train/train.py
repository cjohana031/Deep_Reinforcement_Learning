import numpy as np
from environments import LunarEnvironment
from agents import RandomAgent
from utils.logger import Logger


def train(agent, env, episodes=1000, max_steps=1000):
    logger = Logger()
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.learn(observation, action, reward, next_observation, done)
            
            total_reward += reward
            steps += 1
            observation = next_observation
            
            if done:
                break
        
        logger.log_episode(episode, total_reward, steps)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {steps}")
    
    return logger.get_history()


if __name__ == "__main__":
    env = LunarEnvironment(render_mode=None, seed=42)
    agent = RandomAgent(action_space_size=env.get_action_space_size())
    
    print(f"Starting training with {agent.name}")
    print(f"Action space size: {env.get_action_space_size()}")
    print(f"Observation space shape: {env.get_observation_space_shape()}")
    
    history = train(agent, env, episodes=1000)
    
    print("\nTraining completed!")
    print(f"Average reward over last 100 episodes: {np.mean(history['rewards'][-100:]):.2f}")
    
    env.close()
import numpy as np
import torch
from pathlib import Path

from environments.double_dunk_env import DoubleDunkEnvironment
from agents.ddqn import DDQNAgent
from agents.qrdqn import QRDQNAgent


def evaluate_ddqn(episodes=10, render=False, model_path=None):
    print("Evaluating DDQN agent on DoubleDunk...")
    
    # Initialize environment
    render_mode = 'human' if render else None
    env = DoubleDunkEnvironment(render_mode=render_mode, seed=42)
    action_dim = env.get_action_space_size()
    
    # Initialize agent
    agent = DDQNAgent(action_dim=action_dim)
    
    # Load model
    if model_path:
        success = agent.load(model_path)
    else:
        success = agent.load()
    
    if not success:
        print("Failed to load model. Make sure to train the agent first.")
        return
    
    print(f"Using device: {agent.device}")
    
    # Evaluation
    episode_rewards = []
    episode_steps = []
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            if render:
                env.render()
            
            # Select action (no exploration)
            action = agent.act(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        print(f"Episode {episode:2d}: Reward = {episode_reward:8.2f}, Steps = {steps:4d}")
    
    # Statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    print(f"\nEvaluation Results ({episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    env.close()
    return avg_reward


def evaluate_qrdqn(episodes=10, render=False, model_path=None):
    print("Evaluating QR-DQN agent on DoubleDunk...")
    
    # Initialize environment
    render_mode = 'human' if render else None
    env = DoubleDunkEnvironment(render_mode=render_mode, seed=42)
    action_dim = env.get_action_space_size()
    
    # Initialize agent
    agent = QRDQNAgent(action_dim=action_dim)
    
    # Load model
    if model_path:
        success = agent.load(model_path)
    else:
        success = agent.load()
    
    if not success:
        print("Failed to load model. Make sure to train the agent first.")
        return
    
    print(f"Using device: {agent.device}")
    
    # Evaluation
    episode_rewards = []
    episode_steps = []
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            if render:
                env.render()
            
            # Select action (no exploration)
            action = agent.act(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        print(f"Episode {episode:2d}: Reward = {episode_reward:8.2f}, Steps = {steps:4d}")
    
    # Statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    print(f"\nEvaluation Results ({episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    env.close()
    return avg_reward


if __name__ == "__main__":
    evaluate_ddqn(episodes=5, render=True)
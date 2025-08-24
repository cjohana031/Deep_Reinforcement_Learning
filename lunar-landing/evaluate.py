import numpy as np
from environments import LunarEnvironment
from agents import RandomAgent
import time


def evaluate(agent, env, episodes=10, max_steps=1000, render=True):
    rewards = []
    steps_list = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(max_steps):
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.01)
            
            if done:
                break
        
        rewards.append(total_reward)
        steps_list.append(steps)
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        if terminated:
            print("  Status: Successfully landed!" if total_reward > 100 else "  Status: Crashed")
        elif truncated:
            print("  Status: Time limit reached")
    
    print(f"\n--- Evaluation Results ---")
    print(f"Average Reward: {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    print(f"Best Reward: {np.max(rewards):.2f}")
    print(f"Worst Reward: {np.min(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.2f}")
    
    return rewards, steps_list


if __name__ == "__main__":
    render_mode = 'human'
    env = LunarEnvironment(render_mode=render_mode, seed=42)
    agent = RandomAgent(action_space_size=env.get_action_space_size())
    
    print(f"Evaluating {agent.name}")
    print(f"Action space size: {env.get_action_space_size()}")
    print(f"Observation space shape: {env.get_observation_space_shape()}")
    
    rewards, steps = evaluate(agent, env, episodes=5, render=True)
    
    env.close()
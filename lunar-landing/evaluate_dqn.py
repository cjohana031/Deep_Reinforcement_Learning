import numpy as np
import torch
import time
from pathlib import Path

from environments import LunarEnvironment
from agents.dqn import DQNAgent


def evaluate_dqn(
    model_path=None,
    episodes=10,
    max_steps=1000,
    render=True,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    render_mode = 'human' if render else None
    env = LunarEnvironment(render_mode=render_mode, seed=seed)
    
    state_dim = env.get_observation_space_shape()[0]
    action_dim = env.get_action_space_size()
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    
    if not agent.load(model_path):
        print("Warning: No trained model found, using random initialization")
    
    rewards = []
    steps_list = []
    
    print(f"Evaluating DQN Agent")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Device: {agent.device}")
    print("-" * 50)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(max_steps):
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if render:
                env.render()
                time.sleep(0.01)
            
            if done:
                break
        
        rewards.append(total_reward)
        steps_list.append(steps)
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        if terminated:
            if total_reward > 200:
                print("  Status: Perfect landing!")
            elif total_reward > 100:
                print("  Status: Successfully landed!")
            else:
                print("  Status: Crashed")
        elif truncated:
            print("  Status: Time limit reached")
    
    print(f"\n" + "=" * 50)
    print(f"--- Evaluation Results ---")
    print(f"Average Reward: {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    print(f"Best Reward: {np.max(rewards):.2f}")
    print(f"Worst Reward: {np.min(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps_list):.2f}")
    print(f"Success Rate: {sum(1 for r in rewards if r > 100) / len(rewards) * 100:.1f}%")
    
    env.close()
    
    return rewards, steps_list


if __name__ == "__main__":
    evaluate_dqn(episodes=5, render=True)
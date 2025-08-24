import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

from environments import LunarEnvironment
from agents.dqn import DQNAgent
from utils.logger import Logger


def train_dqn(
    episodes=1000,
    max_steps=1000,
    save_freq=100,
    eval_freq=50,
    eval_episodes=10,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = LunarEnvironment(render_mode=None, seed=seed)
    eval_env = LunarEnvironment(render_mode=None, seed=seed + 1000)
    
    state_dim = env.get_observation_space_shape()[0]
    action_dim = env.get_action_space_size()
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=100
    )
    
    logger = Logger()
    best_avg_reward = -float('inf')
    training_info = {
        'start_time': datetime.now().isoformat(),
        'episodes': episodes,
        'seed': seed,
        'device': agent.device
    }
    
    print(f"Starting DQN training on {agent.device}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print("-" * 50)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        losses = []
        
        for step in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        logger.log_episode(episode, total_reward, steps)
        
        avg_loss = np.mean(losses) if losses else 0
        
        if episode % 10 == 0:
            avg_reward_100 = np.mean(logger.get_history()['rewards'][-100:]) if episode >= 99 else np.mean(logger.get_history()['rewards'])
            print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | "
                  f"Avg100: {avg_reward_100:7.2f} | Loss: {avg_loss:.4f} | ε: {agent.epsilon:.3f}")
        
        if episode % eval_freq == 0 and episode > 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                state, _ = eval_env.reset()
                eval_reward = 0
                for _ in range(max_steps):
                    action = agent.act(state, training=False)
                    state, reward, terminated, truncated, _ = eval_env.step(action)
                    eval_reward += reward
                    if terminated or truncated:
                        break
                eval_rewards.append(eval_reward)
            
            avg_eval_reward = np.mean(eval_rewards)
            print(f"  [EVAL] Average reward over {eval_episodes} episodes: {avg_eval_reward:.2f}")
            
            if avg_eval_reward > best_avg_reward:
                best_avg_reward = avg_eval_reward
                agent.save()
                print(f"  [SAVE] New best model! Average reward: {best_avg_reward:.2f}")
        
        if episode % save_freq == 0 and episode > 0:
            agent.save(episode)
    
    agent.save()
    
    training_info['end_time'] = datetime.now().isoformat()
    training_info['final_avg_reward'] = float(np.mean(logger.get_history()['rewards'][-100:]))
    training_info['best_avg_reward'] = float(best_avg_reward)
    
    with open('models/dqn/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.save('logs/dqn/training_history.npz')
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Final average reward (last 100 episodes): {training_info['final_avg_reward']:.2f}")
    print(f"Best evaluation average reward: {best_avg_reward:.2f}")
    print(f"Models saved in: models/dqn/")
    
    env.close()
    eval_env.close()
    
    return logger.get_history()


if __name__ == "__main__":
    train_dqn(episodes=1000)
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import time

from environments.beam_rider_env import BeamRiderEnvironment
from agents.ddqn import DDQNAgent, DDQNUpdater
from utils.logger import Logger


def train_ddqn(episodes=50000, eval_interval=1000, save_interval=10000, render=False):
    print("=" * 70)
    print("STARTING DDQN TRAINING ON BEAMRIDER")
    print("=" * 70)
    
    # Initialize environment
    render_mode = "human" if render else None
    env = BeamRiderEnvironment(render_mode=render_mode, seed=42)
    action_dim = env.get_action_space_size()
    
    print(f"Action space size: {action_dim}")
    print(f"Observation space shape: {env.get_observation_space_shape()}")
    
    # Initialize agent and updater
    agent = DDQNAgent(action_dim=action_dim)
    updater = DDQNUpdater(agent, device=agent.device)
    
    # Set epsilon for agent (needed for act method)
    agent.epsilon = updater.epsilon
    
    # Initialize logger
    logger = Logger(log_dir='logs/ddqn', tensorboard_dir='runs')
    
    print(f"Using device: {agent.device}")
    print(f"Total episodes to train: {episodes:,}")
    print(f"Evaluation every: {eval_interval} episodes")
    print(f"Save checkpoint every: {save_interval} episodes")
    print(f"Rendering: {'Enabled' if render else 'Disabled'}")
    if render:
        print("⚠️  Warning: Rendering will significantly slow down training!")
    print("-" * 70)
    
    # Training loop
    best_reward = float('-inf')
    episode_rewards = []
    
    start_time = time.time()
    last_log_time = start_time
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_loss = []
        
        while True:
            # Select action
            action = agent.act(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            updater.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = updater.update()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update agent's epsilon
            agent.epsilon = updater.epsilon
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Log episode
        logger.log_episode(episode, episode_reward, episode_steps)
        
        # Log training metrics
        if episode_loss:
            avg_loss = np.mean(episode_loss)
            logger.log_training_metrics(episode, loss=avg_loss, epsilon=updater.epsilon)
        
        # Print progress - more frequent for first 1000 episodes, then every 100
        log_interval = 50 if episode <= 1000 else 100
        
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            elapsed_time = time.time() - start_time
            episodes_per_hour = episode / (elapsed_time / 3600) if elapsed_time > 0 else 0
            
            # Calculate ETA
            remaining_episodes = episodes - episode
            eta_hours = remaining_episodes / episodes_per_hour if episodes_per_hour > 0 else 0
            eta_str = f"{int(eta_hours)}h {int((eta_hours % 1) * 60)}m"
            
            # Calculate progress percentage
            progress = (episode / episodes) * 100
            
            # Time since last log
            current_time = time.time()
            time_for_interval = current_time - last_log_time
            last_log_time = current_time
            
            print(f"Episode {episode:6d}/{episodes} ({progress:5.1f}%) | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg100: {avg_reward:6.2f} | "
                  f"ε: {updater.epsilon:.3f} | "
                  f"Buf: {len(updater.replay_buffer):6d} | "
                  f"{log_interval}eps in {time_for_interval:.1f}s | "
                  f"ETA: {eta_str}")
        
        # Evaluate agent
        if episode % eval_interval == 0:
            print(f"🔄 Running evaluation at episode {episode}...")
            eval_start_time = time.time()
            eval_reward = evaluate_agent(agent, env, episodes=5)
            eval_time = time.time() - eval_start_time
            logger.log_training_metrics(episode, eval_reward=eval_reward)
            
            print(f"✅ Evaluation complete ({eval_time:.1f}s): {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save()
                updater.save()
                print(f"🏆 NEW BEST MODEL! Reward: {best_reward:.2f}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            print(f"💾 Saving checkpoint at episode {episode}...")
            checkpoint_start = time.time()
            agent.save(episode)
            updater.save(episode)
            logger.save(f'logs/ddqn/training_history.npz')
            
            # Save training info
            training_info = {
                'episode': episode,
                'best_reward': best_reward,
                'avg_reward_100': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else 0,
                'epsilon': updater.epsilon,
                'buffer_size': len(updater.replay_buffer),
                'training_time_hours': (time.time() - start_time) / 3600
            }
            
            info_path = Path('models/ddqn/training_info.json')
            info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(info_path, 'w') as f:
                json.dump(training_info, f, indent=2)
            
            checkpoint_time = time.time() - checkpoint_start
            print(f"✅ Checkpoint saved ({checkpoint_time:.1f}s)")
    
    # Final save
    print("💾 Saving final models...")
    agent.save()
    updater.save()
    logger.save(f'logs/ddqn/training_history.npz')
    logger.close()
    
    env.close()
    
    total_time = time.time() - start_time
    print("=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Total episodes: {episodes:,}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Average episodes/hour: {episodes/(total_time/3600):.1f}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 70)


def evaluate_agent(agent, env, episodes=5):
    """Evaluate the agent for a few episodes without training"""
    total_rewards = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


if __name__ == "__main__":
    train_ddqn(episodes=50000, render=False)
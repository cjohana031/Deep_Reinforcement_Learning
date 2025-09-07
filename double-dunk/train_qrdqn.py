import torch
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

from agents.qrdqn import QRDQNAgent, QRDQNUpdater
from environments.double_dunk_env import DoubleDunkEnvironment
from utils.logger import Logger

def main():
    # Configuration
    config = {
        "environment": "DoubleDunk",
        "agent": "QR-DQN",
        "num_episodes": 1000,
        "eval_episodes": 10,
        "eval_frequency": 10,
        "save_frequency": 10,
        "max_steps_per_episode": 10000,
        "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        
        # Agent parameters
        "num_quantiles": 51,
        
        # Training parameters
        "target_update_freq": 10000,
        "buffer_size": 100_000,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay_steps": 1_000_000,
        "warmup_steps": 50000,
        "kappa": 1.0,
        
        # Prioritized replay parameters
        "alpha": 0.6,
        "beta": 0.4,
        "beta_increment": 0.001
    }

    print("QR-DQN Training Configuration:")
    print(f"Device: {config['device']}")
    print(f"Episodes: {config['num_episodes']}")
    print(f"Quantiles: {config['num_quantiles']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Buffer size: {config['buffer_size']:,}")
    print(f"Target update frequency: {config['target_update_freq']:,}")
    print("=" * 50)

    # Create environment
    env = DoubleDunkEnvironment()

    # Create agent and updater
    agent = QRDQNAgent(
        action_dim=env.action_space.n,
        model_dir='models/qrdqn',
        input_channels=4,
        num_quantiles=config['num_quantiles']
    )

    updater = QRDQNUpdater(
        agent=agent,
        target_update_freq=config['target_update_freq'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        gamma=config['gamma'],
        epsilon=config['epsilon_start'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay_steps=config['epsilon_decay_steps'],
        kappa=config['kappa'],
        device=config['device']
    )

    # Create logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/qrdqn"
    logger = Logger(log_dir)

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    recent_rewards = []
    best_mean_reward = float('-inf')

    print("Starting QR-DQN training...")
    
    for episode in range(config['num_episodes']):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0

        for step in range(config['max_steps_per_episode']):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            updater.store_transition(state, action, reward, next_state, done)
            
            # Update the agent
            if len(updater.replay_buffer) > config['warmup_steps']:
                loss = updater.update()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)
        
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        # Logging
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        mean_recent_reward = np.mean(recent_rewards)
        
        logger.log_episode(episode, episode_reward, episode_length, avg_loss, updater.epsilon)

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Mean(100): {mean_recent_reward:7.2f} | "
                  f"Length: {episode_length:4d} | "
                  f"Epsilon: {updater.epsilon:.4f} | "
                  f"Buffer: {len(updater.replay_buffer):6d} | "
                  f"Loss: {avg_loss:.4f}")

        # Periodic evaluation
        if episode % config['eval_frequency'] == 0 and episode > 0:
            eval_rewards = []
            for _ in range(config['eval_episodes']):
                eval_state, _ = env.reset()
                eval_reward = 0
                
                for eval_step in range(config['max_steps_per_episode']):
                    eval_action = agent.act(eval_state, training=False)
                    eval_state, eval_r, eval_terminated, eval_truncated, _ = env.step(eval_action)
                    eval_reward += eval_r
                    eval_done = eval_terminated or eval_truncated
                    
                    if eval_done:
                        break
                
                eval_rewards.append(eval_reward)
            
            mean_eval_reward = np.mean(eval_rewards)
            logger.log_training_metrics(episode, eval_reward=mean_eval_reward)
            
            print(f"\n=== Evaluation at Episode {episode} ===")
            print(f"Mean Eval Reward: {mean_eval_reward:.2f} ± {np.std(eval_rewards):.2f}")
            print("=" * 40)

            # Save best model
            if mean_eval_reward > best_mean_reward:
                best_mean_reward = mean_eval_reward
                agent.save()
                updater.save()
                print(f"New best model saved! Mean reward: {best_mean_reward:.2f}")

        # Periodic model saving
        if episode % config['save_frequency'] == 0 and episode > 0:
            agent.save(episode)
            updater.save(episode)

    # Final save
    print("\nTraining completed!")
    agent.save('final')
    updater.save('final')

    # Save training info
    training_info = {
        'config': config,
        'total_episodes': config['num_episodes'],
        'best_mean_reward': best_mean_reward,
        'final_epsilon': updater.epsilon,
        'final_buffer_size': len(updater.replay_buffer),
        'timestamp': timestamp
    }

    info_path = Path(agent.model_dir) / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    logger.save_history(f"qrdqn_training_{timestamp}.json")

    print(f"Training completed! Best mean reward: {best_mean_reward:.2f}")
    print(f"Models saved in: {agent.model_dir}")

if __name__ == "__main__":
    main()
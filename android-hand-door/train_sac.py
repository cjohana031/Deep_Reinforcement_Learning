import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import time
import argparse

from environments.adroit_hand_door_env import AdroitHandDoorEnvironment
from agents.sac import SACAgent
from utils.logger import Logger


def train_sac(episodes=5000,
              eval_interval=100,
              save_interval=200,
              render=False,
              restart=False,
              reward_type='dense',
              max_episode_steps=200):
    """
    Train SAC agent on Adroit Hand Door environment.

    Args:
        episodes: Number of training episodes
        eval_interval: Episodes between evaluations
        save_interval: Episodes between checkpoints
        render: Enable rendering during training
        restart: Load and continue from best saved model
        reward_type: 'dense' or 'sparse' reward function
        max_episode_steps: Maximum steps per episode
    """
    print("=" * 70)
    print("STARTING SAC TRAINING ON ADROIT HAND DOOR")
    print("=" * 70)

    # Initialize environment
    render_mode = "human" if render else None
    env = AdroitHandDoorEnvironment(
        render_mode=render_mode,
        seed=42,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type
    )

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action bounds: {env.get_action_bounds()}")

    # SAC hyperparameters optimized for robotics manipulation
    sac_params = {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'lr_alpha': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,  # Initial entropy coefficient
        'automatic_entropy_tuning': True,
        'hidden_size': 256,
        'buffer_size': 1_000_000,
        'batch_size': 256
    }

    # Initialize SAC agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **sac_params
    )

    # Load existing models if restart flag is set
    if restart:
        print("🔄 Loading existing models...")
        if agent.load():
            print("✅ Successfully loaded existing models")
        else:
            print("❌ No existing models found, starting fresh")

    # Initialize logger
    logger = Logger(log_dir='logs/sac', tensorboard_dir='runs')

    print(f"Using device: {agent.device}")
    print(f"Total episodes to train: {episodes:,}")
    print(f"Evaluation every: {eval_interval} episodes")
    print(f"Save checkpoint every: {save_interval} episodes")
    print(f"Reward type: {reward_type}")
    print(f"Rendering: {'Enabled' if render else 'Disabled'}")
    if render:
        print("⚠️  Warning: Rendering will significantly slow down training!")
    print("-" * 70)

    # Training loop
    best_reward = float('-inf')
    best_success_rate = 0.0
    episode_rewards = []
    success_rates = []

    start_time = time.time()
    last_log_time = start_time

    # Warm-up phase: collect initial experience
    print("🔥 Starting warm-up phase (random actions)...")
    warmup_steps = 1000
    warmup_step = 0

    for episode in range(1, episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_losses = []
        is_warmup = warmup_step < warmup_steps

        while True:
            # Select action
            if is_warmup:
                # Random actions during warmup
                action = np.random.uniform(
                    env.action_space.low,
                    env.action_space.high,
                    size=action_dim
                )
                warmup_step += 1
            else:
                action = agent.act(state, training=True)

            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update agent (only after warmup)
            if not is_warmup and len(agent.replay_buffer) >= agent.batch_size:
                losses = agent.update()
                if losses:
                    episode_losses.append(losses)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        # Calculate success rate (if door opened successfully)
        success_rate = env.get_success_rate(step_info) if 'step_info' in locals() else 0.0
        success_rates.append(success_rate)

        # Log episode
        avg_losses = None
        alpha_value = None
        if episode_losses:
            # Average losses across all updates in the episode
            avg_losses = {}
            for key in episode_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in episode_losses])
            alpha_value = avg_losses.get('alpha', None)

        logger.log_episode(
            episode, episode_reward, episode_steps,
            losses=avg_losses, alpha=alpha_value, success_rate=success_rate
        )

        # Log training metrics
        if avg_losses:
            logger.log_training_metrics(episode, losses=avg_losses, alpha=alpha_value)

        # Print progress - more frequent early on
        log_interval = 10 if episode <= 1000 else 50

        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)

            elapsed_time = time.time() - start_time
            episodes_per_hour = episode / (elapsed_time / 3600) if elapsed_time > 0 else 0

            # Calculate ETA
            remaining_episodes = episodes - episode
            eta_hours = remaining_episodes / episodes_per_hour if episodes_per_hour > 0 else 0
            eta_str = f"{int(eta_hours)}h {int((eta_hours % 1) * 60)}m"

            # Progress percentage
            progress = (episode / episodes) * 100

            # Time since last log
            current_time = time.time()
            time_for_interval = current_time - last_log_time
            last_log_time = current_time

            status = "WARMUP" if is_warmup else "TRAINING"
            alpha_str = f"α: {alpha_value:.3f}" if alpha_value is not None else "α: N/A"

            print(f"[{status}] Episode {episode:6d}/{episodes} ({progress:5.1f}%) | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Avg100: {avg_reward:7.2f} | "
                  f"Success: {success_rate:.2f} | "
                  f"Avg100Success: {avg_success:.3f} | "
                  f"{alpha_str} | "
                  f"Buf: {len(agent.replay_buffer):7d} | "
                  f"{log_interval}eps in {time_for_interval:.1f}s | "
                  f"ETA: {eta_str}")

        # Evaluate agent
        if episode % eval_interval == 0 and not is_warmup:
            print(f"🔄 Running evaluation at episode {episode}...")
            eval_start_time = time.time()
            eval_reward, eval_success = evaluate_agent(env, agent, episodes=5)
            eval_time = time.time() - eval_start_time

            logger.log_training_metrics(episode, eval_reward=eval_reward)

            print(f"✅ Evaluation complete ({eval_time:.1f}s): "
                  f"Reward: {eval_reward:.2f}, Success: {eval_success:.3f}")

            # Save best model based on success rate (primary) and reward (secondary)
            is_better = (eval_success > best_success_rate or
                        (eval_success == best_success_rate and eval_reward > best_reward))

            if is_better:
                best_reward = eval_reward
                best_success_rate = eval_success
                agent.save()
                print(f"🏆 NEW BEST MODEL! "
                      f"Success: {best_success_rate:.3f}, Reward: {best_reward:.2f}")

        # Save checkpoint
        if episode % save_interval == 0:
            print(f"💾 Saving checkpoint at episode {episode}...")
            checkpoint_start = time.time()
            agent.save(episode)
            logger.save(f'logs/sac/training_history.npz')

            # Save training info
            training_info = {
                'episode': episode,
                'best_reward': best_reward,
                'best_success_rate': best_success_rate,
                'avg_reward_100': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else 0,
                'avg_success_100': np.mean(success_rates[-100:]) if len(success_rates) >= 100 else 0,
                'buffer_size': len(agent.replay_buffer),
                'training_time_hours': (time.time() - start_time) / 3600,
                'sac_params': sac_params
            }

            info_path = Path('models/sac/training_info.json')
            info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(info_path, 'w') as f:
                json.dump(training_info, f, indent=2)

            checkpoint_time = time.time() - checkpoint_start
            print(f"✅ Checkpoint saved ({checkpoint_time:.1f}s)")

    # Final save
    print("💾 Saving final models...")
    agent.save()
    logger.save(f'logs/sac/training_history.npz')
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
    print(f"Best success rate: {best_success_rate:.3f}")
    print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final average success rate (last 100): {np.mean(success_rates[-100:]):.3f}")
    print("=" * 70)


def evaluate_agent(env, agent, episodes=5):
    """
    Evaluate the agent for a few episodes without training.

    Returns:
        avg_reward: Average episode reward
        avg_success: Average success rate
    """
    total_rewards = []
    total_successes = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        success = 0

        while True:
            action = agent.act(state, training=False)  # Deterministic policy
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                success = env.get_success_rate(info)
                break

        total_rewards.append(episode_reward)
        total_successes.append(success)

    return np.mean(total_rewards), np.mean(total_successes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAC agent on Adroit Hand Door')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Episodes between evaluations')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='Episodes between checkpoints')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering during training')
    parser.add_argument('--restart', action='store_true',
                        help='Load and continue from best saved model')
    parser.add_argument('--reward_type', choices=['dense', 'sparse'], default='dense',
                        help='Reward function type')
    parser.add_argument('--max_episode_steps', type=int, default=200,
                        help='Maximum steps per episode')

    args = parser.parse_args()

    train_sac(
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        render=args.render,
        restart=args.restart,
        reward_type=args.reward_type,
        max_episode_steps=args.max_episode_steps
    )
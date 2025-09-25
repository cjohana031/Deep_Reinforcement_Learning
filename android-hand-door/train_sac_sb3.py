import numpy as np
import time
from pathlib import Path
import json
from datetime import datetime

from environments.adroit_hand_door_env import AdroitHandDoorEnvironment
from agents.sac_sb3 import SB3SACAgent, TrainingCallback
from utils.logger import Logger


def train_sac(episodes=5000,
              eval_interval=100,
              save_interval=200,
              render=False,
              restart=False,
              reward_type='dense',
              max_episode_steps=200):
    """
    Train SAC agent on Adroit Hand Door environment using Stable Baselines3.

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
    print("STARTING SAC TRAINING ON ADROIT HAND DOOR (Stable Baselines3)")
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
    tensorboard_log = "runs/sac_sb3" if not render else None

    # Initialize SAC agent
    agent = SB3SACAgent(
        env=env,
        model_dir='models/sac_sb3',
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_entropy='auto',
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256])
        ),
        verbose=1,
        seed=42,
        tensorboard_log=tensorboard_log
    )

    # Load existing models if restart flag is set
    if restart:
        print("🔄 Loading existing models...")
        if agent.load():
            print("✅ Successfully loaded existing models")
        else:
            print("❌ No existing models found, starting fresh")

    # Initialize logger
    logger = Logger(log_dir='logs/sac_sb3', tensorboard_dir='runs/sac_sb3')

    print(f"Using device: {agent.model.device}")
    print(f"Total episodes to train: {episodes:,}")
    print(f"Evaluation every: {eval_interval} episodes")
    print(f"Save checkpoint every: {save_interval} episodes")
    print(f"Reward type: {reward_type}")
    print(f"Rendering: {'Enabled' if render else 'Disabled'}")
    if render:
        print("⚠️  Warning: Rendering will significantly slow down training!")
    print("-" * 70)

    # Calculate total timesteps (estimate)
    # Assuming average episode length of max_episode_steps/2
    estimated_steps_per_episode = max_episode_steps // 2
    total_timesteps = episodes * estimated_steps_per_episode

    # Setup training callback
    callback = TrainingCallback(
        eval_freq=eval_interval * estimated_steps_per_episode,
        save_freq=save_interval * estimated_steps_per_episode,
        eval_episodes=5,
        save_path=str(agent.model_dir),
        verbose=1,
        custom_logger=logger
    )

    start_time = time.time()

    try:
        # Train the agent
        print("🚀 Starting training...")
        agent.train(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    # Final save
    print("💾 Saving final models...")
    agent.save()

    # Save training info
    total_time = time.time() - start_time
    training_info = {
        'episodes': episodes,
        'total_timesteps': total_timesteps,
        'training_time_hours': total_time / 3600,
        'reward_type': reward_type,
        'max_episode_steps': max_episode_steps,
        'model_type': 'SAC_SB3'
    }

    info_path = Path('models/sac_sb3/training_info.json')
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    env.close()

    print("=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Total episodes: {episodes:,}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Total training time: {total_time/3600:.2f} hours")
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
    import argparse

    parser = argparse.ArgumentParser(description='Train SAC agent on Adroit Hand Door using SB3')
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
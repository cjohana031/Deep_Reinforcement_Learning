import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import time
import argparse

from environments.adroit_hand_door_env import AdroitHandDoorEnvironment
from agents.sac_sb3 import SB3SACAgent, TrainingCallback
from utils.logger import Logger
from stable_baselines3.common.vec_env  import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env


# Factory for parallel training (VecEnv)
def make_env(render_mode=None, seed=42, max_episode_steps=200, reward_type="dense", rank=0):
    """Create environment factory function for parallel training"""
    def _init():
        env = AdroitHandDoorEnvironment(
            render_mode=render_mode,
            seed=seed + rank,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type
        )
        return env
    return _init


def train_sac_parallel(episodes=5000,
                      n_envs=4,
                      eval_interval=100,
                      save_interval=200,
                      render=False,
                      restart=False,
                      reward_type='dense',
                      max_episode_steps=200,
                      vec_env_cls=None):
    """
    Train SAC agent with parallel environments on Adroit Hand Door.

    Args:
        episodes: Number of training episodes
        n_envs: Number of parallel environments
        eval_interval: Episodes between evaluations
        save_interval: Episodes between checkpoints
        render: Enable rendering during training (only works with n_envs=1)
        restart: Load and continue from best saved model
        reward_type: 'dense' or 'sparse' reward function
        max_episode_steps: Maximum steps per episode
        vec_env_cls: Vectorized environment class (SubprocVecEnv or DummyVecEnv)
    """
    print("=" * 80)
    print("STARTING PARALLEL SAC TRAINING ON ADROIT HAND DOOR")
    print("=" * 80)

    # Disable rendering for parallel training
    if n_envs > 1 and render:
        print("⚠️  Warning: Rendering disabled for parallel training (n_envs > 1)")
        render = False

    vec_env_cls = SubprocVecEnv

    # Create environment factory function for SB3
    def env_factory():
        return AdroitHandDoorEnvironment(
            render_mode="human" if render and n_envs == 1 else None,
            seed=42,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type
        )

    # Get environment info from a single instance
    single_env = env_factory()
    state_dim = single_env.get_state_dim()
    action_dim = single_env.get_action_dim()
    action_bounds = single_env.get_action_bounds()
    single_env.close()

    print(f"Environment info:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Action bounds: {action_bounds}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Vectorization: {vec_env_cls.__name__}")

    # SAC hyperparameters optimized for parallel training
    sac_params = {
        'learning_rate': 1e-4,
        'buffer_size': 1_000_000,
        'learning_starts': 10000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,  # Keep gradient steps constant for stability
        'ent_coef': 'auto',
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'use_sde': False,
        'policy_kwargs': dict(
            net_arch=dict(pi=[512, 512], qf=[512, 512]),
            activation_fn=torch.nn.ReLU
        ),
        'verbose': 1,
        'seed': 42,
        'tensorboard_log': 'runs/sac_parallel',
        'n_envs': n_envs,
        'vec_env_cls': vec_env_cls
    }

    # Initialize SAC agent with parallel environments
    agent = SB3SACAgent(
        env=env_factory,
        model_dir='models/sac_parallel',
        **sac_params
    )

    # Load existing models if restart flag is set
    if restart:
        print("🔄 Loading existing models...")
        if agent.load():
            print("✅ Successfully loaded existing models")
        else:
            print("❌ No existing models found, starting fresh")

    print(f"Using device: {agent.model.device}")
    print(f"Buffer size: {agent.model.buffer_size:,}")
    print(f"Batch size: {agent.model.batch_size}")
    print(f"Gradient steps per update: {agent.model.gradient_steps}")
    print(f"Total timesteps to train: {episodes * max_episode_steps * n_envs:,}")
    print(f"Evaluation every: {eval_interval} episodes")
    print(f"Save checkpoint every: {save_interval} episodes")
    print(f"Reward type: {reward_type}")
    print("-" * 80)

    # Initialize logger for CSV logging
    logger = Logger(log_dir='logs/sac_parallel', tensorboard_dir='runs/sac_parallel')

    # Setup training callback
    callback = TrainingCallback(
        eval_freq=eval_interval * max_episode_steps,  # Convert episodes to timesteps
        save_freq=save_interval * max_episode_steps,
        eval_episodes=5,
        save_path='models/sac_parallel',
        verbose=1,
        custom_logger=logger
    )

    # Calculate total timesteps
    total_timesteps = episodes * max_episode_steps * n_envs

    print(f"🚀 Starting parallel training...")
    print(f"Expected episodes per environment: ~{episodes}")
    print(f"Expected total episodes across all environments: ~{episodes * n_envs}")
    print(f"Training for {total_timesteps:,} total timesteps")

    start_time = time.time()

    # Train the agent
    agent.train(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Final save
    print("💾 Saving final models...")
    agent.save()

    total_time = time.time() - start_time
    print("=" * 80)
    print("🎉 PARALLEL TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Training environments: {n_envs}")
    print(f"Vectorization: {vec_env_cls.__name__}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Timesteps per hour: {total_timesteps/(total_time/3600):,.0f}")
    print(f"Speedup vs single env: ~{n_envs}x (theoretical)")

    # Save training info
    training_info = {
        'n_envs': n_envs,
        'vec_env_cls': vec_env_cls.__name__,
        'total_timesteps': total_timesteps,
        'training_time_hours': total_time / 3600,
        'timesteps_per_hour': total_timesteps / (total_time / 3600),
        'theoretical_speedup': n_envs,
        'sac_params': sac_params,
        'reward_type': reward_type,
        'max_episode_steps': max_episode_steps
    }

    info_path = Path('models/sac_parallel/training_info.json')
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    print("=" * 80)

    # Clean up
    agent.env.close()


def evaluate_parallel_agent(model_path='models/sac_parallel/sac_best.zip', episodes=10):
    """Evaluate a trained parallel agent"""
    print("🔍 Evaluating trained agent...")

    # Create single environment for evaluation
    env_factory = make_env(render_mode="human", seed=42)
    env = env_factory()

    try:
        # Load the trained model
        from stable_baselines3 import SAC
        model = SAC.load(model_path, env=env)

        total_rewards = []
        total_successes = []

        for i in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            success = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    success = env.get_success_rate(info)
                    break

            total_rewards.append(episode_reward)
            total_successes.append(success)

            print(f"Episode {i+1:2d}: Reward = {episode_reward:7.1f}, Success = {success:.2f}")

        avg_reward = np.mean(total_rewards)
        avg_success = np.mean(total_successes)
        std_reward = np.std(total_rewards)

        print(f"\n📊 Evaluation Results ({episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Success: {avg_success:.3f}")
        print(f"Success Rate: {np.sum(np.array(total_successes) > 0.5) / episodes * 100:.1f}%")

        return avg_reward, avg_success

    finally:
        # Ensure environment is properly closed
        if hasattr(env, 'close'):
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAC agent with parallel environments')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--n_envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval_interval', type=int, default=2000,
                        help='Episodes between evaluations')
    parser.add_argument('--save_interval', type=int, default=50000,
                        help='Episodes between checkpoints')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering (only works with n_envs=1)')
    parser.add_argument('--restart', action='store_true',
                        help='Load and continue from best saved model')
    parser.add_argument('--reward_type', choices=['dense', 'sparse'], default='dense',
                        help='Reward function type')
    parser.add_argument('--max_episode_steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--vec_env_cls', choices=['subproc', 'dummy'], default='subproc',
                        help='Vectorized environment class')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained model instead of training')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of evaluation episodes')

    args = parser.parse_args()

    if args.evaluate:
        evaluate_parallel_agent(episodes=args.eval_episodes)
    else:
        vec_env_cls = SubprocVecEnv if args.vec_env_cls == 'subproc' else DummyVecEnv

        train_sac_parallel(
            episodes=args.episodes,
            n_envs=args.n_envs,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            render=args.render,
            restart=args.restart,
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps,
            vec_env_cls=vec_env_cls
        )
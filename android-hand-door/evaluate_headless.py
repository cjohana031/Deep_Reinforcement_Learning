#!/usr/bin/env python3
"""
Headless evaluation script for SAC agent on Adroit Hand Door
Works without display/rendering - shows console metrics only
"""

import numpy as np
import torch
from pathlib import Path
import argparse
import os

# Disable OpenGL/rendering entirely
os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from environments.adroit_hand_door_env import AdroitHandDoorEnvironment
from stable_baselines3 import SAC


def evaluate_headless(model_path='models/sac_parallel/sac_best.zip', episodes=20, verbose=True):
    """Evaluate trained SAC agent without rendering"""

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None, None

    print(f"🤖 Loading model: {model_path}")
    print(f"📊 Running {episodes} evaluation episodes (headless)...")

    # Create environment without rendering
    env = AdroitHandDoorEnvironment(
        render_mode=None,  # No rendering
        seed=42,
        max_episode_steps=200,
        reward_type="dense"
    )

    # Load trained model
    try:
        model = SAC.load(model_path, env=env)
        print(f"✅ Model loaded successfully")
        print(f"🔧 Device: {model.device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        env.close()
        return None, None

    # Run evaluation episodes
    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    print("\n" + "="*60)
    print("EPISODE RESULTS")
    print("="*60)

    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        success = False

        while True:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Check if episode is done
            if terminated or truncated:
                # Extract success info
                success = info.get('success', False)
                if hasattr(env, 'get_success_rate'):
                    success_rate = env.get_success_rate(info)
                    success = success_rate > 0.5
                break

        episode_rewards.append(episode_reward)
        episode_successes.append(1 if success else 0)
        episode_lengths.append(episode_length)

        if verbose:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"Episode {episode+1:2d}: Reward={episode_reward:7.1f}, Length={episode_length:3d}, {status}")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = np.mean(episode_successes) * 100

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes evaluated:     {episodes}")
    print(f"Average reward:         {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {mean_length:.1f} steps")
    print(f"Success rate:           {success_rate:.1f}% ({int(np.sum(episode_successes))}/{episodes})")
    print(f"Min reward:             {np.min(episode_rewards):.2f}")
    print(f"Max reward:             {np.max(episode_rewards):.2f}")

    # Performance assessment
    print("\n" + "="*60)
    print("PERFORMANCE ASSESSMENT")
    print("="*60)
    if success_rate >= 80:
        print("🏆 EXCELLENT: Agent performs very well!")
    elif success_rate >= 60:
        print("🥈 GOOD: Agent performs well")
    elif success_rate >= 40:
        print("🥉 FAIR: Agent has moderate performance")
    elif success_rate >= 20:
        print("⚠️  POOR: Agent needs more training")
    else:
        print("❌ VERY POOR: Agent is not learning effectively")

    env.close()
    return mean_reward, success_rate


def list_available_models():
    """List all available trained models"""
    model_dir = Path('models/sac_parallel')
    if not model_dir.exists():
        print("❌ No models directory found")
        return []

    models = list(model_dir.glob('*.zip'))
    models.sort()

    print("\n📁 Available models:")
    for i, model in enumerate(models):
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  {i+1:2d}. {model.name} ({size_mb:.1f} MB)")

    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SAC agent (headless)')
    parser.add_argument('--model', type=str, default='models/sac_parallel/sac_best.zip',
                        help='Path to model file')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--list', action='store_true',
                        help='List available models')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')

    args = parser.parse_args()

    if args.list:
        models = list_available_models()
        if models:
            print(f"\nTo evaluate a specific model:")
            print(f"python evaluate_headless.py --model models/sac_parallel/MODEL_NAME.zip")
    else:
        evaluate_headless(
            model_path=args.model,
            episodes=args.episodes,
            verbose=not args.quiet
        )
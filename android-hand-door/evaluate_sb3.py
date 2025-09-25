#!/usr/bin/env python3
"""
Evaluation script for trained SAC agent (Stable Baselines3) on Adroit Hand Door environment.
"""

import numpy as np
import time
from pathlib import Path
import argparse

from environments.adroit_hand_door_env import AdroitHandDoorEnvironment
from agents.sac_sb3 import SB3SACAgent


def evaluate_sac(episodes=10,
                 model_path=None,
                 render=False,
                 reward_type='dense',
                 max_episode_steps=200,
                 seed=42,
                 verbose=True):
    """
    Evaluate trained SAC agent (SB3) on Adroit Hand Door environment.

    Args:
        episodes: Number of evaluation episodes
        model_path: Path to model checkpoint
        render: Enable rendering
        reward_type: 'dense' or 'sparse' reward function
        max_episode_steps: Maximum steps per episode
        seed: Random seed
        verbose: Print detailed results

    Returns:
        dict: Evaluation statistics
    """
    if verbose:
        print("=" * 70)
        print("EVALUATING SAC AGENT (SB3) ON ADROIT HAND DOOR")
        print("=" * 70)

    # Initialize environment
    render_mode = "human" if render else None
    env = AdroitHandDoorEnvironment(
        render_mode=render_mode,
        seed=seed,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type
    )

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    if verbose:
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Action bounds: {env.get_action_bounds()}")

    # Initialize SAC agent
    agent = SB3SACAgent(
        env=env,
        model_dir='models/sac_sb3',
        verbose=0
    )

    # Load model
    if model_path:
        model_path = Path(model_path)
        if not model_path.suffix:  # Add .zip if no extension
            model_path = model_path.with_suffix('.zip')

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load from specific path (remove .zip extension for SB3)
        load_path = str(model_path.with_suffix(''))
    else:
        load_path = None

    if not agent.load(load_path):
        raise RuntimeError("Failed to load model")

    if verbose:
        print(f"Model loaded successfully from: {model_path or 'best model'}")
        print(f"Evaluating for {episodes} episodes...")
        print("-" * 70)

    # Evaluation
    episode_rewards = []
    episode_lengths = []
    success_rates = []

    start_time = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0

        episode_start_time = time.time()

        while True:
            # Select action deterministically
            action = agent.act(state, training=False)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        episode_time = time.time() - episode_start_time

        # Calculate success rate
        success_rate = env.get_success_rate(info) if info else 0.0

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        success_rates.append(success_rate)

        if verbose:
            print(f"Episode {episode:3d}/{episodes} | "
                  f"Reward: {episode_reward:8.1f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Success: {success_rate:.2f} | "
                  f"Time: {episode_time:.2f}s")

    # Calculate statistics
    total_time = time.time() - start_time

    stats = {
        'episodes': episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_success_rate': np.mean(success_rates),
        'success_rate': np.mean(success_rates),  # Alias for compatibility
        'total_successes': np.sum(success_rates),
        'evaluation_time': total_time,
        'reward_type': reward_type
    }

    if verbose:
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Episodes evaluated: {episodes}")
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Reward type: {reward_type}")
        print("-" * 70)
        print("REWARDS:")
        print(f"  Mean:     {stats['mean_reward']:8.2f}")
        print(f"  Std:      {stats['std_reward']:8.2f}")
        print(f"  Min:      {stats['min_reward']:8.2f}")
        print(f"  Max:      {stats['max_reward']:8.2f}")
        print("-" * 70)
        print("EPISODE LENGTHS:")
        print(f"  Mean:     {stats['mean_length']:8.1f}")
        print(f"  Std:      {stats['std_length']:8.1f}")
        print("-" * 70)
        print("SUCCESS RATE:")
        print(f"  Success Rate:     {stats['mean_success_rate']:6.3f} ({stats['mean_success_rate']*100:.1f}%)")
        print(f"  Total Successes:  {int(stats['total_successes'])}/{episodes}")
        print("=" * 70)

    env.close()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate trained SAC agent (SB3) on Adroit Hand Door',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (if not specified, uses best model)')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--reward_type', choices=['dense', 'sparse'], default='dense',
                        help='Reward function type')
    parser.add_argument('--max_episode_steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    try:
        stats = evaluate_sac(
            episodes=args.episodes,
            model_path=args.model_path,
            render=args.render,
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            verbose=True
        )

        # Save results if multiple episodes
        if args.episodes > 1:
            results_path = Path(f'evaluation_results_{int(time.time())}.json')
            import json
            with open(results_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nResults saved to: {results_path}")

    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        exit(1)
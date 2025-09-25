#!/usr/bin/env python3
"""
Evaluation script for trained SAC agent on Adroit Hand Door environment.
"""

import numpy as np
import torch
import time
from pathlib import Path
import argparse

from environments.adroit_hand_door_env import AdroitHandDoorEnvironment
from agents.sac import SACAgent


def evaluate_sac(episodes=10,
                 model_path=None,
                 render=False,
                 reward_type='dense',
                 max_episode_steps=200,
                 seed=42,
                 verbose=True):
    """
    Evaluate trained SAC agent on Adroit Hand Door environment.

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
        print("EVALUATING SAC AGENT ON ADROIT HAND DOOR")
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
        print(f"Environment: AdroitHandDoor-v1 ({reward_type} rewards)")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Max episode steps: {max_episode_steps}")

    # Initialize agent
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)

    # Load trained model
    if model_path is None:
        model_path = Path('models/sac/sac_best.pth')
    else:
        model_path = Path(model_path)

    if not agent.load(model_path):
        raise FileNotFoundError(f"Could not load model from {model_path}")

    if verbose:
        print(f"Loaded model from: {model_path}")
        print(f"Device: {agent.device}")
        print(f"Episodes to evaluate: {episodes}")
        print("-" * 70)

    # Evaluation
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    episode_times = []

    total_start_time = time.time()

    for episode in range(1, episodes + 1):
        episode_start_time = time.time()
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_success = 0

        if verbose:
            print(f"Episode {episode}/{episodes}: ", end="", flush=True)

        while True:
            # Use deterministic policy for evaluation
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if terminated or truncated:
                episode_success = env.get_success_rate(info)
                break

        episode_time = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        success_rates.append(episode_success)
        episode_times.append(episode_time)

        if verbose:
            success_str = "SUCCESS" if episode_success > 0.5 else "FAILED"
            print(f"Reward: {episode_reward:8.2f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Time: {episode_time:5.1f}s | "
                  f"Status: {success_str}")

        if render:
            time.sleep(0.1)  # Small delay for better visualization

    total_time = time.time() - total_start_time
    env.close()

    # Calculate statistics
    stats = {
        'episodes': episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': np.mean(success_rates),
        'total_time': total_time,
        'mean_episode_time': np.mean(episode_times),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates
    }

    if verbose:
        print("=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Episodes evaluated: {stats['episodes']}")
        print(f"Total time: {stats['total_time']:.1f}s ({stats['mean_episode_time']:.1f}s per episode)")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Mean reward: {stats['mean_reward']:8.2f} ± {stats['std_reward']:.2f}")
        print(f"  Min reward:  {stats['min_reward']:8.2f}")
        print(f"  Max reward:  {stats['max_reward']:8.2f}")
        print()
        print(f"  Mean episode length: {stats['mean_length']:6.1f} ± {stats['std_length']:.1f} steps")
        print(f"  Success rate:        {stats['success_rate']:6.1%} ({int(stats['success_rate'] * episodes)}/{episodes} episodes)")
        print()

        # Additional analysis
        if len(episode_rewards) > 1:
            print("ADDITIONAL ANALYSIS:")
            successful_episodes = [r for i, r in enumerate(episode_rewards) if success_rates[i] > 0.5]
            if successful_episodes:
                print(f"  Reward (successful episodes): {np.mean(successful_episodes):6.2f} ± {np.std(successful_episodes):.2f}")

            failed_episodes = [r for i, r in enumerate(episode_rewards) if success_rates[i] <= 0.5]
            if failed_episodes:
                print(f"  Reward (failed episodes):     {np.mean(failed_episodes):6.2f} ± {np.std(failed_episodes):.2f}")

        print("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAC agent on Adroit Hand Door')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering during evaluation')
    parser.add_argument('--reward_type', choices=['dense', 'sparse'], default='dense',
                        help='Reward function type')
    parser.add_argument('--max_episode_steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    try:
        evaluate_sac(
            episodes=args.episodes,
            model_path=args.model_path,
            render=args.render,
            reward_type=args.reward_type,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
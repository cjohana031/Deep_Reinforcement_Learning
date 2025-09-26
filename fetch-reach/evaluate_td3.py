#!/usr/bin/env python3
"""
Evaluation script for TD3 agent on FetchReachDense-v4 environment.

This script evaluates a trained TD3 agent and can optionally render the environment
to visualize the learned policy.
"""

import argparse
import numpy as np
import torch
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from agents.td3 import TD3Agent
from environments.hand_reach_env import make_hand_reach_env


class TD3Evaluator:
    """Evaluates a trained TD3 agent."""

    def __init__(self, args):
        """
        Initialize evaluator.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.setup_environment()
        self.setup_agent()

    def setup_environment(self):
        """Setup the evaluation environment."""
        print("Setting up evaluation environment...")

        render_mode = "human" if self.args.render else None

        try:
            self.env = make_hand_reach_env(
                render_mode=render_mode,
                max_episode_steps=self.args.max_episode_steps
            )
            print(f"Environment created successfully!")
            print(f"Render mode: {render_mode}")
        except Exception as e:
            print(f"Error creating environment: {e}")
            print("Make sure gymnasium-robotics is installed:")
            print("pip install gymnasium-robotics")
            sys.exit(1)

    def setup_agent(self):
        """Setup and load the trained TD3 agent."""
        print("Setting up TD3 agent...")

        self.agent = TD3Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            max_action=1.0,
            model_dir=self.args.model_dir
        )

        # Load trained model
        if self.args.model_path:
            model_path = self.args.model_path
        else:
            model_path = Path(self.args.model_dir) / "td3_best.pth"

        if self.agent.load(model_path):
            print(f"Successfully loaded trained model from {model_path}")
        else:
            print(f"Failed to load model from {model_path}")
            print("Make sure the model file exists and is valid.")
            sys.exit(1)

        # Set exploration noise to 0 for evaluation
        self.agent.set_exploration_noise(0.0)

    def run_episode(self, episode_num: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single evaluation episode.

        Args:
            episode_num: Episode number for logging
            verbose: Whether to print episode details

        Returns:
            Dictionary containing episode metrics
        """
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        successes = 0
        success_steps = []

        if verbose:
            print(f"\n--- Episode {episode_num} ---")

        start_time = time.time()

        while True:
            # Select action (deterministic for evaluation)
            action = self.agent.act(obs, training=False)

            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Update metrics
            episode_reward += reward
            episode_length += 1

            # Check for success
            obs_dict = self.env.env.unwrapped._get_obs()
            if self.env.get_success_rate(obs_dict):
                successes += 1
                success_steps.append(episode_length)

            # Optional: Print step info
            if verbose and self.args.verbose_steps:
                achieved_goal = obs_dict['achieved_goal']
                desired_goal = obs_dict['desired_goal']
                distance = np.linalg.norm(achieved_goal - desired_goal)
                print(f"  Step {episode_length:3d}: Reward={reward:6.3f}, Distance={distance:.4f}")

            # Render if requested
            if self.args.render:
                self.env.render()
                if self.args.render_delay > 0:
                    time.sleep(self.args.render_delay)

            obs = next_obs

            if done:
                break

        episode_time = time.time() - start_time
        success_rate = successes / episode_length if episode_length > 0 else 0.0

        if verbose:
            print(f"  Reward: {episode_reward:.3f}")
            print(f"  Length: {episode_length} steps")
            print(f"  Success rate: {success_rate:.3f} ({successes}/{episode_length} steps)")
            print(f"  Episode time: {episode_time:.2f}s")
            if success_steps:
                print(f"  Success steps: {success_steps[:5]}" + (" ..." if len(success_steps) > 5 else ""))

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success_rate': success_rate,
            'successes': successes,
            'episode_time': episode_time,
            'success_steps': success_steps
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation over multiple episodes.

        Returns:
            Dictionary of evaluation statistics
        """
        print(f"\nEvaluating TD3 agent for {self.args.num_episodes} episodes...")
        print("=" * 60)

        all_rewards = []
        all_lengths = []
        all_success_rates = []
        all_successes = []
        all_times = []
        all_success_steps = []

        for episode in range(1, self.args.num_episodes + 1):
            metrics = self.run_episode(episode, verbose=self.args.verbose_episodes)

            all_rewards.append(metrics['episode_reward'])
            all_lengths.append(metrics['episode_length'])
            all_success_rates.append(metrics['success_rate'])
            all_successes.append(metrics['successes'])
            all_times.append(metrics['episode_time'])
            all_success_steps.extend(metrics['success_steps'])

        # Calculate statistics
        rewards = np.array(all_rewards)
        lengths = np.array(all_lengths)
        success_rates = np.array(all_success_rates)
        successes = np.array(all_successes)
        times = np.array(all_times)

        stats = {
            # Reward statistics
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'reward_median': np.median(rewards),

            # Episode length statistics
            'length_mean': np.mean(lengths),
            'length_std': np.std(lengths),
            'length_min': np.min(lengths),
            'length_max': np.max(lengths),

            # Success statistics
            'success_rate_mean': np.mean(success_rates),
            'success_rate_std': np.std(success_rates),
            'total_successes': np.sum(successes),
            'total_steps': np.sum(lengths),
            'overall_success_rate': np.sum(successes) / np.sum(lengths) if np.sum(lengths) > 0 else 0.0,

            # Time statistics
            'time_per_episode': np.mean(times),
            'total_time': np.sum(times),

            # Success timing
            'success_steps_mean': np.mean(all_success_steps) if all_success_steps else 0.0,
            'num_successful_episodes': np.sum(success_rates > 0)
        }

        return stats

    def print_evaluation_results(self, stats: Dict[str, float]):
        """
        Print formatted evaluation results.

        Args:
            stats: Dictionary of evaluation statistics
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nEpisode Statistics ({self.args.num_episodes} episodes):")
        print(f"  Reward:        {stats['reward_mean']:8.3f} ± {stats['reward_std']:6.3f} "
              f"[{stats['reward_min']:6.3f}, {stats['reward_max']:6.3f}]")
        print(f"  Length:        {stats['length_mean']:8.1f} ± {stats['length_std']:6.1f} steps "
              f"[{stats['length_min']:3.0f}, {stats['length_max']:3.0f}]")
        print(f"  Time/episode:  {stats['time_per_episode']:8.2f}s")

        print(f"\nSuccess Statistics:")
        print(f"  Success rate:      {stats['success_rate_mean']:8.3f} ± {stats['success_rate_std']:6.3f}")
        print(f"  Overall success:   {stats['overall_success_rate']:8.3f} "
              f"({stats['total_successes']:.0f}/{stats['total_steps']:.0f} steps)")
        print(f"  Successful eps:    {stats['num_successful_episodes']:.0f}/{self.args.num_episodes} "
              f"({stats['num_successful_episodes']/self.args.num_episodes*100:.1f}%)")

        if stats['success_steps_mean'] > 0:
            print(f"  Avg success step:  {stats['success_steps_mean']:8.1f}")

        print(f"\nTotal evaluation time: {stats['total_time']:.2f}s")
        print("=" * 60)

    def run(self):
        """Run the evaluation."""
        try:
            stats = self.evaluate()
            self.print_evaluation_results(stats)

            # Save results if requested
            if self.args.save_results:
                self.save_results(stats)

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            self.env.close()

    def save_results(self, stats: Dict[str, float]):
        """
        Save evaluation results to file.

        Args:
            stats: Evaluation statistics
        """
        import json
        from datetime import datetime

        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.args.model_path) if self.args.model_path else str(Path(self.args.model_dir) / "td3_best.pth"),
            'num_episodes': self.args.num_episodes,
            'environment': 'FetchReachDense-v4',
            'algorithm': 'TD3',
            'statistics': stats
        }

        results_path = Path(self.args.results_dir) / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate TD3 agent on FetchReachDense-v4')

    # Model parameters
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model (default: model_dir/td3_best.pth)')
    parser.add_argument('--model-dir', type=str, default='models/td3',
                        help='Directory containing trained models')

    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--max-episode-steps', type=int, default=50,
                        help='Maximum steps per episode')

    # Visualization parameters
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--render-delay', type=float, default=0.02,
                        help='Delay between rendered frames (seconds)')

    # Output parameters
    parser.add_argument('--verbose-episodes', action='store_true',
                        help='Print detailed episode information')
    parser.add_argument('--verbose-steps', action='store_true',
                        help='Print step-by-step information')
    parser.add_argument('--save-results', action='store_true',
                        help='Save evaluation results to file')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save evaluation results')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("TD3 Evaluation for FetchReachDense-v4")
    print("=" * 40)
    print(f"Episodes: {args.num_episodes}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Render: {args.render}")
    print(f"Random seed: {args.seed}")
    print("=" * 40)

    # Create evaluator and run evaluation
    evaluator = TD3Evaluator(args)
    evaluator.run()

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Main entry point for Adroit Hand Door SAC training and evaluation.

This script provides a unified interface for training and evaluating SAC agents
on the Adroit Hand Door manipulation task.
"""

import argparse
import sys
from pathlib import Path

from train_sac_sb3 import train_sac
from evaluate_sb3 import evaluate_sac


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate SAC agent on Adroit Hand Door',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'],
                        default='train',
                        help='Mode: train agent, evaluate trained agent, or both')

    # Training arguments
    training_group = parser.add_argument_group('Training Arguments')
    training_group.add_argument('--episodes', type=int, default=5000,
                               help='Number of training episodes')
    training_group.add_argument('--eval_interval', type=int, default=100,
                               help='Episodes between evaluations during training')
    training_group.add_argument('--save_interval', type=int, default=200,
                               help='Episodes between checkpoints')
    training_group.add_argument('--restart', action='store_true',
                               help='Load and continue from best saved model')

    # Evaluation arguments
    evaluation_group = parser.add_argument_group('Evaluation Arguments')
    evaluation_group.add_argument('--eval_episodes', type=int, default=10,
                                 help='Number of episodes for evaluation')
    evaluation_group.add_argument('--model_path', type=str, default=None,
                                 help='Path to model checkpoint (if not specified, uses best model)')

    # Environment arguments
    env_group = parser.add_argument_group('Environment Arguments')
    env_group.add_argument('--reward_type', choices=['dense', 'sparse'],
                          default='dense',
                          help='Reward function type')
    env_group.add_argument('--max_episode_steps', type=int, default=200,
                          help='Maximum steps per episode')
    env_group.add_argument('--render', action='store_true',
                          help='Enable rendering')
    env_group.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')

    # Parse arguments
    args = parser.parse_args()

    print("=" * 80)
    print("ADROIT HAND DOOR - SAC REINFORCEMENT LEARNING")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Environment: AdroitHandDoor-v1 ({args.reward_type} rewards)")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Seed: {args.seed}")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    print("=" * 80)

    try:
        if args.mode in ['train', 'both']:
            print("\n🚀 Starting Training Phase...")
            train_sac(
                episodes=args.episodes,
                eval_interval=args.eval_interval,
                save_interval=args.save_interval,
                render=args.render,
                restart=args.restart,
                reward_type=args.reward_type,
                max_episode_steps=args.max_episode_steps
            )
            print("✅ Training completed successfully!")

        if args.mode in ['evaluate', 'both']:
            print("\n🎯 Starting Evaluation Phase...")
            evaluate_sac(
                episodes=args.eval_episodes,
                model_path=args.model_path,
                render=args.render,
                reward_type=args.reward_type,
                max_episode_steps=args.max_episode_steps,
                seed=args.seed
            )
            print("✅ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Training/Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n🎉 All operations completed successfully!")


if __name__ == "__main__":
    main()
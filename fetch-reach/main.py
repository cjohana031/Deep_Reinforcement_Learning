#!/usr/bin/env python3
"""
Main entry point for TD3 training and evaluation on FetchReachDense-v4.

This script provides a unified interface for training and evaluating TD3 agents
on the Shadow Dexterous Hand Reach task.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='TD3 for Shadow Dexterous Hand Reach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a new agent:
    python main.py train --episodes 2000

  Train with custom hyperparameters:
    python main.py train --episodes 1000 --lr-actor 1e-4 --batch-size 128

  Evaluate a trained agent:
    python main.py evaluate --num-episodes 100

  Evaluate with rendering:
    python main.py evaluate --num-episodes 10 --render

  Continue training from checkpoint:
    python main.py train --load-model models/td3/td3_episode_1000.pth
        """
    )

    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')

    # Training subparser
    train_parser = subparsers.add_parser('train', help='Train TD3 agent')

    # Environment parameters
    train_parser.add_argument('--max-episode-steps', type=int, default=50,
                            help='Maximum steps per episode')

    # Training parameters
    train_parser.add_argument('--episodes', type=int, default=2000,
                            help='Number of training episodes')
    train_parser.add_argument('--warmup-steps', type=int, default=10000,
                            help='Random exploration steps before training')

    # TD3 hyperparameters
    train_parser.add_argument('--lr-actor', type=float, default=3e-4,
                            help='Actor learning rate')
    train_parser.add_argument('--lr-critic', type=float, default=3e-4,
                            help='Critic learning rate')
    train_parser.add_argument('--gamma', type=float, default=0.99,
                            help='Discount factor')
    train_parser.add_argument('--tau', type=float, default=0.005,
                            help='Soft update parameter')
    train_parser.add_argument('--policy-noise', type=float, default=0.2,
                            help='Target policy smoothing noise')
    train_parser.add_argument('--noise-clip', type=float, default=0.5,
                            help='Target policy noise clip')
    train_parser.add_argument('--policy-freq', type=int, default=2,
                            help='Policy update frequency')
    train_parser.add_argument('--exploration-noise', type=float, default=0.1,
                            help='Exploration noise')

    # Network parameters
    train_parser.add_argument('--hidden-size', type=int, default=256,
                            help='Hidden layer size')
    train_parser.add_argument('--batch-size', type=int, default=256,
                            help='Batch size')
    train_parser.add_argument('--buffer-size', type=int, default=1000000,
                            help='Replay buffer size')

    # Logging and saving
    train_parser.add_argument('--log-dir', type=str, default='logs',
                            help='Directory for logs')
    train_parser.add_argument('--model-dir', type=str, default='models/td3',
                            help='Directory for saving models')
    train_parser.add_argument('--log-freq', type=int, default=100,
                            help='Logging frequency (episodes)')
    train_parser.add_argument('--eval-freq', type=int, default=100,
                            help='Evaluation frequency (episodes)')
    train_parser.add_argument('--save-freq', type=int, default=500,
                            help='Model saving frequency (episodes)')

    # Training options
    train_parser.add_argument('--load-model', type=str, default=None,
                            help='Path to pretrained model')
    train_parser.add_argument('--early-stop', action='store_true',
                            help='Enable early stopping')
    train_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')

    # Evaluation subparser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate TD3 agent')

    # Model parameters
    eval_parser.add_argument('--model-path', type=str, default=None,
                           help='Path to trained model (default: model_dir/td3_best.pth)')
    eval_parser.add_argument('--model-dir', type=str, default='models/td3',
                           help='Directory containing trained models')

    # Evaluation parameters
    eval_parser.add_argument('--num-episodes', type=int, default=100,
                           help='Number of evaluation episodes')
    eval_parser.add_argument('--max-episode-steps', type=int, default=50,
                           help='Maximum steps per episode')

    # Visualization parameters
    eval_parser.add_argument('--render', action='store_true',
                           help='Render environment during evaluation')
    eval_parser.add_argument('--render-delay', type=float, default=0.02,
                           help='Delay between rendered frames (seconds)')

    # Output parameters
    eval_parser.add_argument('--verbose-episodes', action='store_true',
                           help='Print detailed episode information')
    eval_parser.add_argument('--verbose-steps', action='store_true',
                           help='Print step-by-step information')
    eval_parser.add_argument('--save-results', action='store_true',
                           help='Save evaluation results to file')
    eval_parser.add_argument('--results-dir', type=str, default='results',
                           help='Directory to save evaluation results')

    # Other parameters
    eval_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting TD3 training...")
        from train_td3 import main as train_main
        # Override sys.argv to pass arguments to training script
        sys.argv = ['train_td3.py'] + sys.argv[2:]  # Remove 'main.py train'
        train_main()

    elif args.mode == 'evaluate':
        print("Starting TD3 evaluation...")
        from evaluate_td3 import main as eval_main
        # Override sys.argv to pass arguments to evaluation script
        sys.argv = ['evaluate_td3.py'] + sys.argv[2:]  # Remove 'main.py evaluate'
        eval_main()

    else:
        parser.print_help()
        print("\nPlease specify a mode: 'train' or 'evaluate'")
        sys.exit(1)


if __name__ == "__main__":
    main()
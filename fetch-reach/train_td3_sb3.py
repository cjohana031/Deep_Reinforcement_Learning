#!/usr/bin/env python3
"""
Training script for SB3 TD3 agent on FetchReachDense-v4 environment.

This script implements the training loop using Stable Baselines3 TD3
on the FetchReachDense-v4 environment.
"""

import argparse
import numpy as np
import torch
import os
import sys
import time
import signal
from pathlib import Path
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from agents.td3_sb3 import SB3TD3Agent, TrainingCallback
from environments.hand_reach_env import make_hand_reach_env
from utils.logger import Logger, ProgressTracker


class SB3TrainingManager:
    """Manages the SB3 TD3 training process with proper logging and checkpointing."""

    def __init__(self, args):
        """
        Initialize training manager.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.setup_environment()
        self.setup_logging()
        self.setup_agent()

        # Training state
        self.episode = 0
        self.best_success_rate = -1.0

        # For graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        self._interrupted = False

    def setup_environment(self):
        """Setup the environment."""
        print("Setting up environment...")
        try:
            # Create environment factory for SB3 that returns the raw Gym environment
            def env_factory():
                import gymnasium as gym
                import gymnasium_robotics
                from gymnasium.wrappers import FlattenObservation

                gym.register_envs(gymnasium_robotics)

                # Create the raw gymnasium environment
                env = gym.make(
                    "FetchReachDense-v4",
                    max_episode_steps=self.args.max_episode_steps,
                    render_mode=None
                )

                # Flatten the observation space (Dict -> Box)
                env = FlattenObservation(env)

                return env

            self.env_factory = env_factory

            # Create a single environment to get dimensions
            test_env = env_factory()
            self.state_dim = test_env.observation_space.shape[0]
            self.action_dim = test_env.action_space.shape[0]
            test_env.close()

            print(f"Environment created successfully!")
            print(f"State dimension: {self.state_dim}")
            print(f"Action dimension: {self.action_dim}")
        except Exception as e:
            print(f"Error creating environment: {e}")
            print("Make sure gymnasium-robotics is installed:")
            print("pip install gymnasium-robotics")
            sys.exit(1)

    def setup_logging(self):
        """Setup logging."""
        print("Setting up logging...")

        self.logger = Logger(
            log_dir=self.args.log_dir,
            experiment_name="td3_sb3_hand_reach",
            use_tensorboard=True
        )

        # Save hyperparameters
        hyperparams = {
            'environment': 'FetchReachDense-v4',
            'algorithm': 'TD3_SB3',
            'total_timesteps': self.args.total_timesteps,
            'max_episode_steps': self.args.max_episode_steps,
            'learning_rate': self.args.learning_rate,
            'gamma': self.args.gamma,
            'tau': self.args.tau,
            'target_policy_noise': self.args.target_policy_noise,
            'target_noise_clip': self.args.target_noise_clip,
            'policy_delay': self.args.policy_delay,
            'batch_size': self.args.batch_size,
            'buffer_size': self.args.buffer_size,
            'learning_starts': self.args.learning_starts
        }
        self.logger.save_hyperparameters(hyperparams)

    def setup_agent(self):
        """Setup the SB3 TD3 agent."""
        print("Setting up SB3 TD3 agent...")

        # Policy kwargs for network architecture
        policy_kwargs = dict(
            net_arch=dict(pi=[self.args.hidden_size, self.args.hidden_size],
                         qf=[self.args.hidden_size, self.args.hidden_size]),
            activation_fn=torch.nn.ReLU
        )

        self.agent = SB3TD3Agent(
            env=self.env_factory,
            model_dir=self.args.model_dir,
            learning_rate=self.args.learning_rate,
            buffer_size=self.args.buffer_size,
            learning_starts=self.args.learning_starts,
            batch_size=self.args.batch_size,
            tau=self.args.tau,
            gamma=self.args.gamma,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_delay=self.args.policy_delay,
            target_policy_noise=self.args.target_policy_noise,
            target_noise_clip=self.args.target_noise_clip,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=self.args.seed,
            tensorboard_log=str(Path(self.args.log_dir) / "td3_sb3_tensorboard"),
            n_envs=1
        )

        # Load pretrained model if specified
        if self.args.load_model:
            if self.agent.load(self.args.load_model):
                print(f"Loaded pretrained model from {self.args.load_model}")
            else:
                print(f"Could not load model from {self.args.load_model}, starting fresh")

    def train(self):
        """Main training loop using SB3."""
        self.logger.log_info("Starting SB3 TD3 training...")
        print(f"Training SB3 TD3 agent for {self.args.total_timesteps} timesteps")
        print(f"Environment: FetchReachDense-v4")

        # Create training callback
        callback = TrainingCallback(
            eval_freq=self.args.eval_freq,
            save_freq=self.args.save_freq,
            save_path=self.args.model_dir,
            custom_logger=self.logger
        )

        try:
            # Start training
            self.agent.train(
                total_timesteps=self.args.total_timesteps,
                callback=callback,
                tb_log_name="td3_sb3_training"
            )

            # Final save
            self.agent.save()
            self.logger.log_info("Training completed!")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.agent.save()
            self.logger.log_info("Model saved after interruption")
        except Exception as e:
            print(f"Training error: {e}")
            self.agent.save()
            raise

    def evaluate_agent(self, num_eval_episodes: int = 10):
        """
        Evaluate the trained agent with proper success rate calculation.

        Args:
            num_eval_episodes: Number of episodes to evaluate
        """
        print(f"\nEvaluating agent for {num_eval_episodes} episodes...")
        print("SUCCESS RATE ANALYSIS:")
        print("- Success threshold: distance < 0.05 units")
        print("- FetchReach rewards are ALWAYS negative (negative distance)")
        print("- Higher reward (closer to 0) = better performance")

        # Create evaluation environment WITH SAME REWARD SCALING AS TRAINING
        import gymnasium as gym
        import gymnasium_robotics
        from gymnasium.wrappers import FlattenObservation
        from agents.td3_sb3 import RewardScalingWrapper

        gym.register_envs(gymnasium_robotics)

        # Create raw gymnasium environment (same as training)
        eval_env = gym.make(
            "FetchReachDense-v4",
            max_episode_steps=self.args.max_episode_steps,
            render_mode=None
        )
        eval_env = FlattenObservation(eval_env)

        # Apply same reward scaling as used in training
        eval_env = RewardScalingWrapper(eval_env, reward_scale=10.0, reward_offset=0.0)

        eval_rewards = []
        eval_success_rates = []
        eval_lengths = []

        for episode in range(num_eval_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            successes = 0
            total_steps = 0
            frames = []
            distances = []  # Track distance progression
            initial_distance = None
            final_distance = None

            while True:
                # Use deterministic policy for evaluation
                action = self.agent.act(obs, training=False)
                next_obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1
                total_steps += 1

                # Check for success and track distance
                # Access the original dict observation before flattening
                obs_dict = eval_env.env.unwrapped._get_obs()
                achieved_goal = obs_dict['achieved_goal']
                desired_goal = obs_dict['desired_goal']
                distance = np.linalg.norm(achieved_goal - desired_goal)
                distances.append(distance)

                if initial_distance is None:
                    initial_distance = distance
                final_distance = distance

                # Success when distance < 0.05
                if distance < 0.05:
                    successes += 1

                obs = next_obs

                if done:
                    break

            # Save video of first episode
            if episode == 0 and frames:
                self._save_video(frames, f"evaluation_episode_{episode+1}.mp4")

            success_rate = successes / total_steps if total_steps > 0 else 0.0
            eval_rewards.append(episode_reward)
            eval_success_rates.append(success_rate)
            eval_lengths.append(episode_length)

            # Detailed analysis for first episode
            if episode == 0:
                min_distance = min(distances)
                max_distance = max(distances)
                print(f"\n=== DETAILED ANALYSIS (Episode 1) ===")
                print(f"Initial distance: {initial_distance:.4f}")
                print(f"Final distance:   {final_distance:.4f}")
                print(f"Min distance:     {min_distance:.4f}")
                print(f"Max distance:     {max_distance:.4f}")
                print(f"Distance change:  {final_distance - initial_distance:+.4f}")
                print(f"Success threshold: 0.05")
                print(f"DIAGNOSIS:")
                if final_distance > initial_distance:
                    print("  ❌ PROBLEM: Agent moving AWAY from target")
                    print("  ❌ This suggests wrong reward learning or exploration issues")
                elif min_distance < 0.05:
                    print("  ✅ Agent reached success threshold at some point")
                elif min_distance < initial_distance:
                    print("  🔶 Agent improved but didn't reach threshold")
                else:
                    print("  ❌ Agent didn't improve distance")
                print("=" * 45)

            print(f"Eval Episode {episode + 1}: Reward={episode_reward:.3f}, Success Rate={success_rate:.3f}, "
                  f"Initial Dist={initial_distance:.3f}, Final Dist={final_distance:.3f}")

        eval_env.close()

        # Print evaluation summary
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_success = np.mean(eval_success_rates)
        std_success = np.std(eval_success_rates)

        print(f"\nEvaluation Results ({num_eval_episodes} episodes):")
        print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        print(f"Mean Success Rate: {mean_success:.3f} ± {std_success:.3f}")
        print(f"Mean Episode Length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")

        # Overall diagnosis
        print(f"\n=== TRAINING DIAGNOSIS ===")
        if mean_success < 0.01:
            print("❌ CRITICAL: Success rate near 0% - agent failed to learn")
            if mean_reward < -100:
                print("❌ CRITICAL: Very negative rewards - agent moving away from targets")
                print("🔧 LIKELY CAUSES:")
                print("   1. Wrong reward calculation in training")
                print("   2. Poor exploration during early training")
                print("   3. Agent learned to maximize distance instead of minimize it")
                print("   4. Training callback success calculation is wrong (checks reward > 0)")
            else:
                print("🔶 Agent learned some improvement but didn't reach success threshold")
        elif mean_success < 0.5:
            print("🔶 PARTIAL: Agent sometimes succeeds but inconsistently")
        else:
            print("✅ SUCCESS: Agent learned to reach targets reliably")

        print(f"💡 RECOMMENDATION:")
        if mean_success < 0.01 and mean_reward < -100:
            print("   1. Fix success rate calculation in TrainingCallback (td3_sb3.py:206)")
            print("   2. Check reward calculation - ensure it rewards getting closer to target")
            print("   3. Consider retraining with corrected success monitoring")
        print("=" * 27)

        return {
            'eval_reward_mean': mean_reward,
            'eval_reward_std': std_reward,
            'eval_success_rate_mean': mean_success,
            'eval_success_rate_std': std_success
        }

    def _save_video(self, frames, filename):
        """Save frames as video."""
        try:
            import imageio
            # Create videos directory if it doesn't exist
            video_dir = Path("videos")
            video_dir.mkdir(exist_ok=True)
            video_path = video_dir / filename

            print(f"Saving video with {len(frames)} frames to {video_path}")
            imageio.mimsave(str(video_path), frames, fps=30)
            print(f"Video saved successfully: {video_path}")
        except ImportError:
            print("imageio not installed. Install with: pip install imageio[ffmpeg]")
        except Exception as e:
            print(f"Error saving video: {e}")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signal for graceful shutdown."""
        print("\nReceived interrupt signal. Saving model and exiting...")
        self._interrupted = True

        # Save current model
        if hasattr(self, 'agent'):
            self.agent.save()
            print("Model saved")

        # Close logger
        if hasattr(self, 'logger'):
            self.logger.close()

        sys.exit(0)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train SB3 TD3 agent on FetchReachDense-v4')

    # Environment parameters
    parser.add_argument('--max-episode-steps', type=int, default=400,
                        help='Maximum steps per episode')

    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--learning-starts', type=int, default=1000,
                        help='Number of steps before learning starts')

    # TD3 hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update parameter')
    parser.add_argument('--target-policy-noise', type=float, default=0.2,
                        help='Target policy smoothing noise')
    parser.add_argument('--target-noise-clip', type=float, default=0.5,
                        help='Target policy noise clip')
    parser.add_argument('--policy-delay', type=int, default=2,
                        help='Policy update frequency')

    # Network parameters
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size')

    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--model-dir', type=str, default='models/td3_sb3',
                        help='Directory for saving models')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency (timesteps)')
    parser.add_argument('--save-freq', type=int, default=50000,
                        help='Model saving frequency (timesteps)')

    # Training options
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained model instead of training')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of episodes for evaluation')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("SB3 TD3 Training for FetchReachDense-v4")
    print("=" * 50)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Random seed: {args.seed}")
    print("=" * 50)

    # Create training manager
    trainer = SB3TrainingManager(args)

    if args.evaluate:
        # Load model and evaluate
        if not args.load_model:
            args.load_model = str(Path(args.model_dir) / "td3_best")

        if trainer.agent.load(args.load_model):
            trainer.evaluate_agent(args.eval_episodes)
        else:
            print(f"Could not load model from {args.load_model}")
            sys.exit(1)
    else:
        # Train the agent
        trainer.train()

        # Evaluate after training
        trainer.evaluate_agent(args.eval_episodes)

    # Cleanup
    if hasattr(trainer, 'logger'):
        trainer.logger.close()

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
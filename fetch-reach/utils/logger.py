import os
import csv
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")


class Logger:
    """
    Comprehensive logging utility for training metrics and progress.
    Supports both CSV logging and TensorBoard visualization.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "experiment",
        use_tensorboard: bool = True,
        csv_filename: str = "training_log.csv"
    ):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard logging
            csv_filename: Name of CSV log file
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.csv_filename = csv_filename

        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup CSV logging
        self.csv_path = self.experiment_dir / csv_filename
        self.csv_initialized = False
        self.csv_fieldnames = None

        # Setup TensorBoard logging
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
            print(f"TensorBoard logs will be saved to: {self.experiment_dir / 'tensorboard'}")
        else:
            self.tb_writer = None

        # Setup Python logging
        self.python_logger = logging.getLogger(f"TD3_{experiment_name}")
        self.python_logger.setLevel(logging.INFO)

        # File handler
        log_file = self.experiment_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.python_logger.addHandler(file_handler)
        self.python_logger.addHandler(console_handler)

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.start_time = time.time()

        print(f"Logger initialized. Logs will be saved to: {self.experiment_dir}")

    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """
        Log metrics for a single episode.

        Args:
            episode: Episode number
            metrics: Dictionary of metrics to log
        """
        # Add episode number and timestamp
        metrics_with_meta = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': time.time() - self.start_time,
            **metrics
        }

        # Log to CSV
        self._log_to_csv(metrics_with_meta)

        # Log to TensorBoard
        if self.tb_writer:
            self._log_to_tensorboard(episode, metrics)

        # Track key metrics
        if 'episode_reward' in metrics:
            self.episode_rewards.append(metrics['episode_reward'])

        if 'episode_length' in metrics:
            self.episode_lengths.append(metrics['episode_length'])

        if 'success_rate' in metrics:
            self.success_rates.append(metrics['success_rate'])

    def log_training_step(self, step: int, losses: Dict[str, float]):
        """
        Log training step losses.

        Args:
            step: Training step number
            losses: Dictionary of loss values
        """
        if self.tb_writer:
            for loss_name, loss_value in losses.items():
                self.tb_writer.add_scalar(f"Training/{loss_name}", loss_value, step)

    def _log_to_csv(self, metrics: Dict[str, Any]):
        """Log metrics to CSV file."""
        # Initialize CSV file if first time
        if not self.csv_initialized:
            self.csv_fieldnames = list(metrics.keys())
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
                writer.writeheader()
            self.csv_initialized = True

        # Write metrics row
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
            # Only write fields that exist in the original fieldnames
            filtered_metrics = {k: v for k, v in metrics.items() if k in self.csv_fieldnames}
            writer.writerow(filtered_metrics)

    def _log_to_tensorboard(self, episode: int, metrics: Dict[str, Any]):
        """Log metrics to TensorBoard."""
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.tb_writer.add_scalar(f"Episode/{metric_name}", metric_value, episode)

    def log_info(self, message: str):
        """Log informational message."""
        self.python_logger.info(message)

    def log_warning(self, message: str):
        """Log warning message."""
        self.python_logger.warning(message)

    def log_error(self, message: str):
        """Log error message."""
        self.python_logger.error(message)

    def get_stats(self) -> Dict[str, float]:
        """
        Get training statistics.

        Returns:
            Dictionary of training statistics
        """
        stats = {}

        if self.episode_rewards:
            rewards = np.array(self.episode_rewards)
            stats.update({
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'last_100_mean_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            })

        if self.episode_lengths:
            lengths = np.array(self.episode_lengths)
            stats.update({
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths)
            })

        if self.success_rates:
            success_rates = np.array(self.success_rates)
            stats.update({
                'mean_success_rate': np.mean(success_rates),
                'last_100_success_rate': np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
            })

        stats['total_episodes'] = len(self.episode_rewards)
        stats['elapsed_time'] = time.time() - self.start_time

        return stats

    def save_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Save hyperparameters to file.

        Args:
            hyperparams: Dictionary of hyperparameters
        """
        import json

        hyperparams_path = self.experiment_dir / "hyperparameters.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)

        self.log_info(f"Hyperparameters saved to {hyperparams_path}")

    def close(self):
        """Close logger and clean up resources."""
        if self.tb_writer:
            self.tb_writer.close()

        # Close handlers to release file locks
        for handler in self.python_logger.handlers[:]:
            handler.close()
            self.python_logger.removeHandler(handler)

    def print_stats(self):
        """Print current training statistics."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("TRAINING STATISTICS")
        print("="*60)

        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:8.3f}")
            else:
                print(f"{key:25s}: {value}")

        print("="*60)


class ProgressTracker:
    """Progress tracking utility with tqdm support."""

    def __init__(self, total_episodes: int, log_interval: int = 100, use_tqdm: bool = True):
        """
        Initialize progress tracker.

        Args:
            total_episodes: Total number of episodes to train
            log_interval: Interval for progress updates
            use_tqdm: Whether to use tqdm progress bar
        """
        self.total_episodes = total_episodes
        self.log_interval = log_interval
        self.start_time = time.time()
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE

        # Initialize tqdm progress bar if available
        if self.use_tqdm:
            self.pbar = tqdm(
                total=total_episodes,
                desc="Training",
                unit="ep",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        else:
            self.pbar = None

    def update(self, episode: int, reward: float, success_rate: float = None, extra_info: dict = None):
        """
        Update progress and print if needed.

        Args:
            episode: Current episode number
            reward: Episode reward
            success_rate: Success rate for the episode
            extra_info: Additional information to display
        """
        if self.pbar:
            # Update tqdm progress bar
            postfix = {
                'reward': f'{reward:6.2f}',
            }

            if success_rate is not None:
                postfix['success'] = f'{success_rate:.3f}'

            if extra_info:
                postfix.update(extra_info)

            self.pbar.set_postfix(postfix)
            self.pbar.update(1)
        else:
            # Fall back to simple print-based progress
            if episode % self.log_interval == 0 or episode == 1:
                elapsed = time.time() - self.start_time
                progress = episode / self.total_episodes * 100
                episodes_per_sec = episode / elapsed if elapsed > 0 else 0

                status_msg = (f"Episode {episode:6d}/{self.total_episodes} ({progress:5.1f}%) | "
                             f"Reward: {reward:8.3f} | "
                             f"Speed: {episodes_per_sec:.1f} eps/s | "
                             f"Time: {elapsed/60:.1f}min")

                if success_rate is not None:
                    status_msg += f" | Success: {success_rate:.3f}"

                print(status_msg)

    def close(self):
        """Close the progress tracker."""
        if self.pbar:
            self.pbar.close()


if __name__ == "__main__":
    # Test the logger
    logger = Logger(
        log_dir="test_logs",
        experiment_name="td3_hand_reach",
        use_tensorboard=True
    )

    # Test episode logging
    for episode in range(1, 6):
        metrics = {
            'episode_reward': np.random.uniform(-100, 100),
            'episode_length': np.random.randint(10, 100),
            'success_rate': np.random.uniform(0, 1),
            'actor_loss': np.random.uniform(0, 1),
            'critic_loss': np.random.uniform(0, 1)
        }
        logger.log_episode(episode, metrics)

    # Test logging messages
    logger.log_info("Test info message")
    logger.log_warning("Test warning message")

    # Print stats
    logger.print_stats()

    # Save hyperparameters
    hyperparams = {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'batch_size': 256,
        'gamma': 0.99
    }
    logger.save_hyperparameters(hyperparams)

    logger.close()
    print("Logger test completed!")
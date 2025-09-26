import json
import os
import numpy as np
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir='logs', tensorboard_dir='runs'):
        self.log_dir = log_dir
        self.history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'timestamps': [],
            'actor_losses': [],
            'critic1_losses': [],
            'critic2_losses': [],
            'entropy_losses': [],
            'alphas': [],
            'eval_rewards': [],
            'success_rates': []
        }

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize TensorBoard writer
        run_name = f"SAC_ShadowHand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tensorboard_dir = os.path.join(tensorboard_dir, run_name)
        self.writer = SummaryWriter(self.tensorboard_dir)

        # Initialize CSV logging
        self.csv_path = os.path.join(log_dir, 'training_log.csv')
        self._init_csv()

    def log_episode(self, episode, reward, steps, losses=None, alpha=None, success_rate=None):
        timestamp = datetime.now().isoformat()
        self.history['episodes'].append(episode)
        self.history['rewards'].append(reward)
        self.history['steps'].append(steps)
        self.history['timestamps'].append(timestamp)

        if success_rate is not None:
            self.history['success_rates'].append(success_rate)

        # Log to CSV
        self._log_to_csv(episode, reward, steps, timestamp, losses, alpha, success_rate)

        # Log to TensorBoard
        self.writer.add_scalar('Episode/Reward', reward, episode)
        self.writer.add_scalar('Episode/Steps', steps, episode)

        if success_rate is not None:
            self.writer.add_scalar('Episode/Success_Rate', success_rate, episode)

        # Log moving averages
        if len(self.history['rewards']) >= 10:
            avg_10 = np.mean(self.history['rewards'][-10:])
            self.writer.add_scalar('Average/Reward_10ep', avg_10, episode)

        if len(self.history['rewards']) >= 100:
            avg_100 = np.mean(self.history['rewards'][-100:])
            self.writer.add_scalar('Average/Reward_100ep', avg_100, episode)

        # Log success rate averages for goal-conditioned tasks
        if len(self.history['success_rates']) >= 10:
            success_avg_10 = np.mean(self.history['success_rates'][-10:])
            self.writer.add_scalar('Average/Success_Rate_10ep', success_avg_10, episode)

        if len(self.history['success_rates']) >= 100:
            success_avg_100 = np.mean(self.history['success_rates'][-100:])
            self.writer.add_scalar('Average/Success_Rate_100ep', success_avg_100, episode)

    def log_training_metrics(self, episode, losses=None, alpha=None, eval_reward=None):
        """Log SAC training metrics to TensorBoard"""
        if losses:
            if 'actor_loss' in losses:
                self.writer.add_scalar('Training/Actor_Loss', losses['actor_loss'], episode)
                self.history['actor_losses'].append(losses['actor_loss'])

            if 'critic1_loss' in losses:
                self.writer.add_scalar('Training/Critic1_Loss', losses['critic1_loss'], episode)
                self.history['critic1_losses'].append(losses['critic1_loss'])

            if 'critic2_loss' in losses:
                self.writer.add_scalar('Training/Critic2_Loss', losses['critic2_loss'], episode)
                self.history['critic2_losses'].append(losses['critic2_loss'])

            if 'entropy_loss' in losses:
                self.writer.add_scalar('Training/Entropy_Loss', losses['entropy_loss'], episode)
                self.history['entropy_losses'].append(losses['entropy_loss'])

        if alpha is not None:
            self.writer.add_scalar('Training/Alpha', alpha, episode)
            self.history['alphas'].append(alpha)

        if eval_reward is not None:
            self.writer.add_scalar('Evaluation/Reward', eval_reward, episode)
            self.history['eval_rewards'].append(eval_reward)

    def save_history(self, filename=None):
        if filename is None:
            filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Training history saved to {filepath}")

    def get_history(self):
        return self.history

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, **self.history)
        print(f"Training history saved to {filepath}")

    def get_statistics(self, last_n=100):
        if len(self.history['rewards']) < last_n:
            last_n = len(self.history['rewards'])

        if last_n == 0:
            return {}

        recent_rewards = self.history['rewards'][-last_n:]
        recent_steps = self.history['steps'][-last_n:]
        recent_success = self.history['success_rates'][-last_n:] if self.history['success_rates'] else [0]

        stats = {
            'mean_reward': sum(recent_rewards) / len(recent_rewards),
            'max_reward': max(recent_rewards),
            'min_reward': min(recent_rewards),
            'mean_steps': sum(recent_steps) / len(recent_steps),
            'mean_success_rate': sum(recent_success) / len(recent_success) if recent_success else 0,
            'total_episodes': len(self.history['episodes'])
        }

        return stats

    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'episode', 'reward', 'steps', 'timestamp', 'actor_loss',
                'critic1_loss', 'critic2_loss', 'entropy_loss', 'alpha',
                'success_rate', 'eval_reward'
            ])

    def _log_to_csv(self, episode, reward, steps, timestamp, losses=None, alpha=None, success_rate=None):
        """Append a row to the CSV file"""
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Extract loss values
            actor_loss = losses.get('actor_loss') if losses else None
            critic1_loss = losses.get('critic1_loss') if losses else None
            critic2_loss = losses.get('critic2_loss') if losses else None
            entropy_loss = losses.get('entropy_loss') if losses else None

            writer.writerow([
                episode, reward, steps, timestamp, actor_loss,
                critic1_loss, critic2_loss, entropy_loss, alpha,
                success_rate, None  # eval_reward placeholder
            ])

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
        print(f"TensorBoard logs saved to {self.tensorboard_dir}")
        print(f"CSV training log saved to {self.csv_path}")
        return self.tensorboard_dir
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch


class SB3SACAgent:
    """Wrapper around Stable Baselines3 SAC implementation"""

    def __init__(
        self,
        env,
        model_dir='models/sac',
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        policy_kwargs=None,
        verbose=1,
        seed=42,
        tensorboard_log=None,
        n_envs=1,
        vec_env_cls=None
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if n_envs > 1:
            # Choose vectorized environment class
            if vec_env_cls is None:
                vec_env_cls = SubprocVecEnv if n_envs >= 4 else DummyVecEnv

            # If env is a callable (factory function)
            if callable(env):
                # Works for gym.make or custom make_env factories
                self.env = make_vec_env(env, n_envs=n_envs, vec_env_cls=vec_env_cls, seed=seed)
            else:
                # If user passed an env instance, wrap it in a DummyVecEnv
                # NOTE: we cannot replicate an env instance safely across subprocesses
                if vec_env_cls == SubprocVecEnv:
                    raise ValueError(
                        "❌ Cannot use SubprocVecEnv with an already-created environment instance. "
                        "Please pass a factory (e.g. lambda: MyEnv())."
                    )
                self.env = DummyVecEnv([lambda: env])
        else:
            # Single environment case
            if callable(env):
                # Create single environment from factory
                self.env = env()
            else:
                # Use the environment instance directly
                self.env = env

        # Set default policy kwargs for better performance
        if policy_kwargs is None:
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
                activation_fn=torch.nn.ReLU
            )

        # Create SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tensorboard_log,
            device='auto'
        )

        # Setup logger if tensorboard_log is provided
        if tensorboard_log:
            new_logger = configure(tensorboard_log, ["stdout", "csv", "tensorboard"])
            self.model.set_logger(new_logger)

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        action, _ = self.model.predict(state, deterministic=not training)
        return action

    def train(self, total_timesteps: int, callback=None, **kwargs):
        """Train the SAC agent"""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )

    def save(self, episode=None):
        """Save model checkpoint"""
        filename = f"sac_episode_{episode}" if episode else "sac_final"
        filepath = self.model_dir / filename
        self.model.save(str(filepath))
        print(f"Model saved to {filepath}.zip")

        # Also save as best model
        if episode is None:
            best_path = self.model_dir / "sac_best"
            self.model.save(str(best_path))

    def load(self, filepath=None):
        """Load model checkpoint"""
        if filepath is None:
            filepath = self.model_dir / "sac_best"
            if not Path(f"{filepath}.zip").exists():
                filepath = self.model_dir / "sac_final"

        if isinstance(filepath, Path):
            filepath = str(filepath)

        if not Path(f"{filepath}.zip").exists():
            print(f"No model found at {filepath}.zip")
            return False

        self.model = SAC.load(filepath, env=self.env)
        print(f"Model loaded from {filepath}.zip")
        return True

    def get_policy_state(self):
        """Get current policy parameters for compatibility"""
        try:
            if isinstance(self.model.ent_coef, float):
                alpha = self.model.ent_coef
            elif hasattr(self.model, 'ent_coef_tensor') and self.model.ent_coef_tensor is not None:
                alpha = self.model.ent_coef_tensor.item()
            else:
                alpha = 0.1  # Default fallback
        except:
            alpha = 0.1  # Fallback if any error occurs

        return {
            'alpha': alpha,
            'actor_loss': 0.0,  # SB3 doesn't expose individual losses during training
            'critic_loss': 0.0
        }


class TrainingCallback(BaseCallback):
    """
    Custom callback for SAC training to track metrics and save checkpoints
    """
    def __init__(self,
                 eval_freq: int = 1000,
                 save_freq: int = 10000,
                 eval_episodes: int = 5,
                 save_path: str = None,
                 verbose: int = 1,
                 custom_logger=None):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.eval_episodes = eval_episodes
        self.save_path = save_path
        self.custom_logger = custom_logger

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -np.inf

        # Buffered logging for CSV
        self.csv_buffer_size = 1000  # Buffer size in timesteps
        self.buffer_data = []
        self.last_flush_timestep = 0

    def _on_step(self) -> bool:
        """Called at each step"""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        # Get episode info
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info['r'])
            self.episode_lengths.append(ep_info['l'])
            self.episode_count += 1

            # Buffer data for CSV logging
            if self.custom_logger:
                success_rate_episode = 1.0 if ep_info['r'] > 0 else 0.0
                # Get alpha value safely
                try:
                    if isinstance(self.model.ent_coef, float):
                        alpha_value = self.model.ent_coef
                    elif hasattr(self.model, 'ent_coef_tensor') and self.model.ent_coef_tensor is not None:
                        alpha_value = self.model.ent_coef_tensor.item()
                    else:
                        alpha_value = 0.1  # Default fallback value
                except:
                    alpha_value = 0.1  # Fallback if any error occurs

                self.buffer_data.append({
                    'episode': self.episode_count,
                    'reward': ep_info['r'],
                    'steps': ep_info['l'],
                    'success_rate': success_rate_episode,
                    'alpha': alpha_value,
                    'timesteps': self.num_timesteps
                })

                if self.episode_count <= 5:  # Debug first 5 episodes
                    print(f"[DEBUG] Buffered episode {self.episode_count}, buffer size: {len(self.buffer_data)}, timesteps: {self.num_timesteps}")

                # Log to TensorBoard immediately
                self.custom_logger.writer.add_scalar('Episode/Reward', ep_info['r'], self.episode_count)
                self.custom_logger.writer.add_scalar('Episode/Steps', ep_info['l'], self.episode_count)
                self.custom_logger.writer.add_scalar('Episode/Success_Rate', success_rate_episode, self.episode_count)
                self.custom_logger.writer.add_scalar('Training/Alpha', alpha_value, self.episode_count)

                # Calculate and log moving averages
                if len(self.episode_rewards) >= 10:
                    avg_10 = np.mean(self.episode_rewards[-10:])
                    self.custom_logger.writer.add_scalar('Average/Reward_10ep', avg_10, self.episode_count)

                if len(self.episode_rewards) >= 100:
                    avg_100 = np.mean(self.episode_rewards[-100:])
                    self.custom_logger.writer.add_scalar('Average/Reward_100ep', avg_100, self.episode_count)

                # Flush buffer to CSV every buffer_size timesteps
                timesteps_since_flush = self.num_timesteps - self.last_flush_timestep
                if timesteps_since_flush >= self.csv_buffer_size:
                    print(f"[DEBUG] Flushing CSV at timestep {self.num_timesteps} (buffer has {len(self.buffer_data)} episodes)")
                    self._flush_buffer_to_csv()
                    self.last_flush_timestep = self.num_timesteps

            # Print episode info
            if self.episode_count % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                success_rate = np.sum(np.array(self.episode_rewards[-100:]) > 0) / min(len(self.episode_rewards), 100)

                print(f"[TRAINING] Episode {self.episode_count:6d} | "
                      f"Reward: {ep_info['r']:8.1f} | "
                      f"Avg100: {mean_reward:8.2f} | "
                      f"Success: {1.0 if ep_info['r'] > 0 else 0.0:.2f} | "
                      f"Avg100Success: {success_rate:.3f} | "
                      f"Steps: {self.num_timesteps:8d}")

                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.save_path:
                        self.model.save(f"{self.save_path}/sac_best")

            # Save checkpoint
            if self.save_freq > 0 and self.episode_count % (self.save_freq // 1000) == 0:
                if self.save_path:
                    self.model.save(f"{self.save_path}/sac_episode_{self.episode_count}")

        return True

    def _flush_buffer_to_csv(self):
        """Flush buffered data to CSV"""
        if not self.custom_logger or not self.buffer_data:
            return

        import csv
        from datetime import datetime

        # Add data to CSV
        buffer_len = len(self.buffer_data)
        for data in self.buffer_data:
            timestamp = datetime.now().isoformat()
            with open(self.custom_logger.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    data['episode'],
                    data['reward'],
                    data['steps'],
                    timestamp,
                    None,  # actor_loss (not available in SB3)
                    None,  # critic1_loss
                    None,  # critic2_loss
                    None,  # entropy_loss
                    data['alpha'],
                    data['success_rate'],
                    None   # eval_reward
                ])

        # Clear buffer
        self.buffer_data = []
        print(f"[CSV] Flushed {buffer_len} episodes to {self.custom_logger.csv_path}")

    def _on_training_end(self) -> None:
        """Called when training ends - flush remaining buffer"""
        if self.custom_logger and self.buffer_data:
            self._flush_buffer_to_csv()
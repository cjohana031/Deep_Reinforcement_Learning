import numpy as np
from pathlib import Path
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation
import torch
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


class RewardScalingWrapper(gym.Wrapper):
    """Reward scaling wrapper for FetchReach environments with negative distance-based rewards"""

    def __init__(self, env, reward_scale=10.0, reward_offset=0.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Scale negative distance rewards to improve TD3 critic learning
        # Original: reward = -distance (always negative, approaches 0 when close)
        # Scaled: reward = (reward + offset) * scale
        scaled_reward = (reward + self.reward_offset) * self.reward_scale
        return obs, scaled_reward, terminated, truncated, info


class SB3TD3Agent:
    """Wrapper around Stable Baselines3 TD3 implementation with FetchReach optimizations"""

    def __init__(
        self,
        env,
        model_dir='models/td3',
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs=None,
        verbose=1,
        seed=42,
        tensorboard_log=None,
        n_envs=1,
        vec_env_cls=None,
        reward_scale=10.0,
        reward_offset=0.0,
        use_reward_scaling=True
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Store reward scaling parameters for success detection
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset
        self.use_reward_scaling = use_reward_scaling

        if n_envs > 1:
            # Choose vectorized environment class
            if vec_env_cls is None:
                vec_env_cls = SubprocVecEnv if n_envs >= 4 else DummyVecEnv

            # If env is a callable (factory function)
            if callable(env):
                # Create wrapper factory for reward scaling
                def wrapped_env_factory():
                    base_env = env()
                    if use_reward_scaling:
                        base_env = RewardScalingWrapper(base_env, reward_scale, reward_offset)
                    return base_env

                self.env = make_vec_env(wrapped_env_factory, n_envs=n_envs, vec_env_cls=vec_env_cls, seed=seed)
            else:
                # If user passed an env instance, wrap it in a DummyVecEnv
                # NOTE: we cannot replicate an env instance safely across subprocesses
                if vec_env_cls == SubprocVecEnv:
                    raise ValueError(
                        "❌ Cannot use SubprocVecEnv with an already-created environment instance. "
                        "Please pass a factory (e.g. lambda: MyEnv())."
                    )
                # Apply reward scaling to single env instance
                if use_reward_scaling:
                    env = RewardScalingWrapper(env, reward_scale, reward_offset)
                self.env = DummyVecEnv([lambda: env])
        else:
            # Single environment case
            if callable(env):
                # Create single environment from factory
                self.env = env()
                if use_reward_scaling:
                    self.env = RewardScalingWrapper(self.env, reward_scale, reward_offset)
            else:
                # Use the environment instance directly
                self.env = env
                if use_reward_scaling:
                    self.env = RewardScalingWrapper(self.env, reward_scale, reward_offset)

        # Set default policy kwargs optimized for FetchReach
        if policy_kwargs is None:
            policy_kwargs = dict(
                net_arch=dict(pi=[400, 300], qf=[400, 300]),
                activation_fn=torch.nn.ReLU,
                # Add layer normalization for better stability with scaled rewards
                normalize_images=False,
                optimizer_kwargs=dict(eps=1e-5)
            )

        # Create TD3 model with optimizations for FetchReach
        self.model = TD3(
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
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tensorboard_log,
            device='auto'
        )

        print(f"🚀 TD3 Agent initialized with reward scaling: {use_reward_scaling}")
        if use_reward_scaling:
            print(f"   Reward scale: {reward_scale}, offset: {reward_offset}")
            print(f"   Original reward range: [-1, 0] → Scaled: [{(0 + reward_offset) * reward_scale:.1f}, {(-1 + reward_offset) * reward_scale:.1f}]")

        # Setup logger if tensorboard_log is provided
        if tensorboard_log:
            new_logger = configure(tensorboard_log, ["stdout", "csv", "tensorboard"])
            self.model.set_logger(new_logger)

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy"""
        action, _ = self.model.predict(state, deterministic=not training)
        return action


    def train(self, total_timesteps: int, callback=None, progress_bar: bool = True, **kwargs):
        """Train the TD3 agent"""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
            **kwargs
        )

    def save(self, episode=None):
        """Save model checkpoint"""
        filename = f"td3_episode_{episode}" if episode else "td3_final"
        filepath = self.model_dir / filename
        self.model.save(str(filepath))
        print(f"Model saved to {filepath}.zip")

        # Also save as best model
        if episode is None:
            best_path = self.model_dir / "td3_best"
            self.model.save(str(best_path))

    def load(self, filepath=None):
        """Load model checkpoint"""
        if filepath is None:
            filepath = self.model_dir / "td3_best"
            if not Path(f"{filepath}.zip").exists():
                filepath = self.model_dir / "td3_final"

        if isinstance(filepath, Path):
            filepath = str(filepath)

        if not Path(f"{filepath}.zip").exists():
            print(f"No model found at {filepath}.zip")
            return False

        self.model = TD3.load(filepath, env=self.env)
        print(f"Model loaded from {filepath}.zip")
        return True

    def get_policy_state(self):
        """Get current policy parameters for compatibility"""
        return {
            'policy_delay': getattr(self.model, 'policy_delay', 2),
            'actor_loss': 0.0,  # SB3 doesn't expose individual losses during training
            'critic_loss': 0.0
        }


class TrainingCallback(BaseCallback):
    """
    Custom callback for TD3 training to track metrics and save checkpoints
    Optimized for FetchReach environment with negative distance-based rewards
    """
    def __init__(self,
                 eval_freq: int = 1000,
                 save_freq: int = 10000,
                 eval_episodes: int = 5,
                 save_path: str = None,
                 verbose: int = 1,
                 custom_logger=None,
                 reward_scale: float = 10.0,
                 reward_offset: float = 0.0,
                 use_reward_scaling: bool = True,
                 success_threshold: float = -0.05):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.eval_episodes = eval_episodes
        self.save_path = save_path
        self.custom_logger = custom_logger

        # Reward scaling parameters for proper success detection
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset
        self.use_reward_scaling = use_reward_scaling
        self.success_threshold = success_threshold

        # Calculate scaled success threshold
        if use_reward_scaling:
            self.scaled_success_threshold = (success_threshold + reward_offset) * reward_scale
        else:
            self.scaled_success_threshold = success_threshold

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_mean_reward = -np.inf

        # Buffered logging for CSV
        self.csv_buffer_size = 1000  # Buffer size in timesteps
        self.buffer_data = []
        self.last_flush_timestep = 0

        # Evaluation tracking
        self.last_eval_timestep = 0

        print(f"📋 Callback initialized with success threshold: {success_threshold} (scaled: {self.scaled_success_threshold:.2f})")

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
                # IMPROVED: Use scaled success threshold that accounts for reward scaling
                # For FetchReach: original reward = -distance, success when distance < threshold
                # With scaling: scaled_reward = (reward + offset) * scale
                success_rate_episode = 1.0 if ep_info['r'] > self.scaled_success_threshold else 0.0
                policy_delay = getattr(self.model, 'policy_delay', 2)

                # Convert back to original reward scale for logging if needed
                if self.use_reward_scaling:
                    original_reward = ep_info['r'] / self.reward_scale - self.reward_offset
                else:
                    original_reward = ep_info['r']

                self.buffer_data.append({
                    'episode': self.episode_count,
                    'reward': ep_info['r'],  # Scaled reward
                    'original_reward': original_reward,  # Original unscaled reward
                    'steps': ep_info['l'],
                    'success_rate': success_rate_episode,
                    'policy_delay': policy_delay,
                    'timesteps': self.num_timesteps
                })

                if self.episode_count <= 5:  # Debug first 5 episodes
                    print(f"[DEBUG] Buffered episode {self.episode_count}, buffer size: {len(self.buffer_data)}, timesteps: {self.num_timesteps}")

                # Log to TensorBoard immediately (both scaled and original rewards)
                self.custom_logger.tb_writer.add_scalar('Episode/Reward_Scaled', ep_info['r'], self.episode_count)
                self.custom_logger.tb_writer.add_scalar('Episode/Reward_Original', original_reward, self.episode_count)
                self.custom_logger.tb_writer.add_scalar('Episode/Steps', ep_info['l'], self.episode_count)
                self.custom_logger.tb_writer.add_scalar('Episode/Success_Rate', success_rate_episode, self.episode_count)
                self.custom_logger.tb_writer.add_scalar('Training/Policy_Delay', policy_delay, self.episode_count)

                # Calculate and log moving averages
                if len(self.episode_rewards) >= 10:
                    avg_10 = np.mean(self.episode_rewards[-10:])
                    self.custom_logger.tb_writer.add_scalar('Average/Reward_10ep', avg_10, self.episode_count)

                if len(self.episode_rewards) >= 100:
                    avg_100 = np.mean(self.episode_rewards[-100:])
                    self.custom_logger.tb_writer.add_scalar('Average/Reward_100ep', avg_100, self.episode_count)

                # Flush buffer to CSV every buffer_size timesteps
                timesteps_since_flush = self.num_timesteps - self.last_flush_timestep
                if timesteps_since_flush >= self.csv_buffer_size:
                    print(f"[DEBUG] Flushing CSV at timestep {self.num_timesteps} (buffer has {len(self.buffer_data)} episodes)")
                    self._flush_buffer_to_csv()
                    self.last_flush_timestep = self.num_timesteps

            # Print episode info with improved success calculation
            if self.episode_count % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                # IMPROVED: Use proper scaled threshold for success rate calculation
                recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                success_rate = np.sum(np.array(recent_rewards) > self.scaled_success_threshold) / len(recent_rewards)

                # Calculate original unscaled reward for display
                if self.use_reward_scaling:
                    display_reward = ep_info['r'] / self.reward_scale - self.reward_offset
                    display_mean = mean_reward / self.reward_scale - self.reward_offset
                else:
                    display_reward = ep_info['r']
                    display_mean = mean_reward

                print(f"[TRAINING] Episode {self.episode_count:6d} | "
                      f"Reward: {display_reward:8.3f} | "
                      f"Avg100: {display_mean:8.3f} | "
                      f"Success: {1.0 if ep_info['r'] > self.scaled_success_threshold else 0.0:.2f} | "
                      f"Avg100Success: {success_rate:.3f} | "
                      f"Steps: {self.num_timesteps:8d}")

                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.save_path:
                        self.model.save(f"{self.save_path}/td3_best")

        # Save checkpoint (moved outside the episode % 10 block)
        if self.save_freq > 0 and self.num_timesteps % self.save_freq == 0:
            if self.save_path:
                self.model.save(f"{self.save_path}/td3_timestep_{self.num_timesteps}")

        # Evaluate model at eval_freq intervals
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            eval_rewards = self._evaluate_agent()
            if eval_rewards is not None and self.custom_logger:
                mean_eval_reward = np.mean(eval_rewards)
                self.custom_logger.tb_writer.add_scalar('Evaluation/Mean_Reward', mean_eval_reward, self.num_timesteps)
                print(f"[EVAL] Timestep {self.num_timesteps} | Mean Eval Reward: {mean_eval_reward:.3f}")

        return True

    def _evaluate_agent(self):
        """Evaluate the agent on a separate environment"""
        try:
            # Create evaluation environment
            import gymnasium as gym
            import gymnasium_robotics
            from gymnasium.wrappers import FlattenObservation

            gym.register_envs(gymnasium_robotics)
            eval_env = gym.make("FetchReachDense-v4", max_episode_steps=50, render_mode=None)
            eval_env = FlattenObservation(eval_env)

            eval_rewards = []
            for _ in range(self.eval_episodes):
                obs, _ = eval_env.reset()
                episode_reward = 0.0

                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break

                eval_rewards.append(episode_reward)

            eval_env.close()
            return eval_rewards

        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return None

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
                    data['reward'],  # Scaled reward
                    data.get('original_reward', data['reward']),  # Original reward
                    data['steps'],
                    timestamp,
                    None,  # actor_loss (not available in SB3)
                    None,  # critic1_loss
                    None,  # critic2_loss
                    None,  # entropy_loss
                    data['policy_delay'],
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
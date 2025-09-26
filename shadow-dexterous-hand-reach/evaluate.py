import numpy as np
import argparse
from pathlib import Path
import time

from environments.shadow_hand_reach_env import ShadowHandReachEnvironment
from stable_baselines3 import SAC


def evaluate_agent(model_path, episodes=100, render=False, max_episode_steps=50, seed=42):
    """
    Evaluate a trained SAC agent on Shadow Dexterous Hand Reach

    Args:
        model_path: Path to the saved model
        episodes: Number of evaluation episodes
        render: Whether to render the environment
        max_episode_steps: Maximum steps per episode
        seed: Random seed
    """
    print(f"🔍 Evaluating Shadow Hand Reach agent...")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Rendering: {render}")
    print("-" * 60)

    # Create environment
    env = ShadowHandReachEnvironment(
        render_mode="human" if render else None,
        seed=seed,
        max_episode_steps=max_episode_steps,
        reward_type="dense"
    )

    # Load the trained model
    try:
        model = SAC.load(model_path, env=env)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        return

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    final_distances = []

    print("\nStarting evaluation...")
    start_time = time.time()

    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        success = False
        final_distance = float('inf')

        while True:
            # Use deterministic policy for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                success = info.get('is_success', False)

                # Calculate final distance between achieved and desired goals
                try:
                    if hasattr(env, 'env') and hasattr(env.env, 'env'):
                        gym_env = env.env.env
                        if hasattr(gym_env, 'unwrapped'):
                            unwrapped = gym_env.unwrapped
                            # Get final observation
                            final_obs = unwrapped._get_obs()
                            if isinstance(final_obs, dict):
                                achieved = final_obs['achieved_goal']
                                desired = final_obs['desired_goal']
                                final_distance = np.linalg.norm(achieved - desired)
                except:
                    # Fallback: estimate from reward (for dense rewards)
                    final_distance = abs(reward) if reward < 0 else 0.0

                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(1.0 if success else 0.0)
        final_distances.append(final_distance)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0 or episode < 5:
            print(f"Episode {episode+1:3d}: Reward = {episode_reward:7.2f}, "
                  f"Length = {episode_length:2d}, Success = {success}, "
                  f"Distance = {final_distance:.4f}")

        # Add delay for rendering
        if render:
            time.sleep(0.01)

    eval_time = time.time() - start_time

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = np.mean(episode_successes)
    mean_distance = np.mean([d for d in final_distances if d != float('inf')])

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes evaluated: {episodes}")
    print(f"Evaluation time: {eval_time:.1f} seconds")
    print()
    print(f"Average reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Best reward: {max(episode_rewards):.3f}")
    print(f"Worst reward: {min(episode_rewards):.3f}")
    print()
    print(f"Average episode length: {mean_length:.1f} steps")
    print(f"Success rate: {success_rate:.3f} ({int(success_rate * episodes)}/{episodes} episodes)")
    print(f"Average final distance: {mean_distance:.4f} meters")
    print()

    # Additional analysis
    successful_episodes = [r for r, s in zip(episode_rewards, episode_successes) if s > 0.5]
    if successful_episodes:
        print(f"Average reward in successful episodes: {np.mean(successful_episodes):.3f}")

    failed_episodes = [r for r, s in zip(episode_rewards, episode_successes) if s < 0.5]
    if failed_episodes:
        print(f"Average reward in failed episodes: {np.mean(failed_episodes):.3f}")

    # Performance categorization
    if success_rate >= 0.8:
        performance = "🥇 Excellent"
    elif success_rate >= 0.6:
        performance = "🥈 Good"
    elif success_rate >= 0.4:
        performance = "🥉 Fair"
    elif success_rate >= 0.2:
        performance = "⚠️ Poor"
    else:
        performance = "❌ Very Poor"

    print(f"\nOverall performance: {performance}")
    print("=" * 60)

    # Close environment
    env.close()

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'mean_distance': mean_distance,
        'episodes': episodes
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained SAC agent on Shadow Hand Reach')
    parser.add_argument('--model_path', type=str, default='models/sac_shadow_hand/sac_best.zip',
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--max_episode_steps', type=int, default=50,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        print("Please train a model first or provide the correct path.")
        exit(1)

    evaluate_agent(
        model_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed
    )
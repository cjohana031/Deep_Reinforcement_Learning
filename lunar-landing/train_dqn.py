import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import subprocess
import time
import webbrowser

from environments import LunarEnvironment
from agents.dqn import DQNAgent, DQNUpdater
from utils.logger import Logger


def train_dqn(
    episodes=1000,
    max_steps=1000,
    save_freq=100,
    eval_freq=50,
    eval_episodes=10,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = LunarEnvironment(render_mode=None, seed=seed)
    eval_env = LunarEnvironment(render_mode=None, seed=seed + 1000)

    state_dim = env.get_observation_space_shape()[0]
    action_dim = env.get_action_space_size()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Calculate decay steps based on expected total steps
    expected_steps_per_episode = 100
    total_expected_steps = episodes * expected_steps_per_episode
    epsilon_decay_steps = int(total_expected_steps * 0.8)  # Decay over 80% of training

    agent_updater = DQNUpdater(
        agent=agent,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay_steps=epsilon_decay_steps,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=100,
        device=device,
    )

    logger = Logger()
    best_avg_reward = -float('inf')
    training_info = {
        'start_time': datetime.now().isoformat(),
        'episodes': episodes,
        'seed': seed,
        'device': agent.device,
        'epsilon_decay_steps': epsilon_decay_steps
    }

    print(f"Starting DQN training on {agent.device}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Expected total steps: {total_expected_steps:,.0f}")
    print(f"Epsilon decay steps: {epsilon_decay_steps:,}")
    print("-" * 50)

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        losses = []

        for step in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent_updater.store_transition(state, action, reward, next_state, done)

            loss = agent_updater.update()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        # Calculate average loss for this episode
        avg_loss = np.mean(losses) if losses else None
        
        # Log episode with TD loss and epsilon
        logger.log_episode(
            episode=episode, 
            reward=total_reward, 
            steps=steps, 
            td_loss=avg_loss, 
            epsilon=agent_updater.epsilon,
            network_loss=avg_loss
        )

        # Log additional metrics to history
        logger.log_training_metrics(
            episode=episode,
            loss=avg_loss,
            epsilon=agent_updater.epsilon,
            td_loss=avg_loss
        )

        if episode % 10 == 0:
            avg_reward_100 = np.mean(logger.get_history()['rewards'][-100:]) if episode >= 99 else np.mean(logger.get_history()['rewards'])
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
            print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | "
                  f"Avg100: {avg_reward_100:7.2f} | Loss: {loss_str} | ε: {agent_updater.epsilon:.3f}")

        if episode % eval_freq == 0 and episode > 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                state, _ = eval_env.reset()
                eval_reward = 0
                for _ in range(max_steps):
                    action = agent.act(state, training=False)
                    state, reward, terminated, truncated, _ = eval_env.step(action)
                    eval_reward += reward
                    if terminated or truncated:
                        break
                eval_rewards.append(eval_reward)

            avg_eval_reward = np.mean(eval_rewards)
            print(f"  [EVAL] Average reward over {eval_episodes} episodes: {avg_eval_reward:.2f}")

            # Log evaluation results to CSV
            logger.log_training_metrics(episode=episode, eval_reward=avg_eval_reward)
            
            # Update CSV with eval reward for this episode
            logger._log_to_csv(episode, total_reward, steps, datetime.now().isoformat(), 
                             avg_loss, agent_updater.epsilon, avg_loss, avg_eval_reward)

            if avg_eval_reward > best_avg_reward:
                best_avg_reward = avg_eval_reward
                agent.save()
                agent_updater.save()
                print(f"  [SAVE] New best model! Average reward: {best_avg_reward:.2f}")

        if episode % save_freq == 0 and episode > 0:
            agent.save(episode)
            agent_updater.save(episode)

    agent.save()
    agent_updater.save()

    training_info['end_time'] = datetime.now().isoformat()
    training_info['final_avg_reward'] = float(np.mean(logger.get_history()['rewards'][-100:]))
    training_info['best_avg_reward'] = float(best_avg_reward)
    training_info['final_epsilon'] = float(agent_updater.epsilon)

    with open('models/dqn/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)

    logger.save('logs/dqn/training_history.npz')

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Final average reward (last 100 episodes): {training_info['final_avg_reward']:.2f}")
    print(f"Best evaluation average reward: {best_avg_reward:.2f}")
    print(f"Final epsilon: {training_info['final_epsilon']:.3f}")
    print(f"Models saved in: models/dqn/")

    env.close()
    eval_env.close()

    # Close logger
    logger.close()

    return logger.get_history()


if __name__ == "__main__":
    train_dqn(episodes=1000)
import click
import numpy as np
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint

from environments import LunarEnvironment
from agents import RandomAgent
from agents.dqn import DQNAgent
from train.train import train
from train_dqn import train_dqn
from evaluate import evaluate
from evaluate_dqn import evaluate_dqn

console = Console()


class Config:
    def __init__(self):
        self.env = None
        self.agent = None
        self.seed = None


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@pass_config
def cli(config, seed, verbose):
    """Lunar Lander Deep Reinforcement Learning CLI.
    
    A modern CLI for training and evaluating RL agents on the Lunar Lander environment.
    """
    config.seed = seed
    if verbose:
        console.print(f"[cyan]Random seed set to: {seed}[/cyan]")


@cli.group()
@pass_config
def random_agent(config):
    """Commands for Random Agent."""
    pass


@cli.group()
@pass_config
def dqn(config):
    """Commands for DQN Agent."""
    pass


@random_agent.command()
@click.option('--episodes', default=1000, help='Number of training episodes')
@click.option('--log-interval', default=100, help='Episodes between logging')
@pass_config
def train(config, episodes, log_interval):
    """Train a Random Agent on Lunar Lander."""
    with console.status("[bold green]Initializing training environment...") as status:
        env = LunarEnvironment(render_mode=None, seed=config.seed)
        agent = RandomAgent(action_space_size=env.get_action_space_size())
        
        status.update("[bold yellow]Starting training...")
        
        table = Table(title="Training Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Agent", agent.name)
        table.add_row("Episodes", str(episodes))
        table.add_row("Seed", str(config.seed))
        table.add_row("Log Interval", str(log_interval))
        
        console.print(table)
    
    from train.train import train as train_func
    history = train_func(agent, env, episodes=episodes)
    
    stats_table = Table(title="Training Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    stats_table.add_row("Final avg reward (last 100)", f"{np.mean(history['rewards'][-100:]):.2f}")
    stats_table.add_row("Best episode reward", f"{np.max(history['rewards']):.2f}")
    stats_table.add_row("Worst episode reward", f"{np.min(history['rewards']):.2f}")
    
    console.print(stats_table)
    env.close()
    console.print("[bold green]✓ Training completed successfully![/bold green]")


@random_agent.command()
@click.option('--episodes', default=5, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=False, help='Enable visualization')
@pass_config
def evaluate(config, episodes, render):
    """Evaluate a Random Agent on Lunar Lander."""
    render_mode = 'human' if render else None
    
    with console.status("[bold green]Setting up evaluation...") as status:
        env = LunarEnvironment(render_mode=render_mode, seed=config.seed)
        agent = RandomAgent(action_space_size=env.get_action_space_size())
        
        table = Table(title="Evaluation Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Agent", agent.name)
        table.add_row("Episodes", str(episodes))
        table.add_row("Rendering", "ON" if render else "OFF")
        
        console.print(table)
    
    from evaluate import evaluate as eval_func
    rewards, steps = eval_func(agent, env, episodes=episodes, render=render)
    
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Episode", style="cyan")
    results_table.add_column("Reward", style="green")
    results_table.add_column("Steps", style="yellow")
    
    for i, (r, s) in enumerate(zip(rewards, steps)):
        results_table.add_row(str(i+1), f"{r:.2f}", str(s))
    
    results_table.add_row("", "", "")
    results_table.add_row("Average", f"{np.mean(rewards):.2f}", f"{np.mean(steps):.0f}")
    
    console.print(results_table)
    env.close()
    console.print("[bold green]✓ Evaluation completed![/bold green]")


@dqn.command()
@click.option('--episodes', default=1000, help='Number of training episodes')
@click.option('--lr', default=5e-4, help='Learning rate')
@click.option('--batch-size', default=64, help='Batch size for training')
@click.option('--buffer-size', default=50000, help='Replay buffer size')
@click.option('--gamma', default=0.99, help='Discount factor')
@click.option('--epsilon-decay', default=0.995, help='Epsilon decay rate')
@click.option('--target-update-freq', default=100, help='Target network update frequency')
@click.option('--save-freq', default=100, help='Save model every N episodes')
@click.option('--eval-freq', default=50, help='Evaluate model every N episodes')
@pass_config
def train(config, episodes, lr, batch_size, buffer_size, gamma, epsilon_decay, 
         target_update_freq, save_freq, eval_freq):
    """Train a DQN Agent on Lunar Lander."""
    
    table = Table(title="DQN Training Configuration")
    table.add_column("Hyperparameter", style="cyan")
    table.add_column("Value", style="magenta")
    
    params = [
        ("Episodes", episodes),
        ("Learning Rate", lr),
        ("Batch Size", batch_size),
        ("Buffer Size", buffer_size),
        ("Gamma", gamma),
        ("Epsilon Decay", epsilon_decay),
        ("Target Update Freq", target_update_freq),
        ("Save Frequency", save_freq),
        ("Eval Frequency", eval_freq),
        ("Device", "cuda" if torch.cuda.is_available() else "cpu"),
        ("Seed", config.seed)
    ]
    
    for name, value in params:
        table.add_row(name, str(value))
    
    console.print(table)
    
    with console.status("[bold green]Training DQN agent...") as status:
        history = train_dqn(
            episodes=episodes,
            save_freq=save_freq,
            eval_freq=eval_freq,
            eval_episodes=10,
            seed=config.seed
        )
    
    console.print("[bold green]✓ DQN training completed successfully![/bold green]")
    
    if history:
        final_rewards = history.get('rewards', [])[-100:]
        if final_rewards:
            console.print(f"[cyan]Final average reward (last 100 episodes): {np.mean(final_rewards):.2f}[/cyan]")


@dqn.command()
@click.option('--episodes', default=5, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=False, help='Enable visualization')
@click.option('--model-path', type=click.Path(exists=True), help='Path to saved model')
@pass_config
def evaluate(config, episodes, render, model_path):
    """Evaluate a DQN Agent on Lunar Lander."""
    
    table = Table(title="DQN Evaluation Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Episodes", str(episodes))
    table.add_row("Rendering", "ON" if render else "OFF")
    table.add_row("Model Path", model_path or "Best saved model")
    
    console.print(table)
    
    with console.status("[bold green]Loading model and evaluating...") as status:
        rewards, steps = evaluate_dqn(
            model_path=model_path,
            episodes=episodes,
            render=render,
            seed=config.seed
        )
    
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Episode", style="cyan")
    results_table.add_column("Reward", style="green")
    results_table.add_column("Steps", style="yellow")
    
    for i, (r, s) in enumerate(zip(rewards, steps)):
        results_table.add_row(str(i+1), f"{r:.2f}", str(s))
    
    results_table.add_row("", "", "")
    results_table.add_row("Average", f"{np.mean(rewards):.2f}", f"{np.mean(steps):.0f}")
    results_table.add_row("Std Dev", f"{np.std(rewards):.2f}", f"{np.std(steps):.0f}")
    
    console.print(results_table)
    console.print("[bold green]✓ Evaluation completed![/bold green]")


@cli.command()
@click.option('--agent', type=click.Choice(['random', 'dqn']), default='random', help='Agent type')
@click.option('--train-episodes', default=500, help='Number of training episodes')
@click.option('--eval-episodes', default=5, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=False, help='Enable visualization during evaluation')
@pass_config
def benchmark(config, agent, train_episodes, eval_episodes, render):
    """Run both training and evaluation for an agent."""
    
    console.print(f"[bold cyan]Running benchmark for {agent.upper()} agent[/bold cyan]")
    console.print(f"[yellow]Training for {train_episodes} episodes, then evaluating for {eval_episodes} episodes[/yellow]")
    
    if agent == 'random':
        env = LunarEnvironment(render_mode=None, seed=config.seed)
        agent_obj = RandomAgent(action_space_size=env.get_action_space_size())
        
        console.print("[bold]Training phase...[/bold]")
        from train.train import train as train_func
        history = train_func(agent_obj, env, episodes=train_episodes)
        env.close()
        
        console.print(f"[green]Training complete! Final avg reward: {np.mean(history['rewards'][-100:]):.2f}[/green]")
        
        console.print("[bold]Evaluation phase...[/bold]")
        render_mode = 'human' if render else None
        env = LunarEnvironment(render_mode=render_mode, seed=config.seed)
        agent_obj = RandomAgent(action_space_size=env.get_action_space_size())
        
        from evaluate import evaluate as eval_func
        rewards, steps = eval_func(agent_obj, env, episodes=eval_episodes, render=render)
        
    elif agent == 'dqn':
        console.print("[bold]Training phase...[/bold]")
        history = train_dqn(
            episodes=train_episodes,
            save_freq=100,
            eval_freq=50,
            eval_episodes=10,
            seed=config.seed
        )
        
        if history:
            final_rewards = history.get('rewards', [])[-100:]
            if final_rewards:
                console.print(f"[green]Training complete! Final avg reward: {np.mean(final_rewards):.2f}[/green]")
        
        console.print("[bold]Evaluation phase...[/bold]")
        rewards, steps = evaluate_dqn(
            model_path=None,
            episodes=eval_episodes,
            render=render,
            seed=config.seed
        )
    
    console.print(f"[bold green]✓ Benchmark complete! Average evaluation reward: {np.mean(rewards):.2f}[/bold green]")


@cli.command()
def info():
    """Display information about available agents and environments."""
    
    info_table = Table(title="Lunar Lander RL System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Description", style="white")
    
    info_table.add_row("Random Agent", "Baseline agent that takes random actions")
    info_table.add_row("DQN Agent", "Deep Q-Network with experience replay and target network")
    info_table.add_row("Environment", "OpenAI Gymnasium LunarLander-v3")
    info_table.add_row("Action Space", "4 discrete actions (nothing, left, main, right)")
    info_table.add_row("State Space", "8-dimensional continuous observation")
    info_table.add_row("Reward", "+100 for landing, -100 for crashing, fuel penalties")
    
    console.print(info_table)
    
    device_info = Table(title="System Information")
    device_info.add_column("Property", style="cyan")
    device_info.add_column("Value", style="green")
    
    device_info.add_row("PyTorch Version", torch.__version__)
    device_info.add_row("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
    if torch.cuda.is_available():
        device_info.add_row("CUDA Device", torch.cuda.get_device_name(0))
    
    console.print(device_info)


if __name__ == '__main__':
    cli()
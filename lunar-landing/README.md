# Lunar Lander - Random Agent

A minimal setup for running a random agent in the Lunar Lander environment using Gymnasium.

## Project Structure

```
lunar-landing/
├── agents/              # Agent implementations
│   ├── __init__.py
│   └── random_agent.py  # Random action selection agent
├── environments/        # Environment wrappers
│   ├── __init__.py
│   └── lunar_env.py     # Lunar Lander environment wrapper
├── utils/               # Utility modules
│   ├── __init__.py
│   └── logger.py        # Training logger
├── logs/                # Training logs (auto-created)
├── main.py              # Main runner with CLI
├── train.py             # Training script
├── evaluate.py          # Evaluation script
└── requirements.txt     # Dependencies
```

## Installation

```bash
cd lunar-landing
pip install -r requirements.txt
```

## Usage

### Quick Start

Run training (1000 episodes):
```bash
python main.py --mode train --episodes 1000
```

Run evaluation with rendering (5 episodes):
```bash
python main.py --mode evaluate --eval_episodes 5 --render
```

Run both training and evaluation:
```bash
python main.py --mode both --episodes 500 --eval_episodes 3 --render
```

### Direct Scripts

Train the agent:
```bash
python train.py
```

Evaluate with visualization:
```bash
python evaluate.py
```

### CLI Options

- `--mode`: Choose between 'train', 'evaluate', or 'both'
- `--episodes`: Number of training episodes (default: 1000)
- `--eval_episodes`: Number of evaluation episodes (default: 5)
- `--render`: Enable visualization during evaluation
- `--seed`: Random seed for reproducibility (default: 42)

## Random Agent

The random agent selects actions randomly from the action space:
- Action 0: Do nothing
- Action 1: Fire left orientation engine
- Action 2: Fire main engine
- Action 3: Fire right orientation engine

This serves as a baseline for comparison with more sophisticated agents.


## Reporting

We must store in tensor board the training journey after each epoch to include in the 
final report.
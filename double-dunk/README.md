# DoubleDunk DDQN Implementation

This project implements a Double Deep Q-Network (DDQN) agent for the DoubleDunk Atari game using PyTorch and Gymnasium.

## Project Structure

```
double-dunk/
├── agents/
│   └── ddqn.py              # DDQN agent implementation
├── environments/
│   └── double_dunk_env.py   # DoubleDunk environment wrapper
├── models/
│   └── ddqn/
│       └── networks.py      # Neural network architecture
├── train/
│   └── replay_buffer.py     # Experience replay buffer
├── utils/
│   └── logger.py            # Training logger and TensorBoard integration
├── train_ddqn.py            # Training script
├── evaluate.py              # Evaluation script
├── main.py                  # Main entry point
└── requirements.txt         # Dependencies
```

## Features

- **DDQN Implementation**: Double Deep Q-Network with target network and experience replay
- **Environment Preprocessing**: Frame stacking, grayscale conversion, and frame skipping
- **Comprehensive Logging**: CSV logging and TensorBoard integration
- **Model Checkpointing**: Automatic saving of best models and training checkpoints
- **Evaluation**: Separate evaluation script with rendering support

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
```

## Usage

### Training

Train the DDQN agent:
```bash
python main.py --mode train --episodes 50000
```

### Evaluation

Evaluate a trained model:
```bash
python main.py --mode evaluate --eval_episodes 10 --render
```

### Both Training and Evaluation

```bash
python main.py --mode both --episodes 20000 --eval_episodes 5 --render
```

### Command Line Arguments

- `--mode`: Choose between 'train', 'evaluate', or 'both'
- `--episodes`: Number of training episodes (default: 50000)
- `--eval_episodes`: Number of evaluation episodes (default: 10)
- `--render`: Enable visualization during evaluation
- `--model_path`: Path to specific model file for evaluation
- `--seed`: Random seed for reproducibility (default: 42)

## Configuration

Key hyperparameters can be modified in the training script:

- **Learning Rate**: 2.5e-4
- **Batch Size**: 64
- **Replay Buffer Size**: 100,000
- **Target Network Update Frequency**: 10,000 steps
- **Epsilon Decay**: Linear from 1.0 to 0.01 over training
- **Frame Stack**: 4 frames
- **Frame Skip**: 4 frames

## Monitoring

Training progress can be monitored using:

1. **Console Output**: Real-time training statistics
2. **TensorBoard**: Run `tensorboard --logdir=runs` to view training curves
3. **CSV Logs**: Training history saved in `logs/ddqn/training_log.csv`

## Model Files

Trained models are saved in `models/ddqn/`:
- `ddqn_best.pth`: Best performing model during training
- `ddqn_final.pth`: Final model after training completion
- `ddqn_episode_N.pth`: Checkpoint at episode N
- `training_info.json`: Training metadata and statistics

## Hardware Support

The implementation automatically detects and uses available hardware:
- **CUDA**: For NVIDIA GPUs
- **MPS**: For Apple Silicon Macs
- **CPU**: Fallback for systems without GPU acceleration
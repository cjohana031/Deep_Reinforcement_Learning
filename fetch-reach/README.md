# TD3 for Shadow Dexterous Hand Reach (FetchReachDense-v4)

This project implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm to solve the Shadow Dexterous Hand Reach task using the dense reward variant (FetchReachDense-v4) from gymnasium-robotics.

## Problem Description

The task involves controlling a Shadow Dexterous Hand (20 degrees of freedom) to move fingertips to target positions. The hand is an anthropomorphic robotic hand with:

- **Action Space**: 20-dimensional continuous control (joint angles)
- **Observation Space**: 78-dimensional state (63 for hand kinematics + 15 for goal positions)
- **Goal**: Reach predefined target Cartesian positions with fingertips
- **Reward**: Dense reward based on negative distance to target positions

## Algorithm: Twin Delayed DDPG (TD3)

TD3 is an improvement over DDPG that addresses overestimation bias through:

1. **Twin Critics**: Two critic networks, taking minimum Q-value to reduce overestimation
2. **Delayed Policy Updates**: Update policy less frequently than critics
3. **Target Policy Smoothing**: Add noise to target actions for robustness

## Project Structure

```
fetch-reach/
├── agents/
│   └── td3.py                 # TD3 agent implementation
├── environments/
│   └── hand_reach_env.py      # Environment wrapper
├── models/
│   └── td3/
│       └── networks.py        # Actor and Critic networks
├── train/
│   └── replay_buffer.py       # Experience replay buffer
├── utils/
│   └── logger.py             # Logging utilities
├── logs/                     # Training logs (created during training)
├── models/                   # Saved models (created during training)
├── main.py                   # Main entry point
├── train_td3.py             # Training script
├── evaluate_td3.py          # Evaluation script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. For gymnasium-robotics with MuJoCo support:
```bash
pip install gymnasium-robotics[mujoco]
```

## Usage

### Training

Train a TD3 agent from scratch:
```bash
python main.py train --episodes 2000
```

Train with custom hyperparameters:
```bash
python main.py train \
    --episodes 1000 \
    --lr-actor 1e-4 \
    --lr-critic 1e-4 \
    --batch-size 128 \
    --exploration-noise 0.2
```

Continue training from a checkpoint:
```bash
python main.py train \
    --episodes 1000 \
    --load-model models/td3/td3_episode_500.pth
```

### Evaluation

Evaluate a trained agent:
```bash
python main.py evaluate --num-episodes 100
```

Evaluate with visualization:
```bash
python main.py evaluate \
    --num-episodes 10 \
    --render \
    --verbose-episodes
```

### Direct Script Usage

You can also run the scripts directly:

```bash
# Training
python train_td3.py --episodes 2000 --log-freq 50

# Evaluation
python evaluate_td3.py --num-episodes 100 --render
```

## Hyperparameters

Default TD3 hyperparameters optimized for the HandReach task:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate (Actor) | 3e-4 | Actor network learning rate |
| Learning Rate (Critic) | 3e-4 | Critic network learning rate |
| Discount Factor (γ) | 0.99 | Reward discount factor |
| Soft Update Rate (τ) | 0.005 | Target network update rate |
| Policy Noise | 0.2 | Target policy smoothing noise |
| Noise Clip | 0.5 | Target policy noise clipping |
| Policy Frequency | 2 | Delayed policy update frequency |
| Exploration Noise | 0.1 | Action exploration noise |
| Batch Size | 256 | Training batch size |
| Buffer Size | 1,000,000 | Replay buffer capacity |
| Hidden Size | 256 | Neural network hidden layer size |

## Training Progress

The implementation includes comprehensive logging:

- **CSV Logs**: Episode metrics saved to CSV files
- **TensorBoard**: Real-time visualization of training progress
- **Console Output**: Progress tracking and statistics
- **Model Checkpoints**: Automatic saving of best models

Monitor training progress:
```bash
tensorboard --logdir logs/
```

## Results

Expected performance metrics after training:

- **Training Episodes**: ~2000 episodes for convergence
- **Success Rate**: >80% after proper training
- **Episode Length**: ~50 steps (environment limit)
- **Reward Range**: Depends on distance to targets (dense reward)

## Key Features

1. **Environment Wrapper**: Handles goal-aware observations and flattens dictionary observations
2. **Robust TD3 Implementation**: Includes all key TD3 improvements
3. **Comprehensive Logging**: TensorBoard and CSV logging with detailed metrics
4. **Flexible Training**: Command-line interface with extensive hyperparameter control
5. **Evaluation Tools**: Detailed evaluation with visualization support
6. **Model Management**: Automatic checkpointing and best model saving

## Troubleshooting

### Common Issues

1. **Import Error for gymnasium-robotics**:
   ```bash
   pip install gymnasium-robotics[mujoco]
   ```

2. **MuJoCo License Issues**:
   - MuJoCo is now free, ensure you have the latest version
   - Check gymnasium-robotics installation

3. **GPU Memory Issues**:
   - Reduce batch size: `--batch-size 128`
   - Reduce buffer size: `--buffer-size 500000`

4. **Slow Training**:
   - Ensure CUDA is available for GPU acceleration
   - Check that MuJoCo is properly installed

### Performance Tips

1. **GPU Acceleration**: Ensure PyTorch can use CUDA if available
2. **Batch Size**: Larger batch sizes often improve stability
3. **Exploration**: Adjust exploration noise based on task complexity
4. **Network Size**: Increase hidden size for more complex tasks

## Algorithm Details

### TD3 Key Components

1. **Actor Network**: Deterministic policy mapping states to actions
2. **Twin Critics**: Two Q-networks to reduce overestimation bias
3. **Target Networks**: Slowly updated target networks for stability
4. **Experience Replay**: Large buffer storing past experiences
5. **Target Policy Smoothing**: Noise injection for robustness

### Training Loop

1. Collect experience using current policy with exploration noise
2. Store transitions in replay buffer
3. Sample mini-batch from buffer
4. Update twin critics using Bellman equation
5. Delay policy updates (every 2 critic updates)
6. Soft update target networks

## References

- [TD3 Paper](https://arxiv.org/abs/1802.09477): "Addressing Function Approximation Error in Actor-Critic Methods"
- [Gymnasium-Robotics](https://robotics.farama.org/): Official documentation
- [HandReach Environment](https://robotics.farama.org/envs/shadow_dexterous_hand/reach/): Task description

## License

This project is part of a reinforcement learning course implementation. See course materials for specific licensing terms.
import argparse
from train_ddqn import train_ddqn
from evaluate import evaluate_ddqn


def main():
    parser = argparse.ArgumentParser(description='DDQN BeamRider Training and Evaluation')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], 
                       default='train', help='Mode: train, evaluate, or both')
    parser.add_argument('--episodes', type=int, default=50000, 
                       help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=10, 
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', 
                       help='Render during training and evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model file for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Training DDQN for {args.episodes} episodes...")
        train_ddqn(episodes=args.episodes, render=args.render)
        
    elif args.mode == 'evaluate':
        print(f"Evaluating DDQN for {args.eval_episodes} episodes...")
        evaluate_ddqn(episodes=args.eval_episodes, render=args.render, 
                     model_path=args.model_path)
        
    elif args.mode == 'both':
        print(f"Training DDQN for {args.episodes} episodes...")
        train_ddqn(episodes=args.episodes, render=args.render)
        
        print(f"\nEvaluating trained model for {args.eval_episodes} episodes...")
        evaluate_ddqn(episodes=args.eval_episodes, render=args.render)


if __name__ == "__main__":
    main()
"""
Training script for RL-DAS (Dynamic Algorithm Selection).
Trains PPO agent to learn optimal algorithm selection policy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from typing import List, Dict, Any
import json
from datetime import datetime

from problems.tsp import TSPProblem
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.tabu_search import TabuSearch
from core.environment import OptimizationEnvironment
from agents.policy_network import PolicyNetwork
from agents.value_network import ValueNetwork
from agents.ppo_trainer import PPOTrainer


def create_tsp_instances(num_instances: int, num_cities: int, 
                         instance_type: str = 'random', seed: int = 42) -> List[TSPProblem]:
    """
    Create TSP instances for training.
    
    Args:
        num_instances: Number of instances to create
        num_cities: Number of cities per instance
        instance_type: Type of instance ('random', 'clustered', 'mixed')
        seed: Random seed
        
    Returns:
        List of TSPProblem instances
    """
    np.random.seed(seed)
    instances = []
    
    for i in range(num_instances):
        instance_seed = seed + i
        
        if instance_type == 'random':
            tsp = TSPProblem.create_random_instance(num_cities, seed=instance_seed)
        elif instance_type == 'clustered':
            num_clusters = max(2, num_cities // 20)  # Adaptive cluster count
            tsp = TSPProblem.create_clustered_instance(num_cities, num_clusters, seed=instance_seed)
        elif instance_type == 'mixed':
            # Mix random and clustered
            if i % 2 == 0:
                tsp = TSPProblem.create_random_instance(num_cities, seed=instance_seed)
            else:
                num_clusters = max(2, num_cities // 20)
                tsp = TSPProblem.create_clustered_instance(num_cities, num_clusters, seed=instance_seed)
        else:
            raise ValueError(f"Unknown instance type: {instance_type}")
        
        instances.append(tsp)
    
    return instances


def create_algorithms(tsp: TSPProblem, config: Dict[str, Any]) -> List:
    """
    Create algorithm pool for the environment.
    
    Args:
        tsp: TSP problem instance
        config: Configuration dictionary
        
    Returns:
        List of algorithm instances
    """
    algorithms = []
    
    # Genetic Algorithm
    if config.get('use_ga', True):
        algorithms.append(GeneticAlgorithm(
            tsp,
            population_size=config.get('ga_population_size', 50),
            mutation_rate=config.get('ga_mutation_rate', 0.1),
            crossover_rate=config.get('ga_crossover_rate', 0.8)
        ))
    
    # Simulated Annealing
    if config.get('use_sa', True):
        algorithms.append(SimulatedAnnealing(
            tsp,
            initial_temperature=config.get('sa_initial_temp', 100),
            cooling_rate=config.get('sa_cooling_rate', 0.95)
        ))
    
    # Tabu Search
    if config.get('use_ts', True):
        algorithms.append(TabuSearch(
            tsp,
            tabu_tenure=config.get('ts_tabu_tenure', 10)
        ))
    
    if len(algorithms) == 0:
        raise ValueError("At least one algorithm must be enabled")
    
    return algorithms


def create_environment(tsp: TSPProblem, algorithms: List, config: Dict[str, Any]) -> OptimizationEnvironment:
    """
    Create optimization environment.
    
    Args:
        tsp: TSP problem instance
        algorithms: List of algorithms
        config: Configuration dictionary
        
    Returns:
        OptimizationEnvironment instance
    """
    return OptimizationEnvironment(
        problem=tsp,
        algorithms=algorithms,
        max_evaluations=config.get('max_evaluations', 20000),
        schedule_interval=config.get('schedule_interval', 1000),
        reward_type=config.get('reward_type', 'improvement_with_efficiency')
    )


def train_on_instance(tsp: TSPProblem, trainer: PPOTrainer, config: Dict[str, Any], 
                      instance_id: int) -> Dict[str, Any]:
    """
    Train on a single TSP instance.
    
    Args:
        tsp: TSP problem instance
        trainer: PPO trainer
        config: Configuration dictionary
        instance_id: Instance identifier
        
    Returns:
        Dictionary with training results
    """
    # Create algorithms for this instance
    algorithms = create_algorithms(tsp, config)
    
    # Create environment
    env = create_environment(tsp, algorithms, config)
    
    # Collect trajectories and update
    trajectories = trainer.collect_trajectories(env, config.get('episodes_per_instance', 1))
    metrics = trainer.update(trajectories)
    
    # Get episode statistics
    episode_info = env.get_episode_info()
    
    return {
        'instance_id': instance_id,
        'num_cities': tsp.get_problem_size(),
        'best_cost': episode_info['best_cost'],
        'total_reward': episode_info['total_reward'],
        'total_steps': episode_info['total_steps'],
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Train RL-DAS agent')
    
    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--instances_per_epoch', type=int, default=16,
                       help='Number of instances per epoch')
    parser.add_argument('--num_cities', type=int, default=50,
                       help='Number of cities in TSP instances')
    parser.add_argument('--instance_type', type=str, default='mixed',
                       choices=['random', 'clustered', 'mixed'],
                       help='Type of TSP instances to generate')
    
    # Environment configuration
    parser.add_argument('--max_evaluations', type=int, default=20000,
                       help='Maximum evaluations per episode')
    parser.add_argument('--schedule_interval', type=int, default=1000,
                       help='Evaluations between algorithm selections')
    parser.add_argument('--reward_type', type=str, default='improvement_with_efficiency',
                       choices=['improvement', 'improvement_with_efficiency', 'normalized_improvement'],
                       help='Reward function type')
    
    # PPO configuration
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy bonus coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Maximum gradient norm')
    parser.add_argument('--update_epochs', type=int, default=4,
                       help='Number of update epochs per batch')
    
    # Network configuration
    parser.add_argument('--hidden_dims', type=str, default='64,32',
                       help='Hidden layer dimensions (comma-separated)')
    
    # Algorithm configuration
    parser.add_argument('--use_ga', action='store_true', default=True,
                       help='Use Genetic Algorithm')
    parser.add_argument('--use_sa', action='store_true', default=True,
                       help='Use Simulated Annealing')
    parser.add_argument('--use_ts', action='store_true', default=True,
                       help='Use Tabu Search')
    parser.add_argument('--ga_population_size', type=int, default=50,
                       help='GA population size')
    parser.add_argument('--ga_mutation_rate', type=float, default=0.1,
                       help='GA mutation rate')
    parser.add_argument('--ga_crossover_rate', type=float, default=0.8,
                       help='GA crossover rate')
    parser.add_argument('--sa_initial_temp', type=float, default=100,
                       help='SA initial temperature')
    parser.add_argument('--sa_cooling_rate', type=float, default=0.95,
                       help='SA cooling rate')
    parser.add_argument('--ts_tabu_tenure', type=int, default=10,
                       help='TS tabu tenure')
    
    # Training options
    parser.add_argument('--episodes_per_instance', type=int, default=1,
                       help='Episodes to run per instance')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_freq', type=int, default=20,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--validation_freq', type=int, default=10,
                       help='Run validation every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print training progress')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Parse hidden dimensions
    hidden_dims = [int(d) for d in args.hidden_dims.split(',')]
    
    # Build configuration dictionary
    config = {
        'num_cities': args.num_cities,
        'instance_type': args.instance_type,
        'max_evaluations': args.max_evaluations,
        'schedule_interval': args.schedule_interval,
        'reward_type': args.reward_type,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_epsilon': args.clip_epsilon,
        'value_loss_coef': args.value_loss_coef,
        'entropy_coef': args.entropy_coef,
        'max_grad_norm': args.max_grad_norm,
        'update_epochs': args.update_epochs,
        'hidden_dims': hidden_dims,
        'use_ga': args.use_ga,
        'use_sa': args.use_sa,
        'use_ts': args.use_ts,
        'ga_population_size': args.ga_population_size,
        'ga_mutation_rate': args.ga_mutation_rate,
        'ga_crossover_rate': args.ga_crossover_rate,
        'sa_initial_temp': args.sa_initial_temp,
        'sa_cooling_rate': args.sa_cooling_rate,
        'ts_tabu_tenure': args.ts_tabu_tenure,
        'episodes_per_instance': args.episodes_per_instance,
        'seed': args.seed
    }
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to {config_path}")
    
    # Create a sample TSP instance to determine state/action dimensions
    print("\n" + "="*60)
    print("INITIALIZING TRAINING")
    print("="*60)
    
    sample_tsp = TSPProblem.create_random_instance(args.num_cities, seed=args.seed)
    sample_algorithms = create_algorithms(sample_tsp, config)
    sample_env = create_environment(sample_tsp, sample_algorithms, config)
    
    state_dim = sample_env.get_observation_space_size()
    action_dim = sample_env.get_action_space_size()
    
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Algorithms: {[alg.name for alg in sample_algorithms]}")
    
    # Create networks
    print(f"\nCreating neural networks...")
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims=hidden_dims)
    value_net = ValueNetwork(state_dim, hidden_dims=hidden_dims)
    
    # Create trainer
    trainer = PPOTrainer(
        policy_network=policy_net,
        value_network=value_net,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        device=args.device
    )
    print(f"✓ Trainer initialized")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Instances per epoch: {args.instances_per_epoch}")
    print(f"TSP size: {args.num_cities} cities")
    print(f"Instance type: {args.instance_type}")
    print(f"{'='*60}\n")
    
    training_history = {
        'epoch': [],
        'mean_best_cost': [],
        'mean_reward': [],
        'mean_steps': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': []
    }
    
    best_mean_cost = float('inf')
    
    for epoch in range(args.num_epochs):
        # Create instances for this epoch
        instances = create_tsp_instances(
            args.instances_per_epoch,
            args.num_cities,
            args.instance_type,
            seed=args.seed + epoch * 1000
        )
        
        epoch_results = []
        epoch_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
        
        for instance_id, tsp in enumerate(instances):
            result = train_on_instance(tsp, trainer, config, instance_id)
            epoch_results.append(result)
            
            # Accumulate metrics
            for key in epoch_metrics:
                if key in result['metrics']:
                    epoch_metrics[key].append(result['metrics'][key])
        
        # Compute epoch statistics
        mean_best_cost = np.mean([r['best_cost'] for r in epoch_results])
        mean_reward = np.mean([r['total_reward'] for r in epoch_results])
        mean_steps = np.mean([r['total_steps'] for r in epoch_results])
        mean_policy_loss = np.mean(epoch_metrics['policy_loss'])
        mean_value_loss = np.mean(epoch_metrics['value_loss'])
        mean_entropy = np.mean(epoch_metrics['entropy'])
        
        # Update history
        training_history['epoch'].append(epoch + 1)
        training_history['mean_best_cost'].append(mean_best_cost)
        training_history['mean_reward'].append(mean_reward)
        training_history['mean_steps'].append(mean_steps)
        training_history['policy_loss'].append(mean_policy_loss)
        training_history['value_loss'].append(mean_value_loss)
        training_history['entropy'].append(mean_entropy)
        
        # Print progress
        if args.verbose and (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            print(f"  Mean Best Cost: {mean_best_cost:.2f}")
            print(f"  Mean Reward: {mean_reward:.4f}")
            print(f"  Mean Steps: {mean_steps:.1f}")
            print(f"  Policy Loss: {mean_policy_loss:.4f}")
            print(f"  Value Loss: {mean_value_loss:.4f}")
            print(f"  Entropy: {mean_entropy:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
            
            # Save training history
            history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            if args.verbose:
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Track best performance
        if mean_best_cost < best_mean_cost:
            best_mean_cost = mean_best_cost
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            trainer.save_checkpoint(best_checkpoint_path, epoch + 1)
    
    # Final save
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    trainer.save_checkpoint(final_checkpoint_path, args.num_epochs)
    
    # Save final training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final mean best cost: {mean_best_cost:.2f}")
    print(f"Best mean cost: {best_mean_cost:.2f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Training history saved to: {history_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


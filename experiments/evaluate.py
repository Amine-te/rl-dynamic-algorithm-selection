"""
Evaluation script for RL-DAS (Dynamic Algorithm Selection).
Tests trained models on held-out test instances.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import json
from typing import List, Dict, Any

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
    """Create TSP instances for evaluation."""
    np.random.seed(seed)
    instances = []
    
    for i in range(num_instances):
        instance_seed = seed + i
        
        if instance_type == 'random':
            tsp = TSPProblem.create_random_instance(num_cities, seed=instance_seed)
        elif instance_type == 'clustered':
            num_clusters = max(2, num_cities // 20)
            tsp = TSPProblem.create_clustered_instance(num_cities, num_clusters, seed=instance_seed)
        elif instance_type == 'mixed':
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
    """Create algorithm pool."""
    algorithms = []
    
    if config.get('use_ga', True):
        algorithms.append(GeneticAlgorithm(
            tsp,
            population_size=config.get('ga_population_size', 50),
            mutation_rate=config.get('ga_mutation_rate', 0.1),
            crossover_rate=config.get('ga_crossover_rate', 0.8)
        ))
    
    if config.get('use_sa', True):
        algorithms.append(SimulatedAnnealing(
            tsp,
            initial_temperature=config.get('sa_initial_temp', 100),
            cooling_rate=config.get('sa_cooling_rate', 0.95)
        ))
    
    if config.get('use_ts', True):
        algorithms.append(TabuSearch(
            tsp,
            tabu_tenure=config.get('ts_tabu_tenure', 10)
        ))
    
    return algorithms


def create_environment(tsp: TSPProblem, algorithms: List, config: Dict[str, Any]) -> OptimizationEnvironment:
    """Create optimization environment."""
    return OptimizationEnvironment(
        problem=tsp,
        algorithms=algorithms,
        max_evaluations=config.get('max_evaluations', 20000),
        schedule_interval=config.get('schedule_interval', 1000),
        reward_type=config.get('reward_type', 'improvement_with_efficiency')
    )


def evaluate_with_rl_agent(tsp: TSPProblem, policy_net: PolicyNetwork, 
                           config: Dict[str, Any], deterministic: bool = True) -> Dict[str, Any]:
    """
    Evaluate RL agent on a TSP instance.
    
    Args:
        tsp: TSP problem instance
        policy_net: Trained policy network
        config: Configuration dictionary
        deterministic: If True, use deterministic policy (argmax)
        
    Returns:
        Dictionary with evaluation results
    """
    algorithms = create_algorithms(tsp, config)
    env = create_environment(tsp, algorithms, config)
    
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    selection_counts = {alg.name: 0 for alg in algorithms}
    
    while not done:
        # Select action using trained policy
        action, _ = policy_net.select_action(state, deterministic=deterministic)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        step += 1
        selection_counts[info['algorithm_selected']] += 1
        state = next_state
    
    episode_info = env.get_episode_info()
    
    return {
        'best_cost': episode_info['best_cost'],
        'total_reward': total_reward,
        'total_steps': step,
        'selection_counts': selection_counts,
        'cost_history': episode_info['cost_history']
    }


def evaluate_single_algorithm(tsp: TSPProblem, algorithm, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single algorithm on a TSP instance.
    
    Args:
        tsp: TSP problem instance
        algorithm: Algorithm instance
        config: Configuration dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    algorithm.initialize()
    
    max_evals = config.get('max_evaluations', 20000)
    schedule_interval = config.get('schedule_interval', 1000)
    
    best_cost = float('inf')
    cost_history = []
    
    while algorithm.evaluations_used < max_evals:
        remaining = max_evals - algorithm.evaluations_used
        step_evals = min(schedule_interval, remaining)
        
        solution, cost = algorithm.step(step_evals)
        best_cost = min(best_cost, cost)
        cost_history.append(best_cost)
        
        if algorithm.evaluations_used >= max_evals:
            break
    
    return {
        'best_cost': best_cost,
        'cost_history': cost_history,
        'total_evaluations': algorithm.evaluations_used
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL-DAS agent')
    
    # Model loading
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint and config')
    parser.add_argument('--checkpoint_file', type=str, default='best_model.pt',
                       help='Checkpoint file to load (relative to checkpoint_dir)')
    
    # Evaluation configuration
    parser.add_argument('--num_test_instances', type=int, default=20,
                       help='Number of test instances')
    parser.add_argument('--num_cities', type=int, default=None,
                       help='Number of cities (overrides config if provided)')
    parser.add_argument('--instance_type', type=str, default=None,
                       choices=['random', 'clustered', 'mixed'],
                       help='Instance type (overrides config if provided)')
    parser.add_argument('--test_seed', type=int, default=9999,
                       help='Random seed for test instances')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (argmax)')
    
    # Comparison baselines
    parser.add_argument('--compare_baselines', action='store_true', default=True,
                       help='Compare against individual algorithms')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs per instance for baselines')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override config with command-line arguments
    if args.num_cities is not None:
        config['num_cities'] = args.num_cities
    if args.instance_type is not None:
        config['instance_type'] = args.instance_type
    
    num_cities = config['num_cities']
    instance_type = config.get('instance_type', 'mixed')
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("="*60)
    print("EVALUATION")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test instances: {args.num_test_instances}")
    print(f"TSP size: {num_cities} cities")
    print(f"Instance type: {instance_type}")
    print("="*60)
    
    # Create sample environment to get dimensions
    sample_tsp = TSPProblem.create_random_instance(num_cities, seed=42)
    sample_algorithms = create_algorithms(sample_tsp, config)
    sample_env = create_environment(sample_tsp, sample_algorithms, config)
    
    state_dim = sample_env.get_observation_space_size()
    action_dim = sample_env.get_action_space_size()
    
    # Create networks
    hidden_dims = config.get('hidden_dims', [64, 32])
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims=hidden_dims)
    value_net = ValueNetwork(state_dim, hidden_dims=hidden_dims)
    
    # Load checkpoint
    trainer = PPOTrainer(
        policy_network=policy_net,
        value_network=value_net,
        learning_rate=config.get('learning_rate', 3e-4),
        device='cpu'
    )
    trainer.load_checkpoint(checkpoint_path)
    print(f"✓ Model loaded from checkpoint")
    
    # Create test instances
    test_instances = create_tsp_instances(
        args.num_test_instances,
        num_cities,
        instance_type,
        seed=args.test_seed
    )
    
    # Evaluate RL agent
    print(f"\nEvaluating RL-DAS agent...")
    rl_results = []
    
    for i, tsp in enumerate(test_instances):
        result = evaluate_with_rl_agent(tsp, policy_net, config, args.deterministic)
        rl_results.append(result)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{args.num_test_instances} instances")
    
    # Compute RL statistics
    rl_costs = [r['best_cost'] for r in rl_results]
    rl_mean_cost = np.mean(rl_costs)
    rl_std_cost = np.std(rl_costs)
    rl_min_cost = np.min(rl_costs)
    rl_max_cost = np.max(rl_costs)
    
    print(f"\n{'='*60}")
    print("RL-DAS RESULTS")
    print(f"{'='*60}")
    print(f"Mean Best Cost: {rl_mean_cost:.2f} ± {rl_std_cost:.2f}")
    print(f"Min Cost: {rl_min_cost:.2f}")
    print(f"Max Cost: {rl_max_cost:.2f}")
    
    # Compare with individual algorithms
    if args.compare_baselines:
        print(f"\n{'='*60}")
        print("BASELINE COMPARISON")
        print(f"{'='*60}")
        
        baseline_results = {alg_name: [] for alg_name in [alg.name for alg in sample_algorithms]}
        
        for i, tsp in enumerate(test_instances):
            if (i + 1) % 5 == 0:
                print(f"  Evaluating baselines on instance {i + 1}/{args.num_test_instances}")
            
            for run in range(args.num_runs):
                algorithms = create_algorithms(tsp, config)
                
                for alg in algorithms:
                    alg_result = evaluate_single_algorithm(tsp, alg, config)
                    baseline_results[alg.name].append(alg_result['best_cost'])
        
        # Print baseline statistics
        print(f"\nBaseline Algorithm Results ({args.num_runs} runs per instance):")
        for alg_name, costs in baseline_results.items():
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            min_cost = np.min(costs)
            print(f"  {alg_name:10s}: Mean={mean_cost:8.2f} ± {std_cost:6.2f}, Min={min_cost:8.2f}")
        
        # Compare RL vs best baseline
        all_baseline_costs = []
        for costs in baseline_results.values():
            all_baseline_costs.extend(costs)
        
        best_baseline_mean = min([np.mean(costs) for costs in baseline_results.values()])
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"RL-DAS Mean Cost:     {rl_mean_cost:.2f}")
        print(f"Best Baseline Mean:   {best_baseline_mean:.2f}")
        improvement = ((best_baseline_mean - rl_mean_cost) / best_baseline_mean) * 100
        print(f"Improvement:           {improvement:+.2f}%")
        print(f"{'='*60}")
    
    # Save results
    results = {
        'rl_results': {
            'mean_cost': float(rl_mean_cost),
            'std_cost': float(rl_std_cost),
            'min_cost': float(rl_min_cost),
            'max_cost': float(rl_max_cost),
            'all_costs': [float(c) for c in rl_costs]
        }
    }
    
    if args.compare_baselines:
        results['baseline_results'] = {
            alg_name: {
                'mean_cost': float(np.mean(costs)),
                'std_cost': float(np.std(costs)),
                'min_cost': float(np.min(costs)),
                'all_costs': [float(c) for c in costs]
            }
            for alg_name, costs in baseline_results.items()
        }
    
    results_path = os.path.join(args.checkpoint_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()


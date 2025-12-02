"""
Test script for PPO Trainer.
Verifies that training loop works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from problems.tsp import TSPProblem
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.tabu_search import TabuSearch
from core.environment import OptimizationEnvironment
from agents.policy_network import PolicyNetwork
from agents.value_network import ValueNetwork
from agents.ppo_trainer import PPOTrainer


def test_ppo_trainer():
    """Test PPO trainer with a quick training run."""
    print("="*60)
    print("PPO TRAINER TEST")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    tsp = TSPProblem.create_random_instance(num_cities=20, seed=42)
    algorithms = [
        GeneticAlgorithm(tsp, population_size=30),
        SimulatedAnnealing(tsp, initial_temperature=100, cooling_rate=0.95),
        TabuSearch(tsp, tabu_tenure=10)
    ]
    
    env = OptimizationEnvironment(
        problem=tsp,
        algorithms=algorithms,
        max_evaluations=5000,
        schedule_interval=500
    )
    print(f"   ✓ Environment created")
    
    # Create networks
    print("\n2. Creating neural networks...")
    state_dim = env.get_observation_space_size()
    action_dim = env.get_action_space_size()
    
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims=[64, 32])
    value_net = ValueNetwork(state_dim, hidden_dims=[64, 32])
    print(f"   ✓ Policy network: {state_dim} -> [64, 32] -> {action_dim}")
    print(f"   ✓ Value network: {state_dim} -> [64, 32] -> 1")
    
    # Create trainer
    print("\n3. Creating PPO trainer...")
    trainer = PPOTrainer(
        policy_network=policy_net,
        value_network=value_net,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=4
    )
    print(f"   ✓ PPO trainer initialized")
    
    # Test trajectory collection
    print("\n4. Testing trajectory collection...")
    trajectories = trainer.collect_trajectories(env, num_episodes=2)
    print(f"   ✓ Collected {len(trajectories['states'])} transitions")
    print(f"   ✓ Episode rewards: {trajectories['episode_rewards']}")
    print(f"   ✓ Episode lengths: {trajectories['episode_lengths']}")
    print(f"   ✓ Best costs: {trajectories['best_costs']}")
    
    # Test update
    print("\n5. Testing PPO update...")
    metrics = trainer.update(trajectories)
    print(f"   ✓ Update completed")
    print(f"     Policy loss: {metrics['policy_loss']:.4f}")
    print(f"     Value loss: {metrics['value_loss']:.4f}")
    print(f"     Entropy: {metrics['entropy']:.4f}")
    
    # Test short training run
    print("\n6. Running short training (5 iterations)...")
    history = trainer.train(
        env=env,
        num_iterations=5,
        episodes_per_iteration=4,
        save_interval=3,
        checkpoint_dir='test_checkpoints',
        verbose=True
    )
    
    print(f"\n   ✓ Training completed")
    print(f"   ✓ Final mean reward: {history['mean_episode_reward'][-1]:.4f}")
    print(f"   ✓ Final mean best cost: {history['mean_best_cost'][-1]:.2f}")
    
    # Test checkpoint saving/loading
    print("\n7. Testing checkpoint save/load...")
    trainer.save_checkpoint('test_checkpoints/test_checkpoint.pt', iteration=5)
    
    # Create new trainer and load
    new_trainer = PPOTrainer(
        policy_network=PolicyNetwork(state_dim, action_dim, hidden_dims=[64, 32]),
        value_network=ValueNetwork(state_dim, hidden_dims=[64, 32])
    )
    loaded_iteration = new_trainer.load_checkpoint('test_checkpoints/test_checkpoint.pt')
    print(f"   ✓ Checkpoint loaded (iteration {loaded_iteration})")
    
    print("\n" + "="*60)
    print("✓ ALL PPO TRAINER TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_ppo_trainer()
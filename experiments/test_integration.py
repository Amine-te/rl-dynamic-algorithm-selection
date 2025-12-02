"""
Integration test for the optimization environment.
Tests all components working together with random algorithm selection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from problems.tsp import TSPProblem
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.tabu_search import TabuSearch
from core.environment import OptimizationEnvironment


def test_single_episode():
    """Test a single episode with random algorithm selection."""
    print("="*60)
    print("INTEGRATION TEST: Single Episode with Random Selection")
    print("="*60)
    
    # Create a small TSP instance
    print("\n1. Creating TSP instance (20 cities)...")
    tsp = TSPProblem.create_random_instance(num_cities=20, seed=42)
    print(f"   ✓ Created TSP with {tsp.get_problem_size()} cities")
    
    # Create algorithms
    print("\n2. Initializing algorithms...")
    algorithms = [
        GeneticAlgorithm(tsp, population_size=30),
        SimulatedAnnealing(tsp, initial_temperature=100, cooling_rate=0.95),
        TabuSearch(tsp, tabu_tenure=10)
    ]
    print(f"   ✓ Created {len(algorithms)} algorithms:")
    for alg in algorithms:
        print(f"     - {alg.name}")
    
    # Create environment
    print("\n3. Creating optimization environment...")
    env = OptimizationEnvironment(
        problem=tsp,
        algorithms=algorithms,
        max_evaluations=5000,
        schedule_interval=500,
        reward_type='improvement_with_efficiency'
    )
    print(f"   ✓ Environment created")
    print(f"     - Max evaluations: {env.max_evaluations}")
    print(f"     - Schedule interval: {env.schedule_interval}")
    print(f"     - Observation space size: {env.get_observation_space_size()}")
    print(f"     - Action space size: {env.get_action_space_size()}")
    
    # Reset environment
    print("\n4. Resetting environment...")
    state = env.reset()
    print(f"   ✓ Environment reset")
    print(f"     - Initial state shape: {state.shape}")
    print(f"     - Initial state sample (first 5): {state[:5]}")
    
    # Run episode with random actions
    print("\n5. Running episode with RANDOM algorithm selection...")
    print("-"*60)
    
    step = 0
    total_reward = 0
    done = False
    
    while not done:
        # Random action (algorithm selection)
        action = np.random.randint(0, env.get_action_space_size())
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        # Print step info
        print(f"\nStep {step}:")
        print(f"  Action: {action} ({info['algorithm_selected']})")
        print(f"  Best cost: {info['best_cost']:.2f}")
        print(f"  Reward: {reward:.6f}")
        print(f"  Evaluations: {info['total_evaluations']}/{env.max_evaluations}")
        print(f"  Cost improvement: {info['cost_improvement']:.4f}")
        
        state = next_state
    
    print("\n" + "-"*60)
    print("EPISODE COMPLETED!")
    print("-"*60)
    
    # Get episode info
    episode_info = env.get_episode_info()
    selection_counts = env.get_algorithm_selection_counts()
    
    print(f"\nFinal Results:")
    print(f"  Final best cost: {episode_info['best_cost']:.2f}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Total steps: {episode_info['total_steps']}")
    print(f"  Initial cost: {episode_info['cost_history'][0]:.2f}")
    print(f"  Improvement: {episode_info['cost_history'][0] - episode_info['best_cost']:.2f}")
    print(f"  Improvement %: {((episode_info['cost_history'][0] - episode_info['best_cost']) / episode_info['cost_history'][0] * 100):.2f}%")
    
    print(f"\nAlgorithm Selection Counts:")
    for alg_name, count in selection_counts.items():
        percentage = (count / episode_info['total_steps']) * 100
        print(f"  {alg_name}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("✓ INTEGRATION TEST PASSED!")
    print("="*60)
    
    return episode_info


def test_multiple_episodes():
    """Test multiple episodes to ensure reset works correctly."""
    print("\n\n")
    print("="*60)
    print("INTEGRATION TEST: Multiple Episodes")
    print("="*60)
    
    tsp = TSPProblem.create_random_instance(num_cities=15, seed=123)
    algorithms = [
        GeneticAlgorithm(tsp, population_size=20),
        SimulatedAnnealing(tsp, initial_temperature=100, cooling_rate=0.95),
        TabuSearch(tsp, tabu_tenure=8)
    ]
    
    env = OptimizationEnvironment(
        problem=tsp,
        algorithms=algorithms,
        max_evaluations=3000,
        schedule_interval=500
    )
    
    num_episodes = 3
    results = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = np.random.randint(0, env.get_action_space_size())
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        episode_info = env.get_episode_info()
        results.append({
            'best_cost': episode_info['best_cost'],
            'total_reward': episode_reward,
            'steps': episode_info['total_steps']
        })
        
        print(f"  Best cost: {episode_info['best_cost']:.2f}")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Steps: {episode_info['total_steps']}")
    
    print("\n" + "="*60)
    print("Summary of All Episodes:")
    print("="*60)
    for i, result in enumerate(results):
        print(f"Episode {i+1}: Cost={result['best_cost']:.2f}, "
              f"Reward={result['total_reward']:.4f}, Steps={result['steps']}")
    
    print("\n✓ MULTIPLE EPISODES TEST PASSED!")
    print("="*60)


def test_state_features():
    """Test state feature extraction in detail."""
    print("\n\n")
    print("="*60)
    print("INTEGRATION TEST: State Feature Analysis")
    print("="*60)
    
    tsp = TSPProblem.create_random_instance(num_cities=15, seed=999)
    algorithms = [
        GeneticAlgorithm(tsp, population_size=20),
        SimulatedAnnealing(tsp, initial_temperature=100, cooling_rate=0.95),
        TabuSearch(tsp, tabu_tenure=8)
    ]
    
    env = OptimizationEnvironment(
        problem=tsp,
        algorithms=algorithms,
        max_evaluations=2000,
        schedule_interval=500
    )
    
    print("\nInitial State Features:")
    state = env.reset()
    print(f"  State shape: {state.shape}")
    print(f"  State min: {state.min():.4f}")
    print(f"  State max: {state.max():.4f}")
    print(f"  State mean: {state.mean():.4f}")
    print(f"  State std: {state.std():.4f}")
    
    # Take a few steps and observe state changes
    print("\nState Evolution:")
    for step in range(3):
        action = np.random.randint(0, env.get_action_space_size())
        next_state, reward, done, info = env.step(action)
        
        print(f"\n  After step {step + 1} (Algorithm: {info['algorithm_selected']}):")
        print(f"    State mean: {next_state.mean():.4f}")
        print(f"    State std: {next_state.std():.4f}")
        print(f"    State changed: {not np.allclose(state, next_state)}")
        print(f"    Best cost: {info['best_cost']:.2f}")
        
        state = next_state
        
        if done:
            break
    
    print("\n✓ STATE FEATURES TEST PASSED!")
    print("="*60)


if __name__ == "__main__":
    # Run all tests
    try:
        test_single_episode()
        test_multiple_episodes()
        test_state_features()
        
        print("\n\n" + "🎉"*30)
        print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("Ready to proceed to RL Agent development!")
        print("🎉"*30)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
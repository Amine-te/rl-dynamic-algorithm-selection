"""
Evaluate Stable-Baselines3 PPO model on TSP instances.
Compares both solution quality (cost) and execution time.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import time
from stable_baselines3 import PPO

from problems.tsp import TSPProblem
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.tabu_search import TabuSearch
from core.gym_wrapper import OptimizationGymEnv


def create_tsp_instance(num_cities: int, instance_type: str, seed: int) -> TSPProblem:
    if instance_type == "random":
        return TSPProblem.create_random_instance(num_cities, seed=seed)
    if instance_type == "clustered":
        num_clusters = max(2, num_cities // 20)
        return TSPProblem.create_clustered_instance(num_cities, num_clusters, seed=seed)
    # mixed
    if seed % 2 == 0:
        return TSPProblem.create_random_instance(num_cities, seed=seed)
    num_clusters = max(2, num_cities // 20)
    return TSPProblem.create_clustered_instance(num_cities, num_clusters, seed=seed)


def make_env(tsp: TSPProblem, config: dict):
    algorithms = [
        GeneticAlgorithm(
            tsp,
            population_size=config.get("ga_population_size", 50),
            mutation_rate=config.get("ga_mutation_rate", 0.1),
            crossover_rate=config.get("ga_crossover_rate", 0.8),
        ),
        SimulatedAnnealing(
            tsp,
            initial_temperature=config.get("sa_initial_temp", 100),
            cooling_rate=config.get("sa_cooling_rate", 0.95),
        ),
        TabuSearch(
            tsp,
            tabu_tenure=config.get("ts_tabu_tenure", 10),
        ),
    ]

    return OptimizationGymEnv(
        problem=tsp,
        algorithms=algorithms,
        max_evaluations=config.get("max_evaluations", 20000),
        schedule_interval=config.get("schedule_interval", 1000),
        reward_type=config.get("reward_type", "improvement_with_efficiency"),
    )


def rollout(model, env, deterministic: bool = True):
    start_time = time.time()
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    execution_time = time.time() - start_time
    episode_info = env.get_episode_info()
    return {
        "best_cost": episode_info["best_cost"],
        "execution_time": execution_time,
        "total_reward": total_reward,
        "total_steps": steps,
        "selection_counts": env.get_algorithm_selection_counts(),
    }


def evaluate_baseline(tsp: TSPProblem, alg, config: dict):
    start_time = time.time()
    alg.initialize()
    max_evals = config.get("max_evaluations", 20000)

    # Run algorithm for full evaluation budget at once
    _, best_cost = alg.step(max_evals)
    execution_time = time.time() - start_time

    return best_cost, execution_time


def main():
    parser = argparse.ArgumentParser(description="Evaluate SB3 PPO model")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, default="best_model/best_model.zip")
    parser.add_argument("--num_test_instances", type=int, default=20)
    parser.add_argument("--num_cities", type=int, default=None)
    parser.add_argument("--instance_type", type=str, default=None,
                        choices=["random", "clustered", "mixed"])
    parser.add_argument("--test_seed", type=int, default=9999)
    parser.add_argument("--compare_baselines", action="store_true", default=True)
    args = parser.parse_args()

    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    if args.num_cities is not None:
        config["num_cities"] = args.num_cities
    if args.instance_type is not None:
        config["instance_type"] = args.instance_type

    num_cities = config["num_cities"]
    instance_type = config.get("instance_type", "mixed")

    model_path = os.path.join(args.checkpoint_dir, args.model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("=" * 60)
    print("EVALUATING SB3 PPO MODEL")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Instances: {args.num_test_instances}")
    print(f"TSP size: {num_cities}")
    print(f"Instance type: {instance_type}")
    print("=" * 60)

    model = PPO.load(model_path)

    # Evaluate RL-DAS
    rl_costs = []
    rl_times = []
    for i in range(args.num_test_instances):
        tsp = create_tsp_instance(num_cities, instance_type, args.test_seed + i)
        env = make_env(tsp, config)
        result = rollout(model, env, deterministic=True)
        rl_costs.append(result["best_cost"])
        rl_times.append(result["execution_time"])
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{args.num_test_instances} instances")

    rl_mean_cost = float(np.mean(rl_costs))
    rl_std_cost = float(np.std(rl_costs))
    rl_min_cost = float(np.min(rl_costs))
    rl_max_cost = float(np.max(rl_costs))
    rl_mean_time = float(np.mean(rl_times))
    rl_std_time = float(np.std(rl_times))

    print("\n" + "=" * 60)
    print("RL-DAS RESULTS")
    print("=" * 60)
    print(f"Mean Best Cost: {rl_mean_cost:.2f} ± {rl_std_cost:.2f}")
    print(f"Min / Max Cost: {rl_min_cost:.2f} / {rl_max_cost:.2f}")
    print(f"Mean Execution Time: {rl_mean_time:.3f}s ± {rl_std_time:.3f}s")

    results = {
        "rl_results": {
            "mean_cost": rl_mean_cost,
            "std_cost": rl_std_cost,
            "min_cost": rl_min_cost,
            "max_cost": rl_max_cost,
            "mean_time": rl_mean_time,
            "std_time": rl_std_time,
            "all_costs": [float(c) for c in rl_costs],
            "all_times": [float(t) for t in rl_times],
        }
    }

    # Baselines
    if args.compare_baselines:
        print("\n" + "=" * 60)
        print("BASELINE COMPARISON")
        print("=" * 60)

        baseline_costs = {"GA": [], "SA": [], "TS": []}
        baseline_times = {"GA": [], "SA": [], "TS": []}
        for i in range(args.num_test_instances):
            tsp = create_tsp_instance(num_cities, instance_type, args.test_seed + 10_000 + i)

            ga = GeneticAlgorithm(tsp, population_size=config.get("ga_population_size", 50))
            sa = SimulatedAnnealing(
                tsp,
                initial_temperature=config.get("sa_initial_temp", 100),
                cooling_rate=config.get("sa_cooling_rate", 0.95),
            )
            ts = TabuSearch(tsp, tabu_tenure=config.get("ts_tabu_tenure", 10))

            cost_ga, time_ga = evaluate_baseline(tsp, ga, config)
            cost_sa, time_sa = evaluate_baseline(tsp, sa, config)
            cost_ts, time_ts = evaluate_baseline(tsp, ts, config)

            baseline_costs["GA"].append(cost_ga)
            baseline_costs["SA"].append(cost_sa)
            baseline_costs["TS"].append(cost_ts)
            baseline_times["GA"].append(time_ga)
            baseline_times["SA"].append(time_sa)
            baseline_times["TS"].append(time_ts)

        print("Baseline Cost Results:")
        for name, costs in baseline_costs.items():
            print(f"  {name}: mean={np.mean(costs):.2f} ± {np.std(costs):.2f}, min={np.min(costs):.2f}")

        print("\nBaseline Time Results:")
        for name, times in baseline_times.items():
            print(f"  {name}: mean={np.mean(times):.3f}s ± {np.std(times):.3f}s")

        best_baseline_mean = min(np.mean(c) for c in baseline_costs.values())
        fastest_baseline_mean = min(np.mean(t) for t in baseline_times.values())

        cost_improvement = ((best_baseline_mean - rl_mean_cost) / best_baseline_mean) * 100
        time_overhead = ((rl_mean_time - fastest_baseline_mean) / fastest_baseline_mean) * 100

        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print("Cost Comparison:")
        print(f"  RL-DAS Mean:        {rl_mean_cost:.2f}")
        print(f"  Best Baseline:      {best_baseline_mean:.2f}")
        print(f"  Cost Improvement:   {cost_improvement:+.2f}%")
        print("Time Comparison:")
        print(f"  RL-DAS Mean Time:   {rl_mean_time:.3f}s")
        print(f"  Fastest Baseline:   {fastest_baseline_mean:.3f}s")
        print(f"  Time Overhead:      {time_overhead:+.2f}%")

        results["baseline_results"] = {
            k: {
                "mean_cost": float(np.mean(baseline_costs[k])),
                "std_cost": float(np.std(baseline_costs[k])),
                "min_cost": float(np.min(baseline_costs[k])),
                "mean_time": float(np.mean(baseline_times[k])),
                "std_time": float(np.std(baseline_times[k])),
                "all_costs": [float(c) for c in baseline_costs[k]],
                "all_times": [float(t) for t in baseline_times[k]],
            }
            for k in baseline_costs.keys()
        }

    results_path = os.path.join(args.checkpoint_dir, "evaluation_results_sb3.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()


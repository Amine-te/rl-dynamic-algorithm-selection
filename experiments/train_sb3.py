"""
Train RL-DAS with Stable-Baselines3 PPO.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import ast
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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
    def _init():
        algorithms = [
            GeneticAlgorithm(
                tsp, population_size=config.get("ga_population_size", 50),
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

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train RL-DAS with SB3 PPO")

    # Training
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--num_cities", type=int, default=50)
    parser.add_argument("--instance_type", type=str, default="mixed",
                        choices=["random", "clustered", "mixed"])

    # Environment
    parser.add_argument("--max_evaluations", type=int, default=20_000)
    parser.add_argument("--schedule_interval", type=int, default=1_000)
    parser.add_argument("--reward_type", type=str, default="improvement_with_efficiency",
                        choices=["improvement", "improvement_with_efficiency", "normalized_improvement"])

    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--net_arch", type=str, default="[64, 64]",
                        help="Network architecture, e.g., \"[128,64]\"")

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/tsp_sb3")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--eval_freq", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    net_arch = ast.literal_eval(args.net_arch)

    # Config for reproducibility
    config = {
        "num_cities": args.num_cities,
        "instance_type": args.instance_type,
        "max_evaluations": args.max_evaluations,
        "schedule_interval": args.schedule_interval,
        "reward_type": args.reward_type,
        "ga_population_size": 50,
        "ga_mutation_rate": 0.1,
        "ga_crossover_rate": 0.8,
        "sa_initial_temp": 100,
        "sa_cooling_rate": 0.95,
        "ts_tabu_tenure": 10,
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training and eval envs (same distribution, different seeds)
    train_tsp = create_tsp_instance(args.num_cities, args.instance_type, args.seed)
    eval_tsp = create_tsp_instance(args.num_cities, args.instance_type, args.seed + 1000)

    train_env = DummyVecEnv([make_env(train_tsp, config)])
    eval_env = DummyVecEnv([make_env(eval_tsp, config)])

    print("=" * 60)
    print("TRAINING RL-DAS WITH STABLE-BASELINES3 PPO")
    print("=" * 60)
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    print(f"Network architecture: {net_arch}")
    print("=" * 60)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs={"net_arch": net_arch},
        tensorboard_log=os.path.join(args.checkpoint_dir, "tensorboard"),
        verbose=args.verbose,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.checkpoint_dir, "checkpoints"),
        name_prefix="ppo_model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best_model"),
        log_path=os.path.join(args.checkpoint_dir, "logs"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(args.checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\n✓ Training completed")
    print(f"✓ Final model saved: {final_path}.zip")
    print(f"✓ Best model saved in: {os.path.join(args.checkpoint_dir, 'best_model')}")
    print(f"✓ TensorBoard logs: {os.path.join(args.checkpoint_dir, 'tensorboard')}")


if __name__ == "__main__":
    main()


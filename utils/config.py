"""
Configuration management for RL-DAS.
Provides default configurations and utilities for loading/saving configs.
"""

import json
import os
from typing import Dict, Any


# Default PPO configuration
DEFAULT_PPO_CONFIG = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'update_epochs': 4
}

# Default network configuration
DEFAULT_NETWORK_CONFIG = {
    'hidden_dims': [64, 32],
    'activation': 'tanh',
    'value_activation': 'relu'
}

# Default environment configuration
DEFAULT_ENV_CONFIG = {
    'max_evaluations': 20000,
    'train_max_evaluations': 20000,  # For training (can be shorter)
    'schedule_interval': 1000,
    'reward_type': 'improvement_with_efficiency',
    'num_algorithms': 3,
    'feature_sample_size': 50
}

# Default training configuration
DEFAULT_TRAIN_CONFIG = {
    'num_epochs': 200,
    'instances_per_epoch': 16,
    'episodes_per_instance': 1,
    'validation_freq': 10,
    'checkpoint_freq': 20,
    'early_stopping_patience': 30
}

# Default algorithm configuration
DEFAULT_ALGORITHM_CONFIG = {
    'use_ga': True,
    'use_sa': True,
    'use_ts': True,
    'ga_population_size': 50,
    'ga_mutation_rate': 0.1,
    'ga_crossover_rate': 0.8,
    'sa_initial_temp': 100,
    'sa_cooling_rate': 0.95,
    'ts_tabu_tenure': 10
}

# Default problem configuration
DEFAULT_PROBLEM_CONFIG = {
    'num_cities': 50,
    'instance_type': 'mixed',  # 'random', 'clustered', 'mixed'
    'train_seed': 42,
    'test_seed': 9999
}


def get_default_config() -> Dict[str, Any]:
    """
    Get complete default configuration.
    
    Returns:
        Dictionary with all default configurations
    """
    return {
        'ppo': DEFAULT_PPO_CONFIG.copy(),
        'network': DEFAULT_NETWORK_CONFIG.copy(),
        'env': DEFAULT_ENV_CONFIG.copy(),
        'train': DEFAULT_TRAIN_CONFIG.copy(),
        'algorithm': DEFAULT_ALGORITHM_CONFIG.copy(),
        'problem': DEFAULT_PROBLEM_CONFIG.copy()
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_tsp_config(num_cities: int = 50, instance_type: str = 'mixed') -> Dict[str, Any]:
    """
    Get configuration for TSP training.
    
    Args:
        num_cities: Number of cities
        instance_type: Type of instances ('random', 'clustered', 'mixed')
        
    Returns:
        Configuration dictionary
    """
    config = get_default_config()
    config['problem']['num_cities'] = num_cities
    config['problem']['instance_type'] = instance_type
    return config


def get_small_config() -> Dict[str, Any]:
    """Get configuration for quick testing (small instances, fewer epochs)."""
    config = get_default_config()
    config['problem']['num_cities'] = 20
    config['env']['max_evaluations'] = 5000
    config['env']['schedule_interval'] = 500
    config['train']['num_epochs'] = 50
    config['train']['instances_per_epoch'] = 8
    return config


def get_large_config() -> Dict[str, Any]:
    """Get configuration for full training (larger instances, more epochs)."""
    config = get_default_config()
    config['problem']['num_cities'] = 75
    config['env']['max_evaluations'] = 100000
    config['env']['schedule_interval'] = 2500
    config['train']['num_epochs'] = 300
    config['train']['instances_per_epoch'] = 32
    return config


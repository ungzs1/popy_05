"""
Run model performance simulations and statistical comparisons.

This script:
1. Simulates 100 runs of 100,000 trials for each model
2. Computes reward rates
3. Performs Mann-Whitney U tests for specific model pairs
4. Saves results to CSV files
"""

import os
import gymnasium as gym
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from popy.simulation_tools import *
from popy.simulation_helpers import simulate_agent
from popy.config import PROJECT_PATH_LOCAL


MODELS = {
    "WSLS": {
        "agent_class": WSLSAgent,
        "fixed_params": {},
        "free_params": ["epsilon"],
    },
    "Standard RL - stickiness": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": False},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
    "Inferential RL - stickiness": {    
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": True},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
    "Foraging": {
        "agent_class": ForagingAgent,
        "fixed_params": {"reset_on_switch": True},
        "free_params": ["alpha", "beta", "V0"],
    },
    "HSMMAgent": {
        "agent_class": HSMMAgent,
        "fixed_params": {},
        "free_params": [],
    },
}


def run_simulations(n_trials=100_000, n_reps=100):
    """Run simulations for all models and collect reward rates."""
    
    # Load best parameters
    floc = os.path.join(PROJECT_PATH_LOCAL, 'notebooks', 'behav_modeling', 'results', 'optimizing', 'optimization_results.csv')
    best_params = pd.read_csv(floc)
    
    # Create environment
    env = gym.make(
        "zsombi/monkey-bandit-task-v0", 
        n_arms=3, 
        max_episode_steps=n_trials,
    )
    
    # Run simulations
    model_performances = {}
    performance_records = []
    
    for model_name in MODELS.keys():
        print(f"\nSimulating {model_name}...")
        model_performances[model_name] = []
        
        model_params = MODELS[model_name]
        agent_class = model_params["agent_class"]
        fixed_params = model_params["fixed_params"]
        
        # Get parameters for this model
        params = {}
        for param in model_params["free_params"]:
            if param not in best_params.columns:
                raise ValueError(f"Parameter {param} not found in best_params DataFrame.")
            params[param] = best_params.loc[best_params['Model'] == model_name, param].values[0]
        
        for rep in tqdm(range(n_reps)):
            # Simulate behavior with best parameters
            behav = simulate_agent(
                agent_class=agent_class, 
                params=params,
                env=env, 
                fixed_params=fixed_params, 
                behavioral_variables=[],
                verbose=False
            )
            reward_rate = behav['reward'].mean()
            
            model_performances[model_name].append(reward_rate)
            performance_records.append({
                'Model': model_name,
                'Replication': rep,
                'Reward_Rate': reward_rate
            })
    
    # Save performance data
    perf_df = pd.DataFrame(performance_records)
    perf_path = os.path.join(PROJECT_PATH_LOCAL, 'notebooks', 'behav_modeling', 'results', 'optimizing', 'statistics_performances.csv')
    perf_df.to_csv(perf_path, index=False)
    print(f"\nSaved performance data to {perf_path}")
    
    return model_performances


def main():
    """Main execution function."""
    print("=" * 80)
    print("Running Model Performance Simulations and Statistical Tests")
    print("=" * 80)
    
    # Run simulations
    run_simulations(n_trials=100_000, n_reps=100)
    
if __name__ == "__main__":
    main()

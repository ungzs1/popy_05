import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from skopt.space import Real
from multiprocessing import Pool, cpu_count

# Import agents and utilities from the project
from popy.simulation_tools import *  # Provides agents like ForagingAgent, QLearner
from popy.simulation_helpers import simulate_agent, fit_simulate
from popy.config import PROJECT_PATH_LOCAL

ANALYSIS_PARAMETERS = {
    'T': 10_000,  # Total trials per simulation
    'N': 150,     # Number of simulations to run
}

def make_env():
    """Create the monkey-bandit environment used for simulations."""
    return gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=ANALYSIS_PARAMETERS['T'])

def _run_foraging_simulation(n):
    """Helper function to run a single foraging parameter recovery iteration."""
    start_time = time.time()
    print(f"[Foraging] Progress: {n+1}/{ANALYSIS_PARAMETERS['N']}")
    
    # Create environment for this process
    env = make_env()
    
    # Search spaces used by the optimizer
    alpha_range = Real(0.01, .7, name='alpha')
    beta_range = Real(0.01, 50.0, name='beta')
    V0_range = Real(0.05, 0.7, name='V0')
    param_space = [alpha_range, beta_range, V0_range]

    params_true = {
        'alpha': np.random.uniform(0.01, .7),
        'beta': np.random.exponential(7),
        'V0': np.random.uniform(0.05, 0.7),
    }
    
    agent_class = ForagingAgent
    fixed_params = {'reset_on_switch': True}

    behavior_simulated = simulate_agent(
        agent_class=agent_class,
        params=params_true,
        env=env,
        fixed_params=fixed_params,
    )
    behavior_simulated['monkey'] = 'simulated_agent'
    behavior_simulated['session'] = 0

    results, _ = fit_simulate(
        agent_class,
        param_space,
        env,
        behavior_simulated,
        fixed_params,
        CV_splits=None,
        strict=False,
        make_plots=False,
        n_calls=200,
        n_initial_points=100,
        n_jobs=1,
        verbose=False,
    )

    params_recovered = {
        'alpha': results['alpha'],
        'beta': results['beta'],
        'V0': results['V0'],
    }

    result_dict = {
        'alpha_true': params_true['alpha'],
        'beta_true': params_true['beta'],
        'V0_true': params_true['V0'],
        'alpha_recovered': params_recovered['alpha'],
        'beta_recovered': params_recovered['beta'],
        'V0_recovered': params_recovered['V0'],
        'LPT_best': results['LPT_best'],
    }

    elapsed_time = time.time() - start_time
    print(
        f"True: alpha={params_true['alpha']:.2f}, beta={params_true['beta']:.2f}, V0={params_true['V0']:.2f} | "
        f"Recovered: alpha={params_recovered['alpha']:.2f}, beta={params_recovered['beta']:.2f}, V0={params_recovered['V0']:.2f} | "
        f"{elapsed_time:.2f}s"
    )
    
    return result_dict


def generate_foraging_parameter_recovery(env, output_csv_path: str):
    """
    Run parameter recovery for the Foraging agent across a grid of true parameters.
    Saves a CSV with true vs recovered parameters and summary metrics.
    """
    print(f"[Foraging] Running {ANALYSIS_PARAMETERS['N']} simulations on {cpu_count()} CPUs")
    
    with Pool() as pool:
        recovery_results = pool.map(_run_foraging_simulation, range(ANALYSIS_PARAMETERS['N']))

    recovery_df = pd.DataFrame(recovery_results)
    recovery_df.to_csv(output_csv_path, index=False)
    print(f"[Foraging] Completed {len(recovery_results)} simulations. Saved to {output_csv_path}")


def _run_irl_stickiness_simulation(n):
    """Helper function to run a single IRL stickiness parameter recovery iteration."""
    start_time = time.time()
    print(f"[IRL-stickiness] Progress: {n+1}/{ANALYSIS_PARAMETERS['N']}")
    
    # Create environment for this process
    env = make_env()
    
    # Search spaces used by the optimizer
    alpha_range = Real(0.01, 1, name='alpha')
    beta_range = Real(.5, 50.0, name='beta')
    stickyness_range = Real(0.0, 10.0, name='stickyness_bias')
    param_space = [alpha_range, beta_range, stickyness_range]

    params_true = {
        'alpha': np.random.uniform(0.01, 1),
        'beta': np.random.exponential(7),
        'stickyness_bias': np.random.exponential(2),
    }

    agent_class = QLearner
    fixed_params = {'structure_aware': True}

    behavior_simulated = simulate_agent( 
        agent_class=agent_class,
        params=params_true,
        env=env,
        fixed_params=fixed_params,
    )
    behavior_simulated['monkey'] = 'simulated_agent'
    behavior_simulated['session'] = 0

    results, _ = fit_simulate(
        agent_class,
        param_space,
        env,
        behavior_simulated,
        fixed_params,
        CV_splits=None,
        strict=False,
        make_plots=False,
        n_calls=200,
        n_initial_points=100,
        n_jobs=1,
        verbose=False,
    )

    params_recovered = {
        'alpha': results['alpha'],
        'beta': results['beta'],
        'stickyness_bias': results['stickyness_bias'],
    }

    result_dict = {
        'alpha_true': params_true['alpha'],
        'beta_true': params_true['beta'],
        'stickyness_true': params_true['stickyness_bias'],
        'alpha_recovered': params_recovered['alpha'],
        'beta_recovered': params_recovered['beta'],
        'stickyness_recovered': params_recovered['stickyness_bias'],
        'LPT_best': results['LPT_best'],
    }

    elapsed_time = time.time() - start_time
    print(
        f"True: alpha={params_true['alpha']:.2f}, beta={params_true['beta']:.2f}, stickyness={params_true['stickyness_bias']:.2f} | "
        f"Recovered: alpha={params_recovered['alpha']:.2f}, beta={params_recovered['beta']:.2f}, stickyness={params_recovered['stickyness_bias']:.2f} | "
        f"{elapsed_time:.2f}s"
    )
    
    return result_dict


def generate_irl_stickiness_parameter_recovery(env, output_csv_path: str):
    """
    Run parameter recovery for an Inferential RL (QLearner) with stickiness.
    Saves a CSV with true vs recovered parameters and summary metrics.
    """
    print(f"[IRL-stickiness] Running {ANALYSIS_PARAMETERS['N']} simulations on {cpu_count()} CPUs")
    
    with Pool() as pool:
        recovery_results = pool.map(_run_irl_stickiness_simulation, range(ANALYSIS_PARAMETERS['N']))

    recovery_df = pd.DataFrame(recovery_results)
    recovery_df.to_csv(output_csv_path, index=False)
    print(f"[IRL-stickiness] Completed {len(recovery_results)} simulations. Saved to {output_csv_path}")


if __name__ == "__main__":
    env = make_env()
    generate_foraging_parameter_recovery(
        env,
        output_csv_path=os.path.join(PROJECT_PATH_LOCAL, "notebooks", "behav_modeling", "results", "parameter_recovery_foraging.csv"),
    )

    '''generate_irl_stickiness_parameter_recovery(
        env,
        output_csv_path=os.path.join(PROJECT_PATH_LOCAL, "notebooks", "behav_modeling", "results", "parameter_recovery_inferential_rl_stickyness.csv"),
    )'''

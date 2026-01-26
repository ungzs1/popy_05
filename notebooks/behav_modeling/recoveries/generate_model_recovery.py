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
    'T': 1_000,  # Total trials per simulation
    'N': 2,     # Number of simulations to run
}

MODELS = {
    0: 'Repeating agent', 
    1: 'Standard RL', 
    2: 'Standard RL - stickyness', 
    3: 'Inferential RL', 
    4: 'Inferential RL - stickyness', 
    5: 'Foraging'
}

n_initial_points = 200
n_calls = 350

epsilon_range = Real(.01, .3, name='epsilon')
alpha_range = Real(0.01, 1, name='alpha')
beta_range = Real(.5, 80.0, name='beta')
stickiness_range = Real(0.0, 50.0, name='stickiness_bias')
V0_range = Real(0.05, .4, name='V0')


def _get_model_params_and_class(model_index, env, seed=None):
    """Helper function to get agent class, true parameters, fixed parameters, and parameter space based on model index."""
    if seed is not None:
        np.random.seed(seed)
    
    if model_index == 0:
        agent_class = RepeatingAgent
        fixed_params = {}
        params_true = None
        param_space = [epsilon_range]
    elif model_index == 1:
        agent_class = QLearner
        fixed_params = {'structure_aware': False}
        params_true = {
            'alpha': np.random.uniform(0.01, .7),
            'beta': np.random.exponential(7),
        }
        param_space = [alpha_range, beta_range]
    elif model_index == 2:
        agent_class = QLearner
        fixed_params = {'structure_aware': False}
        params_true = {
            'alpha': np.random.uniform(0.01, .7),
            'beta': np.random.exponential(7),
            'stickyness_bias': np.random.uniform(0.00, 3),
        }
        param_space = [alpha_range, beta_range, stickiness_range]
    elif model_index == 3:
        agent_class = QLearner
        fixed_params = {'structure_aware': True}
        params_true = {
            'alpha': np.random.uniform(0.01, .7),
            'beta': np.random.exponential(7),
        }
        param_space = [alpha_range, beta_range]
    elif model_index == 4:
        agent_class = QLearner
        fixed_params = {'structure_aware': True}
        params_true = {
            'alpha': np.random.uniform(0.01, .7),
            'beta': np.random.exponential(7),
            'stickyness_bias': np.random.uniform(0.00, 3),
        }
        param_space = [alpha_range, beta_range, stickiness_range]
    elif model_index == 5:
        agent_class = ForagingAgent
        fixed_params = {'reset_on_switch': True}
        params_true = {
            'alpha': np.random.uniform(0.01, .7),
            'beta': np.random.exponential(7),
            'V0': np.random.uniform(0.05, 0.7),
        }
        param_space = [alpha_range, beta_range, V0_range]

    else:
        raise ValueError(f"Unknown model index: {model_index}")
    
    return agent_class, params_true, fixed_params, param_space

    

# Search spaces used by the optimizer
alpha_range = Real(0.01, .7, name='alpha')
beta_range = Real(0.01, 50.0, name='beta')
V0_range = Real(0.05, 0.7, name='V0')
stickiness_range = Real(0.0, 10.0, name='stickyness_bias')
epsilon_range = Real(0.0001, 1.0, name='epsilon')

def make_env():
    """Create the monkey-bandit environment used for simulations."""
    return gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=ANALYSIS_PARAMETERS['T'])

def _run_model_recovery(sim_model_index, fit_model_index, iteration):
    """Helper function to run a single model recovery iteration.
    
    sim_model_index: index of the true model used for simulation
    fit_model_index: index of the model used for fitting
    iteration: iteration number (used to generate same simulation for all fit models)
    """
    start_time = time.time()
    print(f"[True model: {MODELS[sim_model_index]}] -- [Fit model: {MODELS[fit_model_index]}], iteration {iteration+1}/{ANALYSIS_PARAMETERS['N']}")
    
    # Create environment for this process0
    env = make_env()

    # Use seed based on sim_model_index and iteration to ensure same simulation for all fit models
    sim_seed = sim_model_index * 1000 + iteration
    sim_agent_class, params_true, sim_fixed_params, _ = _get_model_params_and_class(sim_model_index, env, seed=sim_seed)
    fit_agent_class, _, fit_fixed_params, param_space = _get_model_params_and_class(fit_model_index, env)

    behavior_simulated = simulate_agent(
        agent_class=sim_agent_class,
        params=params_true,
        env=env,
        fixed_params=sim_fixed_params,
    )
    behavior_simulated['monkey'] = 'simulated_agent'
    behavior_simulated['session'] = 0

    # Fit the specified model to the simulated data
    results, _ = fit_simulate(
        fit_agent_class,
        param_space,
        env,
        behavior_simulated,
        fixed_params=fit_fixed_params,
        CV_splits=None,
        strict=False,
        make_plots=False,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        n_jobs=1,
        verbose=False,
    )

    result_dict = {
        'model_true': MODELS[sim_model_index],
        'model_fitted': MODELS[fit_model_index],
        'iteration': iteration,
        **(params_true if params_true is not None else {}),
        'LPT_best': results['LPT_best'],
    }

    elapsed_time = time.time() - start_time
    print(
        f"{elapsed_time:.2f}s"
    )
    
    return result_dict


def generate_model_recovery(env, output_csv_path: str):
    """
    Run model recovery for the models (all combinations).
    Saves a CSV with LPT for each run for each true model (used for simulation) and each fitted model (used for fitting). Also saves the true parameters of the true model.
    """
    print(f"[Model recovery] Running {ANALYSIS_PARAMETERS['N']} simulations on {cpu_count()} CPUs")
    
    # Create task list: for each iteration, fit all models to the same simulation
    tasks = []
    for iteration in range(ANALYSIS_PARAMETERS['N']):
        for sim_model_index in range(len(MODELS)):
            for fit_model_index in range(len(MODELS)):
                tasks.append((sim_model_index, fit_model_index, iteration))
    
    with Pool() as pool:
        recovery_results = pool.starmap(_run_model_recovery, tasks)

    recovery_df = pd.DataFrame(recovery_results)
    recovery_df.to_csv(output_csv_path, index=False)
    print(f"[Model recovery] Completed {len(recovery_results)} simulations. Saved to {output_csv_path}")


if __name__ == "__main__":
    env = make_env()
    generate_model_recovery(
        env,
        output_csv_path=os.path.join(PROJECT_PATH_LOCAL, "notebooks", "behav_modeling", "results", "model_recovery_foraging.csv"),
    )

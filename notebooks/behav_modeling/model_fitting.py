# # Summary
# 
# Lets fit the models to the actual data.

# ---
# # Setup

# @title Imports
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations



from popy.simulation_tools import *
from popy.io_tools import load_behavior, load_behavior_yuri
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *

from simulation_helpers import simulate_agent, estimate_ll, fit_simulate


# data loading

def get_data_custom(monkey):
    if monkey in ['ka', 'po']:
        behav_monkey = load_behavior(monkey)
        behav_monkey = drop_time_fields(behav_monkey)
    elif monkey in ['yu_sham', 'yu_DCZ']:
        behav_monkey = load_behavior_yuri()
        behav_monkey = behav_monkey.loc[behav_monkey['monkey'] == monkey]
    else:
        raise ValueError(f'Unknown monkey: {monkey}')
    behav_monkey = add_switch_info(behav_monkey)
    behav_monkey = convert_column_format(behav_monkey, original='behavior')

    behav_monkey = behav_monkey.dropna()

    return behav_monkey

make_plots = False

for monkey in ['ka', 'po']: #yu_sham', 'yu_DCZ']:  # ['ka', 'po', 'yu_sham', 'yu_dcz']
    print(f'--- {monkey} ---')

    behav_monkey = get_data_custom(monkey)
    #behav_monkey = pd.read_pickle('results/behavior_strategic_reset.pkl')

    behav_monkey

    # ## Create environment

    # Create the environment
    env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=100_000)

    # Set container (to collect pandas series into a dataframe)
    results = {}
    behaviors_simulated = {f'MONKEY {monkey.upper()}': behav_monkey}




    # ---
    # # Fit models to the data

    # ## 0. Baselines

    # ### Repeating agent
    # 
    # An agent that repeats the previous action. Uses Softmax to choose the action.

    # Define parameter space for ShiftValueAgent
    model_name = 'Repeating agent'
    agent_class = RepeatingAgent

    param_space = [
        Real(0.01, 0.5, name='epsilon'),
    ]

    fixed_params = {}

    # fit agent, get best params and simulation
    res_temp, behavior_temp = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=3, make_plots=make_plots, n_calls=60, n_initial_points=10, n_jobs=1)
    results[model_name] = res_temp
    behaviors_simulated[model_name] = behavior_temp




    # ### Simple WSLS

    # ### Modified WSLS

    # Define parameter space for ShiftValueAgent
    model_name = 'WSLS agent (long history)'
    agent_class = WSLSAgent_custom

    param_space = [
        Real(.01, .3, name='epsilon'),
    ]

    fixed_params = {}

    res_temp, behavior_temp = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=3, make_plots=make_plots, n_calls=60, n_initial_points=10)
    results[model_name] = res_temp
    behaviors_simulated[model_name] = behavior_temp

    # ## 1. RL

    # ### Simple RL agent

    # Define parameter space for ShiftValueAgent
    agent_class = QLearner
    model_name = 'Q-Learner'

    param_space = [
        Real(0.3, .8, name='alpha'),
        Real(2, 20.0, name='beta'),
    ]

    fixed_params = {
        'structure_aware': False
    }

    res_temp, behavior_temp = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=3, make_plots=make_plots, n_calls=150, n_initial_points=100)
    results[model_name] = res_temp
    behaviors_simulated[model_name] = behavior_temp




    # ### Counterfactual

    # Define parameter space for ShiftValueAgent
    agent_class = QLearner
    model_name = 'Q-Learner counterfactual'

    param_space = [
        Real(0.05, .5, name='alpha'),
        Real(2, 20.0, name='beta'),
    ]

    fixed_params = {
        'structure_aware': True
    }

    res_temp, behavior_temp = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=3, make_plots=make_plots, n_calls=150, n_initial_points=100)
    results[model_name] = res_temp
    behaviors_simulated[model_name] = behavior_temp




    # ### Multiple learning rates

    '''# Define parameter space for ShiftValueAgent
    agent_class = QLearner
    model_name = 'Q-Learner multiple alphas'

    param_space = [
        Real(0.01, .6, name='alpha'),
        Real(0.01, .3, name='alpha_unchosen'),
        Real(3, 20.0, name='beta'),
    ]

    fixed_params = {
        'structure_aware': True
    }

    res_temp, behav_q_learn_multiple = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, 
                                                    model_name, make_plots=True,  verbose=False,
                                                n_calls=150, n_initial_points=100)
    results_list.append(res_temp)
    behaviors_simulated[model_name] = behav_q_learn_multiple'''




    # ## 2. Shift value

    # ### No reset

    # Define parameter space for ShiftValueAgent
    agent_class = ShiftValueAgent
    model_name = 'Shift-value agent'

    param_space = [
        Real(0.2, 0.7, name='alpha'),
        Real(2, 20.0, name='beta'),
        Real(0.05, .4, name='V0')
    ]

    fixed_params = {
        'reset_on_switch': False
    }

    res_temp, behavior_temp = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=3, make_plots=make_plots, n_calls=150, n_initial_points=100)
    results[model_name] = res_temp
    behaviors_simulated[model_name] = behavior_temp




    # ### Reset

    # Define parameter space for ShiftValueAgent
    agent_class = ShiftValueAgent
    model_name = 'Shift-value agent with reset'

    param_space = [
        Real(0.1, 0.7, name='alpha'),
        Real(1, 20.0, name='beta'),
        Real(0.05, .4, name='V0')
    ]

    fixed_params = {
        'reset_on_switch': True
    }

    res_shift_threshold, behavior_temp = fit_simulate(agent_class, param_space, env, behav_monkey, fixed_params, CV_splits=3, make_plots=make_plots, n_calls=150, n_initial_points=100)
    results[model_name] = res_shift_threshold
    behaviors_simulated[model_name] = behavior_temp

    # ---
    # ### Save Shift-value simulation




    # ## 3. HMM
    # 
    # The bayes has the best parameters in principle. Also we dont go with this model, and it has a lot of parameters.

    # ---
    # # Save 




    # Generate simulation data

    params = {'alpha': res_shift_threshold['alpha'], 'beta': res_shift_threshold['beta'], 'V0': res_shift_threshold['V0']}

    fixed_params = {'reset_on_switch': True}
    simulation = simulate_agent(ShiftValueAgent,
                                params=params,
                                env=env,
                                fixed_params={'reset_on_switch': True},
                                behavioral_variables=['V'],
                                n_trials=len(behav_monkey))
                                
    simulation = convert_column_format(simulation, original='simulation')

    simulation['monkey'] = f'{monkey}_simulation'  # add change from 'simulation' to 'simulation_monkey' for plotting
    simulation = simulation.rename(columns={'V': 'stay_value'})

    # add metadata to the behav dataframe
    simulation.attrs['model'] = 'Shift-value agent with reset'
    simulation.attrs['parameters'] = params
    simulation.attrs['fixed_parameters'] = fixed_params

    from popy.config import PROJECT_PATH_LOCAL
    floc = os.path.join(PROJECT_PATH_LOCAL, 'data', 'processed', 'behavior', f'{monkey}_simulation.pkl')
    simulation.to_pickle(floc)
    print(f'Behavior simulation saved to {floc}')

    # save fitting results

    res_all = pd.DataFrame.from_dict(results, orient='index').reset_index()
    res_all = res_all.rename(columns={'index': 'Model'})

    # reorder columns: [Model, epsilon, alpha, beta, V0, etc]
    columns = res_all.columns
    new_columns = ['Model', 'epsilon', 'alpha', 'beta', 'V0'] + [col for col in columns if col not in ['Model', 'epsilon', 'alpha', 'beta', 'V0']]
    res_all = res_all[new_columns]

    # remove Model=Q-Learner multiple alphas
    #res_all = res_all.loc[res_all['Model'].isin(['WSLS agent (long history)', 'Q-Learner', 'Q-Learner counterfactual', 'Shift-value agent with reset'])]

    # save results
    from popy.config import PROJECT_PATH_LOCAL
    floc = os.path.join(PROJECT_PATH_LOCAL, 'notebooks', 'behav_modeling', 'results', f'simulation_results_{monkey}.csv')
    res_all.to_csv(floc)
    print(f'Saved results to {floc}')

    # save behaviors

    behavs = []
    bahv_monkey = behaviors_simulated[f'MONKEY {monkey.upper()}']
    behav_monkey = bahv_monkey.drop(columns=['switch'])
    behav_monkey['model'] = 'recording'
    behavs.append(behav_monkey)

    for key, behav_temo in behaviors_simulated.items():
        if key != f'MONKEY {monkey.upper()}':
            behav_temo['monkey'] = monkey
            behav_temo['session'] = 0
            behav_temo['model'] = key
            behavs.append(behav_temo)

    behaviors_simulated_all = pd.concat(behavs, axis=0)
    cols = ['monkey', 'model', 'session'] + [col for col in behav_monkey.columns if col not in ['monkey', 'session', 'model']]
    behaviors_simulated_all = behaviors_simulated_all[cols]

    # reset index
    behaviors_simulated_all = behaviors_simulated_all.reset_index(drop=True)

    # save
    from popy.config import PROJECT_PATH_LOCAL
    floc = os.path.join(PROJECT_PATH_LOCAL, 'notebooks', 'behav_modeling', 'results', f'simulation_behaviors_{monkey}.pkl')
    behaviors_simulated_all.to_pickle(floc)
    print(f'Behaviors saved to {floc}')



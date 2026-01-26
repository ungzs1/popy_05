import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from skopt import gp_minimize, dummy_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations


from skopt import gp_minimize, dummy_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

from popy.config import PROJECT_PATH_LOCAL
from popy.simulation_tools import *
from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *

from popy.simulation_helpers import *

def res_to_dataframe(results):
    res_all = pd.DataFrame.from_dict(results, orient='index').reset_index()  # convert to dataframe
    res_all = res_all.rename(columns={'index': 'Model'})  # rename index column to Model

    cols = list(res_all.columns)
    first_columns = ['Model', 'epsilon', 'alpha', 'alpha_unchosen', 'beta', 'V0', 
                     'forgetting_rate', 'forgetting_threshold', 
                     'stickyness_bias', 
                     'b1', 'b2', 'b3', 
                     'abandoned_bias', 'abandoned_decay']
    present = [c for c in first_columns if c in cols]
    rest = [c for c in cols if c not in present]
    new_columns = present + rest

    res_all = res_all[new_columns]

    return res_all

def behavs_to_dataframe(behaviors_simulated):
    # combine all behaviors into one dataframe
    behavs = []

    # then add the simulated behaviors
    for key, behav_temp in behaviors_simulated.items():
        behav_temp['session'] = 0
        behav_temp['model'] = key
        behavs.append(behav_temp)

    behaviors_simulated_all = pd.concat(behavs, axis=0)

    # reorder columns
    cols = ['model', 'session'] + [col for col in behav_temp.columns if col not in ['session', 'model']]
    behaviors_simulated_all = behaviors_simulated_all[cols]

    # reset index
    behaviors_simulated_all = behaviors_simulated_all.reset_index(drop=True)

    return behaviors_simulated_all

def save_res_and_behav(results, behaviors_simulated, floc):
    res_all = res_to_dataframe(results)
    floc_res_temp = os.path.join(floc, f'optimization_results.csv')
    res_all.to_csv(floc_res_temp, index=False)

    # save behaviors
    behaviors_simulated_all = behavs_to_dataframe(behaviors_simulated)
    floc_simulations_temp = os.path.join(floc, f'simulation_behaviors.pkl')
    behaviors_simulated_all.to_pickle(floc_simulations_temp)


n_initial_points = 1
n_calls = 2
n_jobs = -1
verbose = False
n_simulation_trials = 100_000
make_plots = False

epsilon_range = Real(.0, .5, name='epsilon')
alpha_range = Real(0.01, 1, name='alpha')
alpha_unchosen_range = Real(0, .8, name='alpha_unchosen')
beta_range = Real(10, 100, name='beta')
stickyness_range = Real(0.0, 50.0, name='stickyness_bias')
forgetting_rate_range = Real(0.0, 1.0, name='forgetting_rate')
forgetting_threshold_range = Real(0.0, 1.0, name='forgetting_threshold')
'''b1_range = Real(-50.0, 50.0, name='b1')
b2_range = Real(-5.0, 5.0, name='b2')
b3_range = Real(-50.0, 50.0, name='b3')'''
V0_range = Real(0.05, .4, name='V0')
abandoned_bias_range = Real(-50.0, 0.0, name='abandoned_bias')
abandoned_decay_range = Real(0.0, 1.0, name='abandoned_decay')

env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=n_simulation_trials)
floc = os.path.join(PROJECT_PATH_LOCAL, 'notebooks', 'behav_modeling', 'results', 'model_optimizing')
    

best_params_all, behavs = {}, {}  # Set container (to collect pandas series into a dataframe)


# ### Modified WSLS
model_name = 'WSLS agent'
agent_class = WSLSAgent_custom
fixed_params = {}
param_space = [epsilon_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

save_res_and_behav(best_params_all, behavs, floc)
print('Fitted WSLS agent')



# ### Simple RL agent
model_name = 'Standard RL'
agent_class = QLearner
fixed_params = {'structure_aware': False}
param_space = [alpha_range, beta_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

# ### Simple RL agent - stickyness
model_name = 'Standard RL - stickyness'
agent_class = QLearner
fixed_params = {'structure_aware': False}
param_space = [alpha_range, beta_range, stickyness_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

# ### Simple RL agent - forgetting
model_name = 'Standard RL - forgetting'
agent_class = QLearner
fixed_params = {'structure_aware': False}
param_space = [alpha_range, beta_range, forgetting_rate_range, forgetting_threshold_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

# ### Simple RL agent - forgetting + stickyness
model_name = 'Standard RL - forgetting + stickyness'
agent_class = QLearner
fixed_params = {'structure_aware': False}
param_space = [alpha_range, beta_range, forgetting_rate_range, forgetting_threshold_range, stickyness_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params


print('Fitted Standard RL agents')
save_res_and_behav(best_params_all, behavs, floc)



# ### Inferential RL
model_name = 'Inferential RL'
agent_class = QLearner
fixed_params = {'structure_aware': True}
param_space = [alpha_range, beta_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

# ### Inferential RL - stickyness
model_name = 'Inferential RL - stickyness'
agent_class = QLearner
fixed_params = {'structure_aware': True}
param_space = [alpha_range, beta_range, stickyness_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

# ### Inferential RL - multiple alphas + stickyness
model_name = 'Inferential RL - stickyness + multiple alphas'
agent_class = QLearner
fixed_params = {'structure_aware': True}
param_space = [alpha_range, beta_range, alpha_unchosen_range, stickyness_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params


print('Fitted Inferential RL agents')
save_res_and_behav(best_params_all, behavs, floc)



# ### Foraging - no reset
model_name = 'Foraging - no reset'
agent_class = ForagingAgent
fixed_params = {'reset_on_switch': False}
param_space = [alpha_range, beta_range, V0_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

# ### Foraging 
model_name = 'Foraging'
agent_class = ForagingAgent
fixed_params = {'reset_on_switch': True}
param_space = [alpha_range, beta_range, V0_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params

# ### Foraging - abandoned bias
model_name = 'Foraging - abandoned bias'
agent_class = ForagingAgent
fixed_params = {'reset_on_switch': True}
param_space = [alpha_range, beta_range, V0_range, abandoned_bias_range, abandoned_decay_range]

res_temp = fit_agent(agent_class, param_space, env, fixed_params=fixed_params, fit_on='rr', make_plots=False, verbose=False, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=n_jobs)
best_params = res_temp['best_params']

behavs[model_name] = simulate_agent(agent_class, best_params, env, fixed_params=fixed_params, behavioral_variables=[])
best_params['mean_reward_rate'] = res_temp['best_reward_rate']
best_params_all[model_name] = best_params


print('Fitted Foraging agents')
save_res_and_behav(best_params_all, behavs, floc)






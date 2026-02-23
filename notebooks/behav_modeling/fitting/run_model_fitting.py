# # Summary
#
# Lets fit the models to the actual data.

# ---
# # Setup

""" Run code in the background:
nohup /home/uzsombi/miniconda3/envs/popy_local/bin/python -u /workspace/uzsombi/inserm/PoPy/notebooks/behav_modeling/fitting/run_model_fitting.py > /workspace/uzsombi/inserm/PoPy/logs/run_model_fitting_$(date +%F_%H%M%S).log 2>&1 & echo $! > /workspace/uzsombi/inserm/PoPy/logs/run_model_fitting.pid

pkill -f "run_model_fitting.py"
"""

# @title Imports
import os

# Limit native-threaded math libs per process (prevents CPU oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
import pandas as pd
from skopt.space import Real
from multiprocessing import Pool
from tqdm import tqdm


from popy.simulation_tools import *
from popy.io_tools import load_behavior, load_behavior_yuri
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *
from popy.config import PROJECT_PATH_LOCAL

from popy.simulation_helpers import fit_simulate

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

def res_to_dataframe(results):
    res_all = pd.DataFrame.from_dict(results, orient='index').reset_index()  # convert to dataframe
    res_all = res_all.rename(columns={'index': 'Model'})  # rename index column to Model

    cols = list(res_all.columns)
    first_columns = ['Model', 'epsilon', 'alpha', 'alpha_unchosen', 'beta', 'V0', 
                     'forgetting_rate', 'forgetting_threshold', 
                     'stickiness_bias', 
                     'b1', 'b2', 'b3', 
                     'abandoned_bias', 'abandoned_decay']
    present = [c for c in first_columns if c in cols]
    rest = [c for c in cols if c not in present]
    new_columns = present + rest

    return res_all[new_columns]

def behavs_to_dataframe(behaviors_simulated, monkey):
    # combine all behaviors into one dataframe
    behavs = []

    # Process simulations: start with the real behavior
    behav_monkey = behaviors_simulated[f'MONKEY {monkey.upper()}']
    behav_monkey = behav_monkey.drop(columns=['switch'])
    behav_monkey['model'] = 'recording'
    behavs.append(behav_monkey)

    # then add the simulated behaviors
    for key, behav_temp in behaviors_simulated.items():
        if key != f'MONKEY {monkey.upper()}':
            behav_temp['monkey'] = monkey
            behav_temp['session'] = 0
            behav_temp['model'] = key
            behavs.append(behav_temp)

    behaviors_simulated_all = pd.concat(behavs, axis=0)

    # reorder columns
    cols = ['monkey', 'model', 'session'] + [col for col in behav_monkey.columns if col not in ['monkey', 'session', 'model']]
    behaviors_simulated_all = behaviors_simulated_all[cols]

    # reset index
    behaviors_simulated_all = behaviors_simulated_all.reset_index(drop=True)

    return behaviors_simulated_all

def save_res_and_behav(results, behaviors_simulated, monkey, floc):
    res_all = res_to_dataframe(results)
    floc_res_temp = os.path.join(floc, f'simulation_results_{monkey}.csv')
    res_all.to_csv(floc_res_temp, index=False)

    # save behaviors
    behaviors_simulated_all = behavs_to_dataframe(behaviors_simulated, monkey)
    floc_simulations_temp = os.path.join(floc, f'simulation_behaviors_{monkey}.pkl')
    behaviors_simulated_all.to_pickle(floc_simulations_temp)


# Init parameters

ANALYSIS_PARAMETERS = {
    "n_calls": 350,
    "n_initial_points": 100,
    "n_cpus": max(1, os.cpu_count() - 1),  
    "CV_splits": True,
    "n_jobs": 1,                                                   
    "verbose": False,
    "make_plots": False,
}

floc = os.path.join(PROJECT_PATH_LOCAL, 'notebooks', 'behav_modeling', 'results', 'fitting')

fit_params = {
    "epsilon": Real(0.01, 0.6, name="epsilon"),
    "alpha": Real(0.01, 1, name="alpha"),
    "alpha_unchosen": Real(0, 0.5, name="alpha_unchosen"),
    "alpha_threshold": Real(0.0, 0.3, name="alpha_threshold"),
    "beta": Real(0.5, 80.0, name="beta"),
    "stickiness_bias": Real(0.0, 50.0, name="stickiness_bias"),
    "forgetting_rate": Real(0.0, 1.0, name="forgetting_rate"),
    "forgetting_threshold": Real(0.0, 1.0, name="forgetting_threshold"),
    #'b1': Real(-50.0, 50.0, name='b1'),
    "b2_bias": Real(-5.0, 5.0, name="b2"),
    #'b3': Real(-50.0, 50.0, name='b3'),
    "V0": Real(0.05, 0.4, name="V0"),
    "abandoned_bias": Real(-50.0, 0.0, name="abandoned_bias"),
    "abandoned_decay": Real(0.0, 1.0, name="abandoned_decay"),
}

MODELS = {
    "Repeating agent": {
        "agent_class": RepeatingAgent,
        "fixed_params": {},
        "free_params": ["epsilon"],
    },

    "WSLS": {
        "agent_class": WSLSAgent,
        "fixed_params": {},
        "free_params": ["epsilon"],
    },

    "Standard RL": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": False},
        "free_params": ["alpha", "beta"],
    },
    "Standard RL - stickiness": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": False},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
    "Standard RL - forgetting": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": False},
        "free_params": ["alpha", "beta", "forgetting_rate", "forgetting_threshold"],
    },
    "Standard RL - stickiness + forgetting": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": False},
        "free_params": ["alpha", "beta", "forgetting_rate", "forgetting_threshold", "stickiness_bias"],
    },

    "Inferential RL": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": True},
        "free_params": ["alpha", "beta"],
    },
    "Inferential RL - stickiness": {    
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": True},
        "free_params": ["alpha", "beta", "stickiness_bias"],
    },
    "Inferential RL - stickiness + spatial bias": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": True},
        "free_params": ["alpha", "beta", "stickiness_bias", "b2_bias"],
    },
    "Inferential RL - stickiness + multiple alphas": {
        "agent_class": QLearner,
        "fixed_params": {"structure_aware": True},
        "free_params": ["alpha", "alpha_unchosen", "beta", "stickiness_bias"],
    },

    "Foraging - no reset": {
        "agent_class": ForagingAgent,
        "fixed_params": {"reset_on_switch": False},
        "free_params": ["alpha", "beta", "V0"],
    },
    "Foraging": {
        "agent_class": ForagingAgent,
        "fixed_params": {"reset_on_switch": True},
        "free_params": ["alpha", "beta", "V0"],
    },
    "Foraging - abandoned bias": {
        "agent_class": ForagingAgent,
        "fixed_params": {"reset_on_switch": True},
        "free_params": ["alpha", "beta",  "V0", "abandoned_bias", "abandoned_decay"],
    },
    "Foraging - abandoned bias + spatial bias": {
        "agent_class": ForagingAgent,
        "fixed_params": {"reset_on_switch": True},
        "free_params": ["alpha", "beta",  "V0", "abandoned_bias", "abandoned_decay", "b2_bias"],
    },
    "Foraging - adaptive threshold": {
        "agent_class": ForagingAgentAdaptive,
        "fixed_params": {},
        "free_params": ["alpha", "beta", "V0", "alpha_threshold"],
    },

    "HSMM": {
        "agent_class": HSMMAgent,
        "fixed_params": {},
        "free_params": ["beta"],
    },
}


# Run parameter fitting per monkey

def _run_model_fitting(model_name, behav_monkey, env):
    model_params = MODELS[model_name]

    agent_class = model_params["agent_class"]
    fixed_params = model_params["fixed_params"]
    free_params = model_params["free_params"]
    param_space = [fit_params[p] for p in free_params]

    results, behaviors_simulated = fit_simulate(
        agent_class,
        param_space,
        env,
        behav_monkey,
        fixed_params,
        fit_on_first_10=False,
        CV_splits=ANALYSIS_PARAMETERS["CV_splits"],
        make_plots=ANALYSIS_PARAMETERS["make_plots"],
        n_calls=ANALYSIS_PARAMETERS["n_calls"],
        n_initial_points=ANALYSIS_PARAMETERS["n_initial_points"],
        n_jobs=ANALYSIS_PARAMETERS["n_jobs"],
    )
     
    # print(f'Fitted {model_name}')

    return {model_name: results}, {model_name: behaviors_simulated}


def _run_model_fitting_star(args):
    """Unpack args for Pool.imap/imap_unordered."""
    return _run_model_fitting(*args)

if __name__ == '__main__':
    for monkey in ['ka', 'po']:  # ['ka', 'po', 'yu_sham', 'yu_DCZ']
        print(f'--- {monkey} ---')

        ### Get data
        env = gym.make("zsombi/monkey-bandit-task-v0", n_arms=3, max_episode_steps=100_000)
        behav_monkey = get_data_custom(monkey)
        results = {}
        behaviors_simulated = {f'MONKEY {monkey.upper()}': behav_monkey}

        ### Fit models in parallel
        tasks = [
            (model_name, behav_monkey, env)
            for model_name in MODELS.keys()
        ]

        with Pool(ANALYSIS_PARAMETERS['n_cpus']) as pool:
            recovery_results = list(
                tqdm(
                    pool.imap_unordered(_run_model_fitting_star, tasks),
                    total=len(tasks),
                    desc=f"Fitting models ({monkey})",
                    unit="model",
                )
            )

        # Collect results
        for res_dict, behav_dict in recovery_results:
            results.update(res_dict)
            behaviors_simulated.update(behav_dict)

        # sort results[Model] MODELS.keys() (same order as in MODELS, but start with "MONKEY {upper(monkey)}")
        save_res_and_behav(results, behaviors_simulated, monkey, floc)


    print('saved all')


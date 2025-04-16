"""
Functions for optimizing parameters (e.g. fitting them for best performance or to fit data). Also includes functions for simulating agents and estimating log likelihoods.
"""

# @title imports
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import pandas as pd

from skopt import gp_minimize, dummy_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

from popy.simulation_tools import *


class MakeRecording():
    """
    Records the environment and agent states during simulation.
    """

    def __init__(self):
        self.recording = []

    def record(self, action, reward, info, agent, agent_vars=[]):
        """
        Record the environment and agent states.

        Parameters
        ----------
        action : int
            The action taken by the agent.
        reward : float
            The reward received by the agent.
        info : dict
            The information returned by the environment.
        agent : Agent
            The agent.
        agent_vars : list
            The agent's "internal" variables to record.
        """
        # store the behavior
        trial_behav = {
            "action": action,
            "reward": reward,
        }

        # save what happens in the environment
        for key, value in info.items():
            trial_behav[key] = value.copy() if isinstance(value, np.ndarray) else value

        # save the agent internal variables
        for var in agent_vars:
            agent_var = getattr(agent, var)
            trial_behav[var] = agent_var.copy() if isinstance(agent_var, np.ndarray) else agent_var

        self.recording.append(trial_behav)

    def get_recording(self):
        """
        Get the recording.
        """ 

        # sort keys in this order: monkey, session, trial_id, block_id, best_action, target_feedback, and followed by the rest
        recordings_df = pd.DataFrame(self.recording)
        first_cols = ["trial_id", "block_id", "best_arm", "action", "reward"]
        other_cols = [col for col in recordings_df.columns if col not in first_cols]
        recordings_df = recordings_df[first_cols + other_cols]
        return recordings_df

### Functions for estimating log likelihood and reward rate ###
def estimate_rr(
        agent_class, 
        params,
        env,
        fixed_params=None):
    """
    Same as simulate_agent but only returns the mean reward rate.
    """
    if fixed_params is None:
        fixed_params = {}

    # create the agent
    agent_params = {**params, **fixed_params}
    agent = agent_class(**agent_params)

    # record the behavior
    rewards = []

    # reset the environment
    _, _ = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.act()
        _, reward, _, done, info = env.step(action)

        # update the agent
        agent.update_values(action, reward)

        rewards.append(reward)

    # convert the behavior to a pandas DataFrame
    return np.mean(rewards)


def estimate_ll(
        agent_class, 
        params,
        behavior_original, 
        fixed_params=None):
    '''
    Estimate the log likelihood of the behavior given an agent.
    '''
    # create the agent
    if fixed_params is None:
        fixed_params = {}

    # create the agent
    agent_params = {**params, **fixed_params}
    agent = agent_class(**agent_params)

    # process the behavior
    behavior = behavior_original.copy()
    behavior.reset_index(drop=True, inplace=True)

    # get likelihood of the behavior
    logp_actions = np.zeros(len(behavior))
    session_previous = 'none'
    for i, row in behavior.iterrows():
        action, reward = row["action"], row["reward"]
        session_curr = row["session"]

        # Skip NaN actions (interrupted trials)
        if np.isnan(action):
            logp_actions[i] = np.log(1/agent.n_arms)
            continue

        # If we are in a new session, reset the agent
        if session_curr != session_previous:
            agent.reset()
            session_previous = session_curr

        # Get the action LL
        p_actions = agent.get_action_probas().copy()  # Get the action probabilities
        p_action = p_actions[int(action)]
        logp_action = np.log(p_action)
        logp_actions[i] = logp_action

        # update the agent
        agent.update_values(int(action), int(reward))

    return np.sum(logp_actions)


### Functions for simulating agents ###


def simulate_agent(
        agent_class, 
        params,
        env,
        fixed_params=None,
        behavioral_variables=[],
        n_trials=None):
    """
    Simulate an agent in an environment for a number of episodes and return the behavior.
    """
    if fixed_params is None:
        fixed_params = {}
    if n_trials is not None:
        env = TimeLimit(env, max_episode_steps=n_trials)

    # create the agent
    agent_params = {**params, **fixed_params}
    agent = agent_class(**agent_params)

    # record the behavior
    recording = MakeRecording()

    # reset the environment
    _, _ = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.act()
        _, reward, _, done, info = env.step(action)

        recording.record(action, reward, info, agent, behavioral_variables)

        # update the agent
        agent.update_values(action, reward)

    # convert the behavior to a pandas DataFrame
    return recording.get_recording()


### Functions for fitting agents and cross-validation ###


def fit_agent(agent_class, param_space, env, behav_data=None, fixed_params=None, fit_on='ll', n_calls=50, n_initial_points=10, n_jobs=1, verbose=False, make_plots=True):
    """
    Fit an agent model to behavioral data using Bayesian optimization.
    """
    if fixed_params is None:
        fixed_params = {}
        
    # Define the objective function with dynamic parameter handling
    @use_named_args(param_space)
    def objective(**params):
        if fit_on == 'll':
            # Run simulation and get log likelihood
            ll = estimate_ll(agent_class, params, behav_data, fixed_params)
            return -ll
        elif fit_on == 'rr':
            # Run simulation and get reward rate
            rr = estimate_rr(agent_class, params, env, fixed_params)
            return -rr
        else:
            raise ValueError(f"Invalid value for 'fit_on': {fit_on}")

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        param_space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        verbose=verbose,
        acq_optimizer="lbfgs",
        n_jobs=n_jobs,
    )

    # Extract best parameters
    param_names = [p.name for p in param_space]
    best_params = dict(zip(param_names, result.x))
        
    # Create plots
    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot convergence
        plot_convergence(result, ax=axes[0])
        axes[0].set_title("Convergence Plot")
        
        # Plot parameter importance if there's more than one parameter
        if len(param_space) > 1:
            plot_objective(result, dimensions=param_names, ax=axes[1])
        else:
            plot_objective(result, ax=axes[1])
        axes[1].set_title("Parameter Importance")
        
        plt.tight_layout()
    
    if fit_on == 'll':
        # Calculate BIC and log likelihood per trial
        best_ll = -result.fun  # Convert back to positive log likelihood
        n_params = len(param_space)
        bic = -2 * best_ll + n_params * np.log(len(behav_data))
        lpt = np.exp(best_ll / len(behav_data))

        return {
            #'result': result,
            'best_params': best_params,
            'best_ll': best_ll,
            'bic': bic,
            'lpt': lpt
        }
    elif fit_on == 'rr':
        return {
            #'result': result,
            'best_params': best_params,
            'best_reward_rate': -result.fun
        }
    else:
        raise ValueError(f"Invalid value for 'fit_on': {fit_on}")


def cross_val_fit(agent_class, param_space, env, behav_data, fixed_params=None, CV_splits=5, n_calls=50, n_initial_points=10):
    """
    Fit an agent model to behavioral data using Bayesian optimization with cross-validation.
    """
    # Split data
    n_trials = len(behav_data)
    n_trials_per_split = n_trials // CV_splits
    split_indices = np.array_split(np.arange(n_trials), CV_splits)

    # Fit model on each split
    results = []
    for i, split_idx in enumerate(split_indices):
        # Get training and test data
        train_idx = np.concatenate([split_indices[j] for j in range(CV_splits) if j != i])
        train_data = behav_data.iloc[train_idx]
        test_data = behav_data.iloc[split_idx]

        # Fit model
        result_temp = fit_agent(agent_class, param_space, env, train_data, fixed_params, n_calls=n_calls, n_initial_points=n_initial_points, n_jobs=1, verbose=False, make_plots=False)

        # Evaluate model on test data
        ll = estimate_ll(agent_class, result_temp['best_params'], test_data, fixed_params=fixed_params)

        #### Save results
        res = {
            #'Model': model_name,
            #'Model family': 'Baseline',
            'CV_fold': i,
            'LL': ll,
            'BIC': -2 * ll + len(param_space) * np.log(len(test_data)),
            'LPT': np.exp(ll / len(test_data)),
        }
        res = {**res, **result_temp['best_params']}  # Add best parameters to results
        results.append(res)

    return pd.DataFrame(results)


### Main function for all analysis of model fitting in once ###


def fit_simulate(agent_class, param_space, env, behav, fixed_params=None,
                CV_splits=None,
                n_calls=50, n_initial_points=10, n_jobs=1, verbose=False, make_plots=False):
    """
    Fit an agent model to behavioral data and simulate the agent to get reward rate and probability of choosing best arm.
    """

    # collect results
    res = {}

    #### Fit the model ####

    results = fit_agent(
        agent_class=agent_class,
        param_space=param_space,
        fixed_params=fixed_params,
        env=env,
        behav_data=behav,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        n_jobs=n_jobs,
        verbose=verbose,
        make_plots=make_plots
    )

    res = {**res, **results['best_params']}  # Add best parameters to results

    res['LL_best'] = results['best_ll']
    res['BIC_best'] = results['bic']
    res['LPT_best'] = results['lpt']

    #### Cross-validate the model ####

    # run cross validation
    if CV_splits is not None:
        CV_scores = cross_val_fit(agent_class, param_space,env, behav, fixed_params, CV_splits=CV_splits, n_calls=60, n_initial_points=10)

        res['LL_CV'] = CV_scores['LL'].mean()
        res['LL_std'] = CV_scores['LL'].std()
        res['BIC_CV'] = CV_scores['BIC'].mean()
        res['BIC_std'] = CV_scores['BIC'].std()
        res['LPT_CV'] = CV_scores['LPT'].mean()
        res['LPT_std'] = CV_scores['LPT'].std()

    #### Simulate agent ####

    # Simulate agent for comparison of reward rate and probability of choosing best arm
    behavior_simulated = simulate_agent(
        agent_class=agent_class, 
        params=results['best_params'],
        env=env,
        fixed_params=fixed_params
    )
    reward_rate = behavior_simulated['reward'].mean()
    proba_best = (behavior_simulated['action'] == behavior_simulated['best_arm']).mean()

    #### Save results
    res['Reward rate'] = reward_rate
    res['Proba best'] = proba_best

    return pd.Series(res), behavior_simulated

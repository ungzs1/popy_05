"""
Functions to transform the session data dataframe. The dataframe is the already processed pandas df, not the raw data!
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit

import popy.config as cfg
from popy.simulation_tools import *



### General tools

def add_trial_in_block(behav):
    """
    Adds a 'trial_id_in_block' column to the behav DataFrame, which represents the trial index within each block.

    Parameters:
    behav (pandas.DataFrame): 
        The input DataFrame containing the session data.

    Returns:
    pandas.DataFrame: 
        The modified DataFrame with the 'trial_id_in_block' column added.
    """
    
    # if already has the property, just return as it is
    if 'trial_id_in_block' in behav.columns:
        return behav
    # group by monkey, session, block
    for info, subdf in behav.groupby(['monkey', 'session', 'block_id']):
        # write back to the original dataframe
        behav.loc[subdf.index, 'trial_id_in_block'] = np.arange(len(subdf))

    # type is int
    behav['trial_id_in_block'] = behav['trial_id_in_block'].astype(int)

    return behav

def drop_time_fields(behav):
    """
    Drop the time fields from the session data.
    """

    # drop columns ending with '_time'
    behav = behav.drop([col for col in behav.columns if col.endswith('_time')], axis=1)

    return behav

def add_date_of_recordig(behav):
    """
    Reads the 'session' column and converts it to a datetime object.

    Parameters:
    -----------
    behav: pd.DataFrame
        Dataframe with the session column.

    Returns:
    --------
    behav: pd.DataFrame
        Dataframe with the session column converted to datetime object.
    """
    def conv_time(t):
        dd, mm, yy = t[:2], t[2:4], t[4:6]
        date_time = np.datetime64(f'20{yy}-{mm}-{dd}')
        return date_time

    behav['date_of_recording'] = behav['session'].apply(conv_time)
    # reorder columns so that date in the first one
    cols = behav.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    behav = behav[cols]
    
    return behav

def convert_column_format(behav, original='simulation'):
    """
    Converts column names and values from behavioral to modeling format (because unfortunately different naming and coding were used).

    Parameters:
    ------------
    behav: pd.DataFrame
        The session data dataframe.
    original: str
        The original format of the data. Can be 'behavior' or 'simulation'.

    Returns:
    ------------
    behav: pd.DataFrame
        The session data dataframe with the column names and values converted to the modeling format
    """
    if original == 'behavior':
        # rename target to action, also convert values from [1,2,3] to [0,1,2]
        target = behav['target'].values
        target[target==1] = 0
        target[target==2] = 1
        target[target==3] = 2
        behav['target'] = target
        # rename to action
        behav = behav.rename(columns={'target': 'action'})

        # rename feedback to reward
        behav = behav.rename(columns={'feedback': 'reward'})

        # change best target to best arm
        best_target = behav['best_target'].values
        best_target[best_target==1] = 0
        best_target[best_target==2] = 1
        best_target[best_target==3] = 2
        behav['best_target'] = best_target
        behav = behav.rename(columns={'best_target': 'best_arm'})
    
    elif original == 'simulation':
        # rename action to target, also convert values from [0,1,2] to [1,2,3]
        actions = behav['action'].values
        actions[actions==2] = 3
        actions[actions==1] = 2
        actions[actions==0] = 1
        behav['action'] = actions
        behav = behav.rename(columns={'action': 'target'})

        # rename reward to feedback
        behav = behav.rename(columns={'reward': 'feedback'})

        # change best arm to best target
        best_arm = behav['best_arm'].values
        best_arm[best_arm==2] = 3
        best_arm[best_arm==1] = 2
        best_arm[best_arm==0] = 1
        behav['best_arm'] = best_arm
        behav = behav.rename(columns={'best_arm': 'best_target'})

        behav['monkey'] = 'simulation'
        behav['session'] = -1

        # reorder so that monkey is first and session is second
        cols = behav.columns.tolist()
        cols = cols[-2:] + cols[:-2]

        behav = behav[cols]

    else:
        raise ValueError("Invalid original format.")
        
    return behav

### Feedback functions

def add_double_feedback(behav):
    new_col = []
    for i, fb in enumerate(behav.feedback):
        if i == 0:
            new_col.append(np.nan)
            continue
        fb_prev = behav.feedback[i - 1]
        if fb == fb_prev:
            new_col.append(fb)
        else:
            new_col.append(np.nan)
    behav['double_feedback'] = new_col
    return behav

def add_history_of_feedback(behav_original, num_trials=8, one_column=True, coding=(0, 1), binary=False):
    """
    Add history of feedback to behav.

    Parameters:
    behav (pandas.DataFrame): 
        The input DataFrame containing the session data.

    num_trials (int):
        The number of trials to consider in the history of feedback.

    one_column (bool):
        If True, the history of feedback will be added as a single column to the DataFrame, named 'history_of_feedback'. Each entry will be a list of the last N feedbacks, as [Rt-1, Rt-2, ..., Rt-N].
        If False, it will be added as multiple columns.

    coding (tuple):
        The coding of the feedback. The first element is the code for non-rewarded trials, and the second element is the code for rewarded trials.

    Returns:
    pandas.DataFrame:
        The input DataFrame with the history of feedback added.

    """
    if binary:
        one_column = True

    # remove existing history of feedback columns
    if one_column:
        behav_original = behav_original.drop(columns=['history_of_feedback'], errors='ignore')
    else:
        for j in range(num_trials):
            behav_original = behav_original.drop(columns=[f'R_{j+1}'], errors='ignore')

    # copy the original dataframe
    behav = behav_original.copy()

    # recode feedback
    behav['feedback'] = behav['feedback'].replace(0, coding[0])
    behav['feedback'] = behav['feedback'].replace(1, coding[1])

    # iterate over subdf's of (monkey, session)
    hist_fb = {f'R_{j+1}': [] for j in range(num_trials)}  # {R-1: [], R-2: [], ...}
    hist_target = {f'T_{j+1}': [] for j in range(num_trials)}  # {T-1: [], T-2: [], ...}
    for _, subdf in behav.groupby(['monkey', 'session']):
        # iterate row by row in behav
        for i, _ in subdf.iterrows():
            for j in range(num_trials):  # iterate over the last N trials
                index = i - j - 1  # get the index of the trial in the past
                if index < subdf.index[0]:  # if the index is out of range, set it to nan
                    hist_fb[f'R_{j+1}'].append(np.nan)
                    hist_target[f'T_{j+1}'].append(np.nan)
                else:
                    fb_historic = subdf.loc[index, 'feedback']  # get the feedback of the trial in the past
                    target_historic = subdf.loc[index, 'target']  # get the target of the trial in the past

                    if np.isnan(fb_historic):  # if the feedback is nan (interrupted trial), set it to nan
                        hist_fb[f'R_{j+1}'].append(np.nan)
                        hist_target[f'T_{j+1}'].append(np.nan)
                    else:  # if everything is fine, append the feedback (or target_weighted feedback if cmw is True)
                        hist_fb[f'R_{j+1}'].append(int(fb_historic))
                        hist_target[f'T_{j+1}'].append(int(target_historic))

                    
    # add to dataframe
    if one_column:
        column_data = []
        for i in range(len(behav)):
            trial_data = []
            add_nan = False
            for j in range(num_trials):
                curr_trial_outcome = hist_fb[f'R_{j+1}'][i]
                if curr_trial_outcome == curr_trial_outcome:
                    trial_data.append(curr_trial_outcome)
                else:
                    add_nan = True
                    trial_data = np.nan
                    break

            if binary and not add_nan:
                # convert the list of 0s and 1s to a single number
                trial_data = np.array(trial_data)
                trial_data = np.sum(trial_data * (2 ** np.arange(num_trials)[::-1]))  # convert to binary number
                trial_data = trial_data.astype(float)  # convert to int

            column_data.append(trial_data)

        behav_original.loc[:, 'history_of_feedback'] = column_data
    else:
        for key in hist_fb.keys():
            behav_original[key] = hist_fb[key]
        for key in hist_target.keys():
            behav_original[key] = hist_target[key]

    return behav_original

### Performance monitoring functions

def add_q_values(behav_original, monkey=None, coding=(0, 1), alpha=None):  # todo use the models with the fit parameters to calculate the q-values
    """
    Calculate the value function (Q-values) for each target based on the history of feedback.

    Also adds a column 'Q_selected' to the dataframe, which contains the value function corresponding to the target that was selected (i.e. the expectation).
    """
    behav = behav_original.copy()

    assert monkey is not None, "Please provide the monkey name to calculate the value function!"

    if alpha is None:
        if monkey == 'ka':
            decay =  0.709  # fit by anna on the exponential model
            alpha = 1 - np.exp(-decay)  # converted to learning rate
        elif monkey == 'po':
            #decay = 0.42924468 # fit by anna
            #alpha = 1 - np.exp(-decay)  # converted to learning rate
            alpha = 0.42924468  # fitted on Popeye's behav data with regression

    # check if history of feedback column exists
    behav = add_history_of_feedback(behav, num_trials=7, one_column=False, coding=coding)

    # drop rows with nan values in the history of feedback
    behav = behav.dropna(subset=behav.columns[behav.columns.str.contains('R_')])
    behav = behav.dropna(subset=behav.columns[behav.columns.str.contains('T_')])
    behav = behav.dropna(subset=['target'])  # drop rows with nan target, i.e. interrupted trials

    len_history = behav.columns.str.contains('R_').sum()  # number of columns with reward history

    # contruct weight vector
    #weight_vector = np.exp(-decay * np.arange(1, len_history+1))
    weight_vector = np.array([alpha * (1 - alpha) ** (len_history-i) for i in np.arange(1, len_history+1)[::-1]])
    weight_vector = weight_vector / np.sum(weight_vector)  # normalize

    # Compute q-values
    qvalues = np.zeros((len(behav), 3))
    for i, (_, row) in enumerate(behav.iterrows()):
        reward_vector = np.array([row[f'R_{i}'] for i in range(1, len_history+1)])  # a vector of 1 if reward, 0 if no reward
        for j in range(3):        
            curr_target = j + 1
            cmw_vector = np.array([1 if row[f'T_{i}'] == curr_target else -1 for i in range(1, len_history+1)])  # a vector of 1 if choice was j, -1 if not
        
            qvalues[i, j] = np.dot(weight_vector, cmw_vector * reward_vector)  # compute q-value

    # add q-values to the dataframe
    behav['Q_1'] = qvalues[:, 0]  # value function corresponding to target 1
    behav['Q_2'] = qvalues[:, 1]  # value function corresponding to target 2
    behav['Q_3'] = qvalues[:, 2]  # value function corresponding to target 3
    behav['Q_selected'] = [row[f'Q_{int(row["target"])}'] for _, row in behav.iterrows()]
    #behav['Q_upcoming'] = [row[f'Q_{int(behav.iloc[i, "target"])}'] for i, row in behav[:-1].iterrows()]

    # write back the q_values to the original dataframe, matching the indices
    behav_original['Q_1'] = behav_original.index.map(behav['Q_1'])
    behav_original['Q_2'] = behav_original.index.map(behav['Q_2'])
    behav_original['Q_3'] = behav_original.index.map(behav['Q_3'])
    behav_original['Q_selected'] = behav_original.index.map(behav['Q_selected'])
    #behav_original['Q_upcoming'] = behav_original.index.map(behav['Q_upcoming'])
    
    return behav_original

def OLD_add_value_function(behav_original, decay=None, monkey=None, digitize=False, n_classes=4, prins_user_msg=False):
    raise ValueError("This function is deprecated, use add_value_function_2 instead.")
    """
    VALUE CORRESPONDING TO TRIAL t IS THE ONE KNOWN BEFORE THE FEEDBACK OF TRIAL t IS RECEIVED!

    This function calculates the weighted average of the history of feedback. The history of feedback attribute should be added prior 
    to running this function (otherwise its added here - but be careful, don't add the history of feedback column to the pooled dataframe (pooled across sessions),
    because it might not be ordered in time properly!).

    The weights are sampled from an exponential decay finction w = a * np.exp(-b * x), which's parameters should be provided in the form of [a, b].
    
    Decay parameter is set to .45 by default, which is the value we derived from the behavioral data of Kate.
    """
    
    if monkey is not None:
        raise ValueError("Please DONT provide the monkey name anymore!")
    
    behav_copy = behav_original.copy()

    for monkey, behav in behav_copy.groupby('monkey'):
        if monkey == 'ka':
            decay =  0.44372805  # fitted on Kate's behav data
            #params = -1 * np.array([-3.25506236,  4.32471672,  2.48795523,  1.34433653,  0.84478453, 0.60457409,  0.15984625,  0.0579089 , -0.00542465])
        elif monkey == 'po':
            decay = 0.42924468 # fitted on Popy's behav data
            #params = -1 * np.array([-2.20732448,  2.35881073,  1.28012092,  0.65484842,  0.4044955 ,0.33183202,  0.22720791,  0.12090216,  0.11315643])
        else:
            continue

        # normalize params
        #params = params / np.sum(params)

        # check if history of feedback column exists
        if 'history_of_feedback' not in behav.columns:
            if prins_user_msg:
                print("adding history of feedback column, but if the data is pooled across sessions, make sure to add the history before pooling data",
                    "to keep the order of time!")
            behav = add_history_of_feedback(behav, num_trials=5)

        # drop rows with nan values in the history of feedback
        behav = behav.dropna(subset=['history_of_feedback'])

        X = np.vstack(behav.history_of_feedback.values)    
        X = X.astype(int)
        num_trials = X.shape[1]    # number of trials to consider
        
        weights = decay * (1-decay)**np.arange(num_trials)   # calculate weights
        weights = weights / np.sum(weights)    # normalize weights

        # compute Q-values by weighted average of the history of feedback
        q_values = X @ weights

        if digitize:
            values = q_values[~np.isnan(q_values)]
            values = np.sort(values)
            bins = [values[int(len(values) * i / n_classes)] for i in range(n_classes)] 
            q_values = np.digitize(q_values, bins) - 1

        # add q_values to the behav, by index
        #idxs = behav.index.values  # indices in df where nan values are dropped
        #behav['value_function'] = np.nan  # add value_function column to the original df, with nan values
        
        behav.loc[:, 'value_function'] = q_values

        # write back the q_values to the original dataframe, matching the indices
        #behav_copy['value_function'] = behav_copy.index.map(behav['value_function'])
        behav_copy.loc[behav.index, 'value_function'] = behav.loc[behav.index, 'value_function'].values

    return behav_copy

def add_value_function(behav_original, digitize=False, n_classes=4):
    """
    VALUE CORRESPONDING TO TRIAL t IS THE ONE KNOWN BEFORE THE FEEDBACK OF TRIAL t IS RECEIVED!

    This function calculates the weighted average of the history of feedback using the shift-value model, with no reset.
    """

    behavior = behav_original.copy()
    behavior.reset_index(drop=True, inplace=True)  # reset index 

    for monkey, behav in behavior.groupby('monkey'):
        # init an agent
        agent_class = ShiftValueAgent
        fixed_params = {'reset_on_switch': False}
        if monkey == 'ka':
            params = {
                'alpha': .44372805}
        elif monkey == 'po':
            params = {
                'alpha': 0.42924468}
        else:
            # for the simulations there is already a value function, but for sanity, we can add it instead of using the simulated one
            continue

        agent = agent_class(**params, **fixed_params)

        shift_values = np.zeros(len(behav))
        session_previous = 'none'
        for i, (_, row) in enumerate(behav.iterrows()):
            # Get latent value function
            shift_values[i] = agent.V

            # Get the action and reward taken by the agent
            action, reward = row["target"], row["feedback"]
            session_curr = row["session"]

            # Skip NaN actions (interrupted trials)
            if np.isnan(action):
                shift_values[i] = np.nan
                continue

            # If we are in a new session, reset the agent
            if session_curr != session_previous:
                agent.reset()
                session_previous = session_curr

            # update the agent
            agent.update_values(int(action), int(reward))

        behav.loc[:, 'value_function'] = shift_values

        # write back the q_values to the original dataframe, matching the indices
        behavior.loc[behav.index, 'value_function'] = behav.loc[behav.index, 'value_function'].values

    if digitize:
        # Create bin edges (from min to max value, or 0-1 if specified)
        bin_edges = np.linspace(0, 1, n_classes+1)
                
        # Perform the binning
        behavior['value_function'] = np.digitize(
            behavior['value_function'], 
            bins=bin_edges, 
            right=False) - 1

    return behavior

def add_shift_value(behav_original, digitize=False, n_classes=4, alpha_ka=None, alpha_po=None):
    """
    VALUE CORRESPONDING TO TRIAL t IS THE ONE KNOWN BEFORE THE FEEDBACK OF TRIAL t IS RECEIVED!

    Model-based value function calculation. 
    
    The value function is calculated based on the Shift-value model with reset.
    """

    behavior = behav_original.copy()
    behavior.reset_index(drop=True, inplace=True)  # reset index 

    for monkey, behav in behavior.groupby('monkey'):
        # init an agent
        agent_class = ShiftValueAgent
        fixed_params = {'reset_on_switch': True}
        if monkey == 'ka':
            params = {
                'alpha': 0.41 if alpha_ka is None else alpha_ka,
                #'beta': None,  # beta is not used unless we make decisions 
                'V0': 0.1}
        elif monkey == 'po':
            params = {
                'alpha': 0.32 if alpha_po is None else alpha_po,
                #'beta': None,  # beta is not used unless we make decisions
                'V0': 0.16}
        else:
            # for the simulations there is already a value function, but for sanity, we can add it instead of using the simulated one
            continue

        agent = agent_class(**params, **fixed_params)

        shift_values = np.zeros(len(behav))
        session_previous = 'none'
        for i, (_, row) in enumerate(behav.iterrows()):
            # Get latent value function
            shift_values[i] = agent.V

            # Get the action and reward taken by the agent
            action, reward = row["target"], row["feedback"]
            session_curr = row["session"]

            # Skip NaN actions (interrupted trials)
            if np.isnan(action):
                shift_values[i] = np.nan
                continue

            # If we are in a new session, reset the agent
            if session_curr != session_previous:
                agent.reset()
                session_previous = session_curr

            # update the agent
            agent.update_values(int(action), int(reward))
        
        behav.loc[:, 'shift_value'] = shift_values

        # write back the q_values to the original dataframe, matching the indices
        behavior.loc[behav.index, 'shift_value'] = behav.loc[behav.index, 'shift_value'].values

    if digitize:
        # Create bin edges (from min to max value, or 0-1 if specified)
        bin_edges = np.linspace(0, 1, n_classes+1)
                
        # Perform the binning
        behavior['shift_value'] = pd.cut(
            behavior['shift_value'], 
            bins=bin_edges, 
            include_lowest=True
        )

    return behavior

def add_RPE(behav, scale_RPE=False):
    if 'value_function' not in behav.columns:
        behav = add_value_function(behav)

    # add reward prediction error
    
    if not scale_RPE:
        rpe = behav.feedback - behav.value_function
        behav['RPE'] = rpe
    else:
        fb = behav.feedback.values
        # y score values
        vf = behav.value_function.values
        vf = vf / np.std(vf)
        rpe = fb - vf
        behav['RPE'] = rpe
        
    
    return behav

### Decision making functions

def add_phase_info(behav_original, exploration_limit=5, transition_limit=5, numeric=False):
    """
    Add phase (i.e. 'search', 'transition' or 'repeat') info to behav. We define the phases based on the number of consecutive selections of the same action.
    
    How many consecutive selection of the same action is considered as exploration and transition is defined by the exploration_limit and transition_limit parameters.

    Note that in the beginning of a session we do not know the phase, the first N trials are considered as np.nan phase (where N is exploration_limit + transition_limit), unless there is a switch in the first N trials which certainly means a serach pahse.

    Parameters:
    behav (pandas.DataFrame): 
        The input DataFrame containing the session data.

    exploration_limit (int):
        The number of consecutive selections of the same action that is considered as exploration (or 'search'). I.e. if exploration_limit=5, then the first 5 consecutive selections of the same action is considered as exploration,
        and the 6th selection is considered as 'transition' or 'repeat' (depending on the transition_limit parameter).

    transition_limit (int):
        The number of consecutive selections of the same action that is considered as transition. I.e. if transition_limit=5, then the 6th to 10th consecutive selections of the same action is considered as transition,
        and the 11th selection is considered as 'repeat'.

    numeric (bool):
        If True, the phase info will be coded numerically (0 for 'search', 1 for 'transition', 2 for 'repeat', and np.nan for nan).

    """
    behav = behav_original.copy()
    phases_all = []

    for (monkey, session), sub_behav in behav.groupby(['monkey', 'session']):
        ## handling the beginning of the trial
        # is there a switch in the first N trials that cover the exploration and transition??
        if sub_behav.target[:exploration_limit+transition_limit].nunique() > 1:
            targets = sub_behav.target[:exploration_limit+transition_limit].values
            switches = [False] + [targets[i] != targets[i-1] for i in range(1, len(targets))]
            first_switch_id = np.argwhere(switches)[0][0]

            phases = [np.nan for x in range(first_switch_id)] + ['search']
            num_cont_selection = 0
            init_analisis_id = first_switch_id+1
        else:
            phases = [np.nan for x in range(exploration_limit+transition_limit)]
            num_cont_selection = exploration_limit+transition_limit
            init_analisis_id = exploration_limit+transition_limit

        last_target = sub_behav.target.values[init_analisis_id-1]
        for _, row in sub_behav[init_analisis_id:].iterrows():  # skip the first N+threshold number of trials, as we dont know what happened before...
            if row.target == last_target:
                num_cont_selection += 1
            else:
                num_cont_selection = 0
                last_target = row.target

            if num_cont_selection < exploration_limit:
                phases.append('search')
            elif num_cont_selection < exploration_limit+transition_limit:
                phases.append('transition')
            else:
                phases.append('repeat')

        if numeric:
            phase_coding = {'search': 0, 'transition': 1, 'repeat': 2, np.nan: np.nan}
            phases = [phase_coding[x] for x in phases]

        phases_all += phases

    behav['phase'] = phases_all

    return behav

def add_switch_info(behav_full, add_trials_since_switch=False, flip_coding=False):
    """
    Is the decision during this trial means a Switch (from the previous trial)?

    Add switch info to behav. A trial is assigned with a True value if the current target is the same as the previous,
    while a False value is assigned if the current target is different from the previous.
    """
    # add switch info
    switch_info = []
    trials_since_switch_info = []
    for session_id, behav in behav_full.groupby(['monkey', 'session']):
        last_switch_index = 0
        for i, index in enumerate(behav.index):
            if i == 0:
                switch = np.nan
            else:
                switch = behav.target[index] != behav.target[index-1]

            switch_info.append(switch)
            trials_since_switch_info.append(index - last_switch_index)

            if switch:
                last_switch_index = index
    behav_full['switch'] = switch_info
    behav_full['switch'] = behav_full['switch'].astype(float)

    if flip_coding:
        behav_full['switch'] = behav_full['switch'].apply(lambda x: not x)

    if add_trials_since_switch:
        behav_full['trials_since_switch'] = trials_since_switch_info

    return behav_full

def add_reaction_time(behav):
    """
    Add reaction time to the session data.

    Reaction time is calculated as the time between the lever release and the target touch.
    """
    # add reaction time
    reaction_time = []
    for i, index in enumerate(behav.index):
        reaction_time.append(behav.target_touch_time[index] - behav.lever_release_time[index])

    behav['RT'] = reaction_time

    return behav

def add_trials_since_switch(behav):
    import warnings
    warnings.warn("This function is deprecated, use add_switch_info with add_trials_since_switch=True instead.")

    behav = add_switch_info(behav, add_trials_since_switch=True)
    return behav

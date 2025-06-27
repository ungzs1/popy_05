"""
Ploting methods generally used ofer multiple different analysis.
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from popy.neural_data_tools import *
from popy.behavior_data_tools import *

import os
import warnings


import numpy as np
import matplotlib.pyplot as plt

from popy.config import COLORS, MODEL_PARAMS
from matplotlib.colors import to_rgba


### behavior plotting tools
##% helper functions
def get_prop_event(behav_all_pd_original, column='switch'):
    """
    Calculates the proportion of events for each monkey and session, across blocks.
    """
    behav_all_pd = behav_all_pd_original.copy()

    monkeys = behav_all_pd['monkey'].unique()
    percent_selection = {monkey: [] for monkey in monkeys}

    # nominal length = 40 - blocks will be interpolated to this length
    LEN_BLOCK = 40

    # fill nans and convert to float
    behav_all_pd[column] = behav_all_pd[column].fillna(method='ffill')
    behav_all_pd[column] = behav_all_pd[column].fillna(method='bfill')
    behav_all_pd[column] = behav_all_pd[column].astype(float)  # see if its best target

    for monkey, behav_monkey in behav_all_pd.groupby('monkey'):
        for session, behav_session in behav_monkey.groupby('session'):
            for block_id, behav_block in behav_session.groupby('block_id'):
                if len(behav_block) < 35:
                    continue

                # get best target selection and shift vectors per block
                column_vector = behav_block[column]

                # interpolate to LEN_BLOCK
                column_vector = np.interp(np.linspace(0, LEN_BLOCK, LEN_BLOCK), np.linspace(0, LEN_BLOCK, len(column_vector)), column_vector)

                # append to list
                percent_selection[monkey].append(column_vector)

        # create array of best target selection
        percent_selection[monkey] = np.array(percent_selection[monkey])
        #percent_shift[monkey] = np.array(percent_shift[monkey])
        
    return percent_selection# percent_shift


def plot_strategy(behav, ax=None, saveas=None, paper_format=False, title=None, show_error=True):

    # get best target selection and shift vectors
    behav = add_switch_info(behav)
    behav['if_best_target'] = behav['target'] == behav['best_target']

    percent_selection = get_prop_event(behav, 'if_best_target')
    percent_shift = get_prop_event(behav, 'switch')

    monkeys = percent_selection.keys()

    ### Print stats
    # percent correct for each monkey and together
    for monkey in monkeys:
        print(f'{monkey} - percent correct: {np.mean(percent_selection[monkey]):.2f}')
        print(f'{monkey} - percent shift: {np.mean(percent_shift[monkey]):.2f}')
    print(f'Both monkeys - percent correct: {np.mean(np.concatenate(list(percent_selection.values()), axis=0)):.2f}')
    print(f'Both monkeys - percent shift: {np.mean(np.concatenate(list(percent_shift.values()), axis=0)):.2f}')
    

    LEN_BLOCK = percent_selection[list(percent_selection.keys())[0]].shape[1]
    if ax is None:
        # plotting, figsize is 130mm x 9 mm
        cm = 1/2.54  # centimeters in inches
        if paper_format:
            # fontsize = 18
            plt.rcParams.update({'font.size': 18})
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(20*cm, 8*cm))   

    if title is not None:
        fig.suptitle(title)
             
    for monkey in monkeys:
        # get mean and std for best target selection and shift
        mean_best_selection = np.mean(percent_selection[monkey], axis=0)
        std_best_selection = np.std(percent_selection[monkey], axis=0)
        sem_best_selection = std_best_selection / np.sqrt(len(percent_selection[monkey]))

        mean_shift = np.mean(percent_shift[monkey], axis=0)
        std_shift = np.std(percent_shift[monkey], axis=0)
        sem_shift = std_shift / np.sqrt(len(percent_shift[monkey]))

        linestyle = 'dashed' if monkey.split('_')[-1] == 'simulation' else 'solid'
        if linestyle == 'solid':
            if monkey in COLORS.keys():
                color = COLORS[monkey]
            else:
                color = None
            if show_error:
                axs[0].fill_between(np.arange(LEN_BLOCK), mean_best_selection-sem_best_selection, mean_best_selection+sem_best_selection, color=color, alpha=0.5)
        else:
            color = COLORS['_'.join(monkey.split('_')[:-1])]
        axs[0].plot(mean_best_selection, color=color, label=monkey.upper(), linestyle=linestyle, linewidth=2)

        axs[1].plot(mean_shift, color=color , label=monkey.upper(), linestyle=linestyle, linewidth=2)
        if linestyle == 'solid' and show_error:
            axs[1].fill_between(np.arange(LEN_BLOCK), mean_shift-sem_shift, mean_shift+sem_shift, color=color, alpha=0.5)

    # plot settings
    ax = axs[0]
    ax.axvline(LEN_BLOCK-5, color='k', linestyle='--', alpha=0.5)
    #ax.set_title('Best target selection')
    ax.set_xlabel('Trials in block')
    ax.set_ylabel('Proportion of best\ntarget selection')
    #ax.set_ylim([0.2, 1])
    # set ticks every .2
    #ax.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax.legend(loc='lower right', frameon=True)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax = axs[1]
    ax.axvline(LEN_BLOCK-5, color='k', linestyle='--', alpha=0.5)
    #ax.set_title('Shift')
    ax.set_xlabel('Trials in block')
    ax.set_ylabel('Probability of shift')
    #ax.set_yticks(np.arange(0, 1.1, 0.1))
    #ax.set_ylim([-0.02, .45])
    ax.legend(loc='upper right', frameon=True)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, axs

def plot_summary_stats(behav_original, title=None):

    fig, axs = plt.subplots(1, 2, figsize=(8, 2))
    
    if title is not None:
        fig.suptitle(title)

    # plot mean +- 2 std
    
    for i, (monkey, behav_original_temp) in enumerate(behav_original.groupby('monkey')):
        behav = behav_original_temp.copy()
        behav = add_block_info(behav)
        behav['best_action'] = behav['action'] == behav['best_arm']
        behav['shift'] = behav['action'] != behav['action'].shift(1)

        # create matrices of first/last 20 trials of each block
        blocks_is_best_selection = []
        blocks_is_rewarded = []
        blocks_is_shift = []

        for block_id, behav_block in behav.groupby('block_id'):
            is_best_selection = behav_block['best_action']
            is_rewarded = behav_block['reward']
            is_shift = behav_block['shift']

            # interploate to have 40 length
            is_best_selection = np.interp(np.linspace(0, len(is_best_selection)-1, 40), np.arange(len(is_best_selection)), is_best_selection)
            is_rewarded = np.interp(np.linspace(0, len(is_rewarded)-1, 40), np.arange(len(is_rewarded)), is_rewarded)
            is_shift = np.interp(np.linspace(0, len(is_shift)-1, 40), np.arange(len(is_shift)), is_shift)

            blocks_is_best_selection.append(is_best_selection)
            blocks_is_rewarded.append(is_rewarded)
            blocks_is_shift.append(is_shift)
        
        blocks_is_best_selection = np.array(blocks_is_best_selection)
        blocks_is_rewarded = np.array(blocks_is_rewarded)
        blocks_is_shift = np.array(blocks_is_shift)
        
        axs[0].plot(np.arange(0, 40), blocks_is_best_selection.mean(axis=0), label=f'{monkey}')
        axs[1].plot(np.arange(0, 40), blocks_is_shift.mean(axis=0))

        #ax.plot(np.arange(0, 40), blocks_is_rewarded.mean(axis=0), color='tab:orange', label='p(reward)')

    ax = axs[0]

    ax.set_xlabel('trials in block')
    ax.set_ylabel('p(best arm)')

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlim(-1, 41)
    ax.set_ylim(0, 1)

    ax.axvline(35, linestyle='--', color='black', alpha=.7)
    ax.grid(alpha=.5)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.5), ncol=1)

    # plot p(shift)
    ax = axs[1]

    ax.set_ylabel('p(shift)')
    ax.set_xlabel('trials in block')

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlim(-1, 41)
    ax.set_ylim(0, .5)

    ax.axvline(35, linestyle='--', color='black', alpha=.7)
    ax.grid(alpha=.5)  

    #plt.tight_layout()
    plt.show()

def plot_performances(behavs):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for key, value in behavs.items():
        prop_best_arm = (value["best_arm"].values == value["action"].values).mean()
        mean_rr = value["reward"].mean()

        if key == 'Bayesian':
            ax.axhline(prop_best_arm, color='black', linestyle='--', label='Normative')
        else:

            if key == 'MONKEY KA':
                color = COLORS['ka']
            elif key == 'MONKEY PO':
                color = COLORS['po']
            else:
                color = 'grey'

            ax.bar(key, prop_best_arm, alpha=0.8, color=color, zorder=2)

    # rotate x-axis labels
    plt.xticks(rotation=90)

    ax.set_ylabel("Proportion of best arm choices")
    ax.set_title("Performance of different agents")
    ax.set_ylim(0.5, 1)
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

def plot_hist_thingy(behav_original, title=None):
    labels = [ f"+ + +", f"o + +" , f"+ o +", f"o o +", f"+ + o", f"o + o", f"+ o o", f"o o o"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})

    if title is not None:
        fig.suptitle(title)

    ax = ax1

    for i, (monkey, behav_data) in enumerate(behav_original.groupby('monkey')):
        behav = behav_data.copy()

        #balance for shifts (random undersampling majority class)
        '''n_switch = np.sum(behav['switch'])
        n_no_switch = len(behav) - n_switch
        n_minority = int(np.min([n_switch, n_no_switch]))
        behav = behav.groupby('switch').apply(lambda x: x.sample(n=n_minority, random_state=42)).reset_index(drop=True)'''

        # downsample each groups
        comb_counts = behav.groupby(['R_1', 'R_2', 'R_3']).size().reset_index(name='counts')
        min_count = comb_counts['counts'].min()  # Determine the minimum count among the combinations
        # Function to undersample each combination
        def undersample(group):
            return group.sample(n=min_count, random_state=42)
        behav = behav.groupby(['R_1', 'R_2', 'R_3']).apply(undersample).reset_index(drop=True)  # Apply undersampling

        all_trials_vector = []
        for (R_1, R_2, R_3), behav_temp in behav.groupby(['R_1', 'R_2', 'R_3']):
            all_trials_vector.append(behav_temp['switch'].mean())

        ax.plot(all_trials_vector, label=f'{monkey}', marker='o', color=COLORS[monkey])

    ax.set_xticks([])
    #ax.set_xticklabels(labels, rotation=90)

    ax.set_xlabel('')
    ax.set_ylabel('p(switch)')

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # legend to the right
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)

    ax = ax2
    for i, label in enumerate(labels):
        for j, marker in enumerate(label.split(' ')):        
            ax.scatter(i, j, marker='o' if marker == 'o' else 'x', c='tab:green' if marker == 'o' else 'tab:red')

    ax.set_xticks([])

    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['t-3', 't-2', 't-1'])
    ax.set_ylim(-1, 3)

    ax.set_ylabel('past trials')

    # make legend: o for rewarded, x for unrewarded
    ax.scatter([], [], color="tab:green", marker="o", label="Rewarded")
    ax.scatter([], [], color="tab:red", marker="x", label="Unrewarded")
    ax.legend(loc='lower center', bbox_to_anchor=(0.8, -0.5), ncol=1)

    ax.spines["right"].set_visible(False)
    #ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()

    return fig, (ax1, ax2)
    

def show_target_selection(session_data_original, title=None, background_value=None, savedir=None, show=True, add_phase=False):
    """
    Generates a figure illustrating the target selection, feedback, and target value.

    Parameters
    ----------
    session_data_original : pandas.DataFrame
        The original session data. 
    title : str, optional
        The title of the figure. Default is None.
    background_value : str, optional
        The name of the column in the session data that contains the value to plot in the background. Default is None.
    savedir : str, optional
        The directory to save the figure. Default is None.
    show : bool, optional
        Whether to display the figure. Default is True.

    Returns
    -------
    None
    """


    # work on a copy of the original data
    session_data = session_data_original.copy()

    # add 'trial in session' column
    if 'trial_id_in_block' not in session_data.columns:
        session_data = add_trial_in_block(session_data)  # add 'trial in block' column
    if add_phase:
        session_data = add_phase_info(session_data, exploration_limit=1)  # add 'phase' column - switch, repeat, transition between the two (which lasts for 5 trials)

    # init plot
    n_cols = 1
    n_rows = len(session_data['block_id'].unique())

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(11, 3*n_rows))

    # create subplots, 
    for i, ax in enumerate(axs.reshape(-1)):  # block by block
        no_of_trials = len(session_data.loc[session_data['block_id'] == i])  # number of trials in the block

        ### plot selected target and best target
        # get the selected target and its id, for REWARDED trials
        rewarded_trials = session_data.loc[(session_data['block_id'] == i) & (session_data['feedback'] == True)]  
        selected_target_rewarded, selected_target_rewarded_id = rewarded_trials['target'], rewarded_trials['trial_id_in_block']
        
        # get the selected target and its id, for UNREWARDED trials
        unrewarded_trials = session_data.loc[(session_data['block_id'] == i) & (session_data['feedback'] == False)]  
        selected_target_unrewarded, selected_target_unrewarded_id = unrewarded_trials['target'], unrewarded_trials['trial_id_in_block']
        
        # trial ids when the trial was interrupted
        interrupted_trials_id = session_data.loc[(session_data['block_id'] == i) & (session_data['feedback'].isnull()), 'trial_id_in_block']

        ax.scatter(selected_target_rewarded_id, selected_target_rewarded, color='black', marker='o',
                   label='rewarded')  # plot rewarded trials
        ax.scatter(selected_target_unrewarded_id, selected_target_unrewarded, color='black', marker='x',
                   label='unrewarded')  # plot unrewarded trials
        if len(interrupted_trials_id) > 0:  # plot red X for all three targets in case of interrupted trials
            ax.scatter(interrupted_trials_id, [1 for x in interrupted_trials_id], color='red', marker='s',
                       label='interrupted trial')
            ax.scatter(interrupted_trials_id, [2 for x in interrupted_trials_id], color='red', marker='s')  
            ax.scatter(interrupted_trials_id, [3 for x in interrupted_trials_id], color='red', marker='s')  

        # plot green lines marking the best target
        ax.plot(np.arange(0, no_of_trials, 1),
                np.ones((no_of_trials,)) * session_data.loc[session_data['block_id'] == i, 'best_target'],
                label='best target', color='green', alpha=.3, linewidth=20)

        # mark last 5 trials per block (here the reward probabilities gradually change)
        ax.vlines(no_of_trials - 6 + 0.5, ymin=0, ymax=30, linestyle='dashed', color='black',
                  label='start of last 5 sessions')

        if add_phase:
            # mark explore and exploit (shift, repeat, transition)
            exploit_trials = session_data.loc[
                (session_data['block_id'] == i) & (session_data['phase'] == 'repeat'),
                'trial_id_in_block']

            explore_trials = session_data.loc[
                (session_data['block_id'] == i) & (session_data['phase'] == 'search'),
                'trial_id_in_block']

            transition_trials = session_data.loc[
                (session_data['block_id'] == i) & (session_data['phase'] == 'transition'),
                'trial_id_in_block']

            for j, trial in enumerate(explore_trials):
                if j == 0:
                    ax.axvspan(trial - .5, trial + .5, facecolor='red', alpha=.2, label='explore')
                else:
                    ax.axvspan(trial - .5, trial + .5, facecolor='red', alpha=.2)
            for j, trial in enumerate(exploit_trials):
                if j == 0:
                    ax.axvspan(trial - .5, trial + .5, facecolor='blue', alpha=.2, label='exploit')
                else:
                    ax.axvspan(trial - .5, trial + .5, facecolor='blue', alpha=.2)
            for j, trial in enumerate(transition_trials):
                if j == 0:
                    ax.axvspan(trial - .5, trial + .5, facecolor='yellow', alpha=.2, label='transition')
                else:
                    ax.axvspan(trial - .5, trial + .5, facecolor='yellow', alpha=.2)

        # plot MEASURE
        if background_value is not None:
            if background_value not in session_data.columns:
                raise ValueError('background_value not in columns')
            else:
                measure_min, measure_max = session_data[background_value].min(), session_data[background_value].max()  # get min and max of the measure
                measure = session_data.loc[(session_data['block_id'] == i), background_value].to_numpy()

                ax_ = ax.twinx()  # create a twin axis
                ax_.plot(measure, color='tab:red', alpha=.6)  # plot the measure on the twin axis
                ax_.set_ylabel(background_value, color='tab:red')
                ax_.set_ylim(measure_min-np.abs(.2*measure_min),
                            measure_max+np.abs(.1*measure_max))
                ax_.tick_params(axis='y', labelcolor='tab:red')

        # plot settings
        ax.set_xlabel('trial number in block/trial number in session')
        ax.set_ylabel('selected target')
        ax.set_xlim(-0.5, 45)
        ax.set_xticks(range(0, 45, 5))
        ax.set_ylim(0.5, 3.5)
        ax.set_yticks([1, 2, 3])
        ax.title.set_text('block id: ' + str(i))

        trial_ids = session_data.loc[session_data['block_id'] == i, 'trial_id'].values
        xticks = []
        for x in range(0, 45, 5):
            if x < len(trial_ids):
                xticks.append(f'{str(x)}\n{str(trial_ids[x])}')
            else:
                xticks.append(str(x))

        ax.set_xticklabels(xticks)
        

        spacing = 1  # This can be your user specified spacing.
        from matplotlib.ticker import MultipleLocator
        minor_locator = MultipleLocator(spacing)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        # Set grid to use minor tick locations. grid in the background
        ax.grid(which='both', zorder=-1.0)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.4),
                  fancybox=True, shadow=True, ncol=5)

    plt.tight_layout()
    if title is not None:
        plt.suptitle(title, y=1)

    if savedir is not None:
        plt.savefig(savedir)
    if show:
        plt.show()
    else:
        plt.close()

def show_target_selection_compact(
        session_data_original, 
        title=None, 
        background_values=None, 
        background_value_lims=None,
        savedir=None, 
        format='paper',
        show=True):
    """
    Generates a figure illustrating the target selection, feedback, and target value.

    Parameters
    ----------
    session_data_original : pandas.DataFrame
        The original session data. 
    title : str, optional
        The title of the figure. Default is None.
    background_value : str, optional
        The name of the column in the session data that contains the value to plot in the background. Default is None.
    savedir : str, optional
        The directory to save the figure. Default is None.
    show : bool, optional
        Whether to display the figure. Default is True.

    Returns
    -------
    None
    """
    # work on a copy of the original data
    session_data = session_data_original.copy()

    if background_values is None and not isinstance(background_values, list):
        background_values = [background_values]

    if len(session_data['monkey'].unique()) != 1:
        raise ValueError('session_data should contain only one monkey')
    monkey = session_data['monkey'].unique()[0]
    v0 = MODEL_PARAMS[monkey]['V0']

    # set the colors of the targets in RGBA format
    target_colors = {1: '#90c6f2ff', 2: '#ffb273ff', 3: '#dea8ddff'}

    # add 'trial in session' column
    if 'trial_id_in_block' not in session_data.columns:
        session_data = add_trial_in_block(session_data)  # add 'trial in block' column
    
    # init plot
    cm_to_in = 0.393701
    n_rows = len(session_data['block_id'].unique())
    if format == 'paper':
        h = 2.5  # height of each block in cm
        w = 10  # width of each block in cm
        s_marker = 12
        linewidth_marker = .6
        height_ratios = [1, 3]  # for the inner grid
    elif format == 'poster':
        h = 3.5
        w = 10
        s_marker = 12
        linewidth_marker = 1
        height_ratios = [1, 5]  # for the inner grid
    else:
        h = 4
        w = 15
        s_marker = 40
        linewidth_marker = 1
        height_ratios = [1, 3]  # for the inner grid
    fig = plt.figure(figsize=(w*cm_to_in, h*n_rows*cm_to_in))
    outer_grid = plt.GridSpec(n_rows, 1)  # Create the main grid for blocks

    for i in range(n_rows):
        # Create a subdivision of the block's grid space
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=outer_grid[i],
                                                    height_ratios=height_ratios,
                                                    hspace=0)
        
        # Create the two axes for this block
        ax_markers = plt.Subplot(fig, inner_grid[0])
        ax_measure = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax_markers)
        fig.add_subplot(ax_measure)

        # set background of axmarkers to best target color
        best_target = session_data.loc[session_data['block_id'] == i, 'best_target'].values[0]
        # Convert the color to RGBA with alpha
        rgba_color = to_rgba(target_colors[best_target], alpha=0.2)

        # Set the face color of the axis
        ax_markers.set_facecolor(rgba_color)


        # plot the target selection
        for target_id in [1, 2, 3]:  # plot the targets
            # get the selected target and its id, for REWARDED trials
            rewarded_trials_of_high_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == True) & (session_data['target'] == target_id) & (session_data['best_target'] == target_id)
                ]['trial_id_in_block']
            rewarded_trials_of_low_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == True) & (session_data['target'] == target_id) & (session_data['best_target'] != target_id)
                ]['trial_id_in_block']
            unrewarded_trials_of_high_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == False) & (session_data['target'] == target_id) & (session_data['best_target'] == target_id)
                ]['trial_id_in_block']
            unrewarded_trials_of_low_target = session_data.loc[
                (session_data['block_id'] == i) & (session_data['feedback'] == False) & (session_data['target'] == target_id) & (session_data['best_target'] != target_id)
                ]['trial_id_in_block']
            
            ax_markers.scatter(
                rewarded_trials_of_high_target,
                np.ones_like(rewarded_trials_of_high_target),
                color=target_colors[target_id], marker='o', facecolors=target_colors[target_id], s=s_marker, linewidth=linewidth_marker)  # plot rewarded trials
            ax_markers.scatter(
                rewarded_trials_of_low_target,
                np.ones_like(rewarded_trials_of_low_target),
                color=target_colors[target_id], marker='o', facecolors=target_colors[target_id], s=s_marker, linewidth=linewidth_marker)  # plot rewarded trials
            ax_markers.scatter(
                unrewarded_trials_of_high_target,
                np.ones_like(unrewarded_trials_of_high_target),
                color=target_colors[target_id], marker='o', facecolors='none', s=s_marker, linewidths=linewidth_marker)
            ax_markers.scatter(
                unrewarded_trials_of_low_target,
                np.ones_like(unrewarded_trials_of_low_target),
                color=target_colors[target_id], marker='o', facecolors='none', s=s_marker, linewidths=linewidth_marker)
            
        # plot interrupted trials
        interrupted_trials_id = session_data.loc[
            (session_data['block_id'] == i) & (session_data['feedback'].isnull())
            ]['trial_id_in_block']

        if len(interrupted_trials_id) > 0:
            ax_markers.scatter(
                interrupted_trials_id,
                np.ones_like(interrupted_trials_id),
                color='black', marker='x', label='interrupted trial')
       
        '''# plot green lines marking the best target
        ax.plot(np.arange(0, no_of_trials, 1),
                np.ones((no_of_trials,)) * session_data.loc[session_data['block_id'] == i, 'best_target'],
                label='best target', color='green', alpha=.3, linewidth=20)'''

        # plot MEASURE (i.e. value)
        bg_val_colors = {'stay_value': 'grey', 'V0': 'grey', 
                         'Q_1': target_colors[1], 'Q_2': target_colors[2], 'Q_3': target_colors[3]}
        alphas = {'V0': 0.4}
        if background_values is not None:
            for background_value in background_values:
                measure = session_data.loc[(session_data['block_id'] == i), background_value].to_numpy()
                color = bg_val_colors.get(background_value, 'grey')
                alpha = alphas.get(background_value, .7)
                ax_measure.plot(measure, color=color, alpha=alpha)  # plot the measure on the twin axis

        # mark last 5 trials per block (here the reward probabilities gradually change)
        no_of_trials = len(session_data.loc[session_data['block_id'] == i])  # number of trials in the block
        for ax in [ax_markers, ax_measure]:
            ax.axvline(no_of_trials - 6 + 0.5, linestyle='dashed', 
                        color='black', alpha=0.5)

        ## PLOT SETTINGS
        #ax_markers.title.set_text('block id: ' + str(i))

        ax_markers.set_xlim(-0.5, 45)
        ax_markers.set_ylim(0.8, 1.2)
        ax_markers.set_xticks(range(0, 45, 10))
        ax_markers.set_xticklabels([])
        ax_markers.set_yticks([])
        ax_markers.set_ylabel('')

        ax_measure.set_xlim(-0.5, 45)
        if background_value_lims is None:
            ax_measure.set_ylim(-.05, 1.05)
            ax_measure.axhline(0, color='grey', linewidth=0.5, linestyle='dashed')
            ax_measure.axhline(1, color='grey', linewidth=0.5, linestyle='dashed')
        else:
            ax_measure.set_ylim(background_value_lims)
            ax_measure.axhline(0, color='grey', linewidth=0.5, linestyle='-')

        if i == n_rows - 1:
            ax_measure.set_xlabel('trials in block (trials in session)')
        if background_value is not None:
            ax_measure.set_ylabel(background_value)

        # set xticks to trial ids
        trial_ids = session_data.loc[session_data['block_id'] == i, 'trial_id'].values
        xticks = []
        for x in range(0, 45, 10):
            if x < len(trial_ids):
                xticks.append(f'{str(x+1)} ({str(trial_ids[x]+1)})')
            else:
                xticks.append(str(x+1))
        ax_measure.set_xticks(range(0, 45, 10))
        ax_measure.set_xticklabels(xticks)

        # remove spines
        ax_markers.spines['top'].set_visible(False)
        ax_markers.spines['right'].set_visible(False)
        ax_markers.spines['bottom'].set_visible(False)

        ax_measure.spines['top'].set_visible(False)
        ax_measure.spines['right'].set_visible(False)
        ax_measure.spines['bottom'].set_visible(False)

        '''ax_measure.grid()
        ax_markers.grid()'''

    # add legend to last subplot, below in the middle
    ax_markers.scatter([], [], color='grey', marker='o', facecolors='grey', s=30, label='Rewarded ')
    ax_markers.scatter([], [], color='grey', marker='o', facecolors='none', s=30, label='Non-Rewarded ')
    ax_markers.scatter([], [], color=target_colors[1], marker='o', label='target 1')   
    ax_markers.scatter([], [], color=target_colors[2], marker='o', label='target 2')
    ax_markers.scatter([], [], color=target_colors[3], marker='o', label='target 3')
    ax_markers.legend(loc='center', bbox_to_anchor=(.5, -8), ncol=1)

    plt.tight_layout()
    if title is not None:
        plt.suptitle(title, y=1)

    if savedir is not None:
        plt.savefig(savedir, dpi=300, bbox_inches='tight', transparent=False)
    if show:
        plt.show()
    else:
        plt.close()


def _get_ticks(n_extra_trials):
    n, m = n_extra_trials

    # Base ticks and labels for the current trial
    base_ticks = [0, 1, 2, 2.5, 3, 3.5, 7.5]
    base_labels = ['st', 'Lt', 'Lv', 'Tt', 'Tv', 'Fb']

    # Generate ticks and labels for extra trials
    xticks = []
    xticklabels = []

    # Add ticks and labels for past trials
    for i in range(n, 0):
        offset = i * (base_ticks[-1])
        xticks.extend([offset + tick for tick in base_ticks[:-1]])
        xticklabels.extend([f'${label} (t{i})$' for label in base_labels])

    # Add ticks and labels for the current trial
    xticks.extend(base_ticks[:-1])
    xticklabels.extend([f'${label} (t)$' for label in base_labels])

    # Add ticks and labels for future trials
    for i in range(1, m + 1):
        offset = i * (base_ticks[-1])
        xticks.extend([tick + offset for tick in base_ticks[:-1]])
        xticklabels.extend([f'${label} (t+{i})$' for label in base_labels])

    # Add the final tick and label for the last future trial
    xticks.append(base_ticks[-1] + m * base_ticks[-1])
    xticklabels.append(f'$fin (t+{m})$' if m > 0 else '$fin (t)$')
    
    return xticks, xticklabels
    
    
def plot_keypoints(ax=None, n_extra_trials=(0, 0), n_trials=None, mark_event='none', fontsize=None, axis='x', rotation=90, xlabels='both'):
    '''
    Plot the key points of the task on the x-axis of the plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the key points on.
    n_extra_trials : int, optional
        The number of extra trials to plot, can be either positiv or negativ or 0. The default is 1.
    n_trials : int, optional
        Deprecated. Use n_extra_trials instead.
    fontsize : int, optional
        The fontsize of the xticks. The default is None.
    axis : str, optional
        The axis to plot the key points on, can be either 'x', 'y' or 'both' (both for plotting e.g. cross-correlation matrices). The default is 'x'.
    rotation : int, optional
        The rotation of the xticks. The default is 90.
    xlabels : str, optional
        Weather to plot the xlabels as events or timepoints. Can be 'events' or 'time' or 'both'. The default is 'events'.
    '''    

    assert n_trials is None, "n_trials is deprecated. Use n_extra_trials instead."
    assert isinstance(n_extra_trials, tuple), "n_extra_trials must be a tuple."    
    
    if ax is None:
        fig, ax = plt.subplots()    
    
    xticks, xticklabels = _get_ticks(n_extra_trials)

    # it is possible toa dd the keypoints on the y axis as well (e.g. cross correlation matrices, etc..)
    if axis == 'x' or axis == 'both':
        ax.set_xticks(xticks)
        if xlabels == 'events':
            ax.set_xticklabels(xticklabels, fontsize=fontsize, rotation=rotation)
        elif xlabels == 'time':
            ax.set_xticklabels([str(tick_) for tick_ in xticks], fontsize=fontsize, rotation=rotation)
            ax.set_xlabel('time (s)')
        elif xlabels == 'both':
            x_ticks_combined = [f"{label_} | {tick_}s" for label_, tick_ in zip(xticklabels, xticks)]
            ax.set_xticklabels(x_ticks_combined, fontsize=fontsize, rotation=rotation)
            ax.set_xlabel('time (s)')
    if axis == 'y' or axis == 'both':
        ax.set_yticks(xticks)
        if xlabels == 'events':
            ax.set_yticklabels(xticklabels, fontsize=fontsize, rotation=rotation)
        elif xlabels == 'time':
            ax.set_yticklabels([str(tick_) for tick_ in xticks], fontsize=fontsize, rotation=rotation)
            ax.set_ylabel('time (s)')
        elif xlabels == 'both':
            y_ticks_combined = [f"{label_} | {tick_}s" for label_, tick_ in zip(xticklabels, xticks)]
            ax.set_yticklabels(y_ticks_combined, fontsize=fontsize, rotation=90-rotation)
            ax.set_ylabel('time (s)')

    if mark_event != 'none':
        for xtick_temp, xtick_label_temp in zip(xticks, xticklabels):
            if xtick_label_temp.replace("$", "").split(' ')[0] == mark_event:
                if axis == 'x' or axis == 'both':
                    ax.axvline(xtick_temp, color='black')
                if axis == 'y' or axis == 'both':
                    ax.axhline(xtick_temp, color='black')
    return ax


# grid plotting

def plot_matrix(matrix, monkey, ax=None, title=None, save=False, show=True):  # NOT HERE ANYMORE
    # raise deprecation warning
    import warnings

    warnings.warn("This method is replaced in a new folder. Please use the new method from popy.plotting.plot_cortical_grid instead.", DeprecationWarning)
    from popy.plotting.plot_cortical_grid import plot_matrix as plot_matrix_new
    plot_matrix_new(matrix, monkey, ax, title, save, show)

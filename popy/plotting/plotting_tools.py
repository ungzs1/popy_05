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
from popy.behavior_data_tools import add_trial_in_block, add_phase_info

import os
import warnings


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
        background_value=None, 
        background_value_lims=None,
        savedir=None, 
        paper_format=True,
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

    # set the colors of the targets in RGBA format
    target_colors = {1: '#90c6f2ff', 2: '#ffb273ff', 3: '#dea8ddff'}

    # add 'trial in session' column
    if 'trial_id_in_block' not in session_data.columns:
        session_data = add_trial_in_block(session_data)  # add 'trial in block' column
    
    # init plot
    cm_to_in = 0.393701
    n_rows = len(session_data['block_id'].unique())
    if paper_format:
        h = 2.5  # height of each block in cm
        w = 10  # width of each block in cm
        s_marker = 12
        linewidth_marker = .6
    else:
        h = 4
        w = 15
        s_marker = 20
        linewidth_marker = 1
    fig = plt.figure(figsize=(w*cm_to_in, h*n_rows*cm_to_in))
    outer_grid = plt.GridSpec(n_rows, 1)  # Create the main grid for blocks

    for i in range(n_rows):
        # Create a subdivision of the block's grid space
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                    subplot_spec=outer_grid[i],
                                                    height_ratios=[1, 3],
                                                    hspace=0)
        
        # Create the two axes for this block
        ax_markers = plt.Subplot(fig, inner_grid[0])
        ax_measure = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax_markers)
        fig.add_subplot(ax_measure)

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
                color=target_colors[target_id], marker='o', facecolors='none', s=s_marker, linewidth=linewidth_marker)  # plot rewarded trials
            ax_markers.scatter(
                unrewarded_trials_of_high_target,
                np.ones_like(unrewarded_trials_of_high_target),
                color=target_colors[target_id], marker='x', facecolors=target_colors[target_id], s=s_marker, linewidths=linewidth_marker)
            ax_markers.scatter(
                unrewarded_trials_of_low_target,
                np.ones_like(unrewarded_trials_of_low_target),
                color=target_colors[target_id], marker='X', facecolors='none', s=s_marker, linewidths=linewidth_marker)
            
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
        measure = session_data.loc[(session_data['block_id'] == i), background_value].to_numpy()
        ax_measure.plot(measure, color='grey', alpha=.6)  # plot the measure on the twin axis

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
    ax_markers.scatter([], [], color='grey', marker='o', facecolors='grey', s=30, label='Rewarded - HIGH target')
    ax_markers.scatter([], [], color='grey', marker='o', facecolors='none', s=30, label='Rewarded - LOW target')
    ax_markers.scatter([], [], color='grey', marker='X', facecolors='grey', s=30, label='Unrewarded - HIGH target')
    ax_markers.scatter([], [], color='grey', marker='X', facecolors='none', s=30, label='Unrewarded - LOW target')
    ax_markers.legend(loc='center', bbox_to_anchor=(.5, -8), ncol=2)

    plt.tight_layout()
    if title is not None:
        plt.suptitle(title, y=1)

    if savedir is not None:
        plt.savefig(savedir, dpi=300, bbox_inches='tight', transparent=True)
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
    
    
def plot_keypoints(ax=None, n_extra_trials=(0, 0), n_trials=None, fontsize=None, axis='x', rotation=90, xlabels='both'):
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

    return ax


# grid plotting

def plot_matrix(matrix, monkey, ax=None, title=None, save=False, show=True):  # NOT HERE ANYMORE
    # raise deprecation warning
    import warnings

    warnings.warn("This method is replaced in a new folder. Please use the new method from popy.plotting.plot_cortical_grid instead.", DeprecationWarning)
    from popy.plotting.plot_cortical_grid import plot_matrix as plot_matrix_new
    plot_matrix_new(matrix, monkey, ax, title, save, show)

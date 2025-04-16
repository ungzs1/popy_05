import sys
sys.path.append("C:\ZSOMBI\OneDrive\PoPy")
sys.path.append("/Users/zsombi/OneDrive/PoPy")

import pandas as pd
import numpy as np

from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.population_decoders import *
from popy.plotting.plotting_tools import *
import popy.config as cfg

from matplotlib.patches import Rectangle, FancyBboxPatch


## Data processings

def mask_N_consecutives(data, N=4, non_signif_coding='nan'):
    if non_signif_coding == 'nan':
        def criteria(num):
            return num == num
    elif non_signif_coding == '0':
        def criteria(num):
            return num != 0

    # find where N consecutive values are False
    mask = np.empty(data.shape)
    for u, unit_data in enumerate(data.values):
        consecutive_count = 0
        candidate_ids = []
        unit_signif_ids = []
        for i, num in enumerate(unit_data):
            # if not nan
            if criteria(num):
                consecutive_count += 1
                candidate_ids.append(i)  # add to candidate indices (may or may not be part of N consecutive bins)
                if consecutive_count >= N:
                    unit_signif_ids += candidate_ids  # add to list of significant indices
                    candidate_ids = []  # reset candidate ids
            else:  # reset
                consecutive_count = 0
                candidate_ids = []
        # create mask for this unit 
        unit_signif = np.zeros_like(unit_data)
        unit_signif[unit_signif_ids] = 1

        mask[u] = unit_signif

    #data_masked = data.where(mask == 1)
    if non_signif_coding == 'nan':
        return data.where(mask == 1)
    elif non_signif_coding == '0':
        return data * mask


# Plotting

def plot_matrix(matrix, monkey, ax=None, title=None, save=False, show=True):  # TODO -- WRITE THIS FUNCTION BACK TO THE POPY PACKAGE!!!!!!!!!!!!!!!!!
    # load sulcus png
    floc = '/Users/zsombi/Desktop/sulcus.png'
    img = plt.imread(floc)

    measures = {'feedback': 'accuracy',
                'value_function': 'R2 score',
                'target': 'accuracy'}

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    #default = cfg.GRID_DEFAULT[monkey]

    # plot heatmap with colorbar
    matrix = matrix.to_numpy()
    significants = matrix.copy()
    significants[significants <= 0] = np.nan
    non_significants = matrix.copy()
    non_significants[non_significants > 0] = np.nan
    # where its not nan, set to 0
    non_significants[~np.isnan(non_significants)] = 0

    # plot heatmap
    im = ax.imshow(significants, cmap='Reds', aspect='auto', origin='upper', extent=[-9, 9, -9, 9])
    # plot grey square where its not significant
    ax.imshow(non_significants, cmap='Greys', alpha=.1, aspect='auto', origin='upper', extent=[-9, 9, -9, 9], vmin=-.1, vmax=.1)

    # plot sulcus
    ax.imshow(img, origin='lower', extent=[-10, 10, -10, 10])
    
    ax.set(xlim=(-9, 9), ylim=(-9, 9), aspect='equal')

    # colorbar, ticks at every .1
    cbar = plt.colorbar(im, ax=ax, fraction=.03, pad=0.04, label=measures[title])

    # put text on top of the grid (LPFC MCC)
    pos_1 = [5, 7]
    pos_2 = [-4, -8]
    ax.text(pos_1[0], pos_1[1], 'MCC', ha="center", va="center", color="black", fontsize=20)
    ax.text(pos_2[0], pos_2[1], 'LPFC', ha="center", va="center", color="black", fontsize=20)

    # text on lateral, medial, anterior, posterior
    ax.text(-8, -10.5, 'post.', ha="center", va="center", color="black", fontsize=12)
    ax.text(8, -10.5, 'ant.', ha="center", va="center", color="black", fontsize=12)
    ax.text(-10.5, -8, 'lat.', ha="center", va="center", color="black", fontsize=12, rotation=90)
    ax.text(-10.5, 8, 'med.', ha="center", va="center", color="black", fontsize=12, rotation=90)

    # show minor ticks at every 1, major at every 2
    ax.set_xticks(np.arange(-8, 9, 1), minor=True)
    ax.set_yticks(np.arange(-8, 9, 1), minor=True)
    ax.set_xticks(np.arange(-8, 9, 2), minor=False)
    ax.set_yticks(np.arange(-8, 9, 2), minor=False)
    ax.grid(True, which='both', axis='both', linestyle='-', linewidth=1, color='grey', alpha=0.05)

    # no spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(False)
    ax.spines['bottom'].set_linewidth(False)

    # plot colorbar
    

    # set title
    if title is not None:
        ax.set_title(title)
    if save:
        #plt.savefig(os.path.join(savedir, f"{title}.pdf"))
        raise NotImplementedError
    if show:
        plt.show()


def get_threshold(score_matrix, target):
    # get scores for the target
    scores_temp = score_matrix.loc[score_matrix.target == target, 'score_matrix'].values
    # stack
    scores_temp = np.vstack(scores_temp)
    print(scores_temp.shape)

    # min value: min of non-zero values
    # max value: max of all values
    min = scores_temp[scores_temp>0].min()
    max = scores_temp.max()
    return [min, max]


def plot_score_matrices(score_matrices, n_trials=2):
    # init some variables
    cmaps = {"LPFC": "Blues", "MCC": "binary"}  # colormaps for LPFC and MCC
    thresholds = {"feedback": [.6, get_threshold(score_matrices, 'feedback')[1]] if 'feedback' in score_matrices.target.values else [0.0, 0.0],
                  "value_function": [0.0, get_threshold(score_matrices, 'value_function')[1]-.1] if 'value_function' in score_matrices.target.values else [0.0, 0.0],
                  "target": [.4, get_threshold(score_matrices, 'target')[1]] if 'target' in score_matrices.target.values else [0.0, 0.0],
                  "value_anna": [0.0, get_threshold(score_matrices, 'value_anna')[1]-.1] if 'value_anna' in score_matrices.target.values else [0.0, 0.0]}
    
    time_of_interest = {"feedback": [3.9],
                        "value_function": [11],
                        "value_anna": [11],
                        "target": [2.5]}
    # plot 
    cm = 1/2.54  # centimeters in inches
    fig, axs = plt.subplots(2, 3, figsize=(38*cm, 15*cm), sharey=True)
    plt.rcParams.update({'font.size': 12})

    #time_vector = np.linspace(0, 7.5*n_trials, score_matrices.score_matrix.values[0].shape[1])
    # time vector starts at 0, has N=score_matrices.score_matrix.values[0].shape[1] elements, and has step size of 0.1
    time_vector = np.arange(0, score_matrices.score_matrix.values[0].shape[1]*.1, .1)
    fbs = [3.5, 11]

    for j, (target, target_df) in enumerate(score_matrices.groupby('target')):
        # plot
        for k, area in enumerate(["LPFC", "MCC"]):
            ax = axs[k, j]
            
            # get data
            curr_heatmap = target_df.loc[target_df.area == area, 'score_matrix'].values[0]
            # reorder heatmap: sort by max value at time of interest
            bin_of_interest = int(time_of_interest[target][0]/.1) if target in time_of_interest.keys() else 0
            curr_heatmap = curr_heatmap[np.argsort(curr_heatmap[:, bin_of_interest])][::-1]
            #curr_heatmap = curr_heatmap[np.argsort(np.max(curr_heatmap, axis=1))][::-1]

            # plot summary
            #ax.twinx().plot(time_vector, np.count_nonzero(), color='r', label='#significant sessions')

            # plot heatmap
            im = ax.imshow(curr_heatmap, cmap=cmaps[area], aspect='auto',
                        vmin=thresholds[target][0] if target in thresholds.keys() else 0,
                        vmax=thresholds[target][1] if target in thresholds.keys() else 1,
                        origin='lower', extent=[time_vector[0], time_vector[-1], 0, len(curr_heatmap)])
            for fb in fbs:
                ax.axvline(x=fb, color='grey', alpha=.5, linewidth=1.5)

            # show time of interest
            ax.axvline(x=time_of_interest[target][0] if target in time_of_interest.keys() else 0,
                        color='purple', linestyle='dashed', alpha=.5, linewidth=2)

            #ax.set_title(f'{variable} {area}')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('No. session', fontsize=12)

            ax.set_title(f'{area}, {target}')
            
            plt.colorbar(im, ax=ax)
            # hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.legend(loc='upper right')

            ax.set_yticks(np.arange(0, len(curr_heatmap), 10))
    #plt.suptitle(f'{monkey} all sessions. Significance per condition set as: mean + .25 std')
    plt.tight_layout()
    plt.show()

    # save as vector graphics
    #     fig.savefig(os.path.join('/Users/zsombi/OneDrive/PoPy/figs/neurofrance', f'decoders_stat_new.svg'), format='svg', dpi=600)


def get_epoch_lens():
    # label epochs
    labels = ['st', 'Lt', 'Lv', 'Tt', 'Tv', 'fb', 'fin']
    ep_times = [0, 1, 2, 2.5, 3, 3.5, 7.5]
    ep_times = {labels[i]: ep_times[i] for i in range(len(labels))}

    return ep_times


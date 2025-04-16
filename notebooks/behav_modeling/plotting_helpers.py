

# @title imports
from popy.simulation_tools import *
from popy.io_tools import *
from popy.behavior_data_tools import *

from simulation_helpers import *
from popy.config import COLORS


# TODO THESE should be in the popy library!
def _add_block_info(behav):
    curr_best_target = -1
    trial_in_block_counter = -1
    block_counter = -1
    trial_in_block_vector = []
    block_id_vector = []
    for i, row in behav.iterrows():
        if row['best_arm'] != curr_best_target:
            trial_in_block_counter = 0
            block_counter += 1
            curr_best_target = row['best_arm']
        else:
            trial_in_block_counter += 1

        trial_in_block_vector.append(trial_in_block_counter)
        block_id_vector.append(block_counter)
    
    behav['block_id'] = block_id_vector
    behav["trial_in_block"] = trial_in_block_vector

    return behav

def _add_history(behav):
    history_r_m1 = []
    history_r_m2 = []
    history_r_m3 = []
    for block_id, behav_block in behav.groupby('block_id'):
        history_r_m1.append(behav_block['reward'].shift(1))
        history_r_m2.append(behav_block['reward'].shift(2))
        history_r_m3.append(behav_block['reward'].shift(3))

    behav['r_m1'] = np.concatenate(history_r_m1)
    behav['r_m2'] = np.concatenate(history_r_m2)
    behav['r_m3'] = np.concatenate(history_r_m3)
    
    return behav

def _add_switch_info(behav):
    switch = []
    for block_id, behav_block in behav.groupby('block_id'):
        switch.append(behav_block['action'] != behav_block['action'].shift(1))

    switch =  np.concatenate(switch)
    behav['switch'] = np.concatenate([np.array([np.nan]), switch[1:]])

    return behav

def _add_trial_since_switch(behav):
    trial_since_switch = -1
    trial_since_switch_vector = []
    for i, row in behav.iterrows():
        trial_since_switch_vector.append(trial_since_switch)
        if row['switch'] == 1:
            trial_since_switch = 0
        else:
            trial_since_switch += 1

    trial_since_switch_vector[0]  = np.nan
    behav['trials_since_witch'] = trial_since_switch_vector
    return behav



def plot_summary_stats(behav_original, title=None):

    fig, axs = plt.subplots(1, 2, figsize=(8, 2))
    
    if title is not None:
        fig.suptitle(title)

    # plot mean +- 2 std
    
    # if its a pandas
    if isinstance(behav_original, pd.DataFrame):
        behav_original = {'': behav_original}

    for i, (key, behav_original_temp) in enumerate(behav_original.items()):
        behav = behav_original_temp.copy()
        behav = _add_block_info(behav)
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
        
        axs[0].plot(np.arange(0, 40), blocks_is_best_selection.mean(axis=0), label=key)
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

    # if its one pandas dataframe
    if isinstance(behav_original, pd.DataFrame):
        behav_original = {'': behav_original}
        
    for i, (key, behav_original_temp) in enumerate(behav_original.items()):
        behav = behav_original_temp.copy()
        behav = _add_history(behav)
        behav = _add_switch_info(behav)
        behav = _add_trial_since_switch(behav)
        behav = behav.dropna()

        #balance for shifts (random undersampling majority class)
        '''n_switch = np.sum(behav['switch'])
        n_no_switch = len(behav) - n_switch
        n_minority = int(np.min([n_switch, n_no_switch]))
        behav = behav.groupby('switch').apply(lambda x: x.sample(n=n_minority, random_state=42)).reset_index(drop=True)'''

        # downsample each groups
        comb_counts = behav.groupby(['r_m1', 'r_m2', 'r_m3']).size().reset_index(name='counts')
        min_count = comb_counts['counts'].min()  # Determine the minimum count among the combinations
        # Function to undersample each combination
        def undersample(group):
            return group.sample(n=min_count, random_state=42)
        behav = behav.groupby(['r_m1', 'r_m2', 'r_m3']).apply(undersample).reset_index(drop=True)  # Apply undersampling

        all_trials_vector = []
        for (r_m1, r_m2, r_m3), behav_temp in behav.groupby(['r_m1', 'r_m2', 'r_m3']):
            all_trials_vector.append(behav_temp['switch'].mean())

        ax.plot(all_trials_vector, label=key, marker='o')

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
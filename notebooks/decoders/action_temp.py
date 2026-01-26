# %%
import sys
sys.path.append("C:\ZSOMBI\OneDrive\PoPy")
sys.path.append("/Users/zsombi/OneDrive/PoPy")

import pandas as pd

from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.population_decoders import *
from popy.plotting.plotting_tools import *
from popy.decoding.decoder_tools import *
from popy.dim_reduction import *
import popy.config as cfg
import popy.config as cfg

def plot_trajectory_fb_value(x, y_1, y_2, ep_times, y1_name='', ax=None, title=None, savedir=None):
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange'}
    linestyles = {0: 'dotted', 1: 'dashed', 2: 'solid'}
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for i, y1_curr in enumerate(np.unique(y_1)):
        for j in [2,1,0]:
            # valid fbs
            y1_curr_ids = np.where(y_1==y1_curr)[0]
            # values that are between 0 and 0.5
            y2_curr_ids = np.where((y_2 > j*0.33) & (y_2 < (j+1)*0.33))[0]
            # get intersection
            ids = np.intersect1d(y1_curr_ids, y2_curr_ids)
            # plot mean trajectory
            ax.plot(x[ids].mean(axis=0), color=colors[y1_curr], linestyle=linestyles[j], label=f'{y1_name} {y1_curr}, value {j}/2')

    # plot epoch times
    for label, time in ep_times.items():
        ax.axvline(time, color='grey' if label != 'fb' else 'tab:red', linestyle='dashed', alpha=.5, linewidth=1)
    times_ticks = list(ep_times.values())[:-1]
    labels_ticks = list(ep_times.keys())[:-1]
    ax.set_xticks(times_ticks)
    ax.set_xticklabels(labels_ticks)

    # vertical at 350
    #ax.set_xlim(240, 550)
    ax.axhline(0, color='grey', alpha=0.5)
    ax.set_xlabel('time (bins)')
    ax.set_ylabel('neural activity along feedback subspace')
    if title:
        ax.set_title(title)
    # legend next to figure
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if savedir:
        plt.savefig(savedir, bbox_inches='tight')


def transform_data(session_data, neural_data_area, mode=None, cut_time=None):
    pca = fit_model(session_data, neural_data_area, mode=mode, cut_time=cut_time)

    PCA_data = neural_data_area.copy(data=pca.transform(neural_data_area.T).T)
    PCA_data.attrs = neural_data_area.attrs
    # change "unit" coordinate to "PC"
    PCA_data = PCA_data.rename({'unit': 'component'})
    # add PC number as coordinate
    PCA_data['component'] = np.arange(PCA_data.shape[0])
    # add explained variance to component dimension
    PCA_data = PCA_data.assign_coords(expl_var=("component", pca.explained_variance_ratio_))

    return PCA_data


# available sessions
project_folder = cfg.PROJECT_PATH_LOCAL

# sessions info
sessions = pd.read_pickle(os.path.join(project_folder, 'data', 'recordings_summary.pickle'))

#  [markdown]
# # Get data
area = 'MCC'
for i, row in sessions.iterrows():
    for area in ['MCC', 'LPFC']:
        monkey, session = row.monkey, row.session
        print(f"Monkey: {monkey}, session: {session}")
        try:
            # Get behavior

            session_data = get_behavior(monkey, session)
            session_data = add_value_function(session_data)  # add action value
            session_data = add_RPE(session_data)  # add reward pred         iction error

            # clean up data
            session_data = drop_time_fields(session_data)  # remove time fields
            session_data = session_data.drop(['block_id', 'best_target'], axis=1)  # drop block_id and best_target

            # Get neural data
            # Load neural data
            out_path = os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'processed', 'rates')
            floc = os.path.join(out_path, f'neural_data_{monkey}_{session}.nc')

            save = False
            load = True

            if load:
                neural_data = xr.open_dataarray(floc)
            else:
                neural_data = get_neural_data(monkey, session, 'rates', sr=100)

                # remove trunctuated units
                neural_data = remove_trunctuated_neurons(neural_data, delay_limit=10)

                # remove low_firing units
                neural_data = remove_low_fr_neurons(neural_data, 1)
                
                # z-score neural data
                neural_data = scale_neural_data(neural_data)

                # normalize neural data in time
                neural_data = time_normalize_session(neural_data)

            if save:
                neural_data.to_netcdf(floc)
                
            ## Format data for decoding

            ## Value function
            # in order to decode Q_t+1, we need to shift the value_function column by 1
            session_data = session_data.dropna()

            # Match neural and behav trials
            # only shared ids
            neural_trials = neural_data.trial_id.values
            behav_trials = session_data.trial_id.values
            shared_trials = np.intersect1d(neural_trials, behav_trials)

            neural_data = neural_data[:, neural_data.trial_id.isin(shared_trials)]
            session_data = session_data[session_data.trial_id.isin(shared_trials)]

            neural_data_area = neural_data[neural_data.area==area]

            #modes = ['full', 'mean', 'full_cut', 'mean_cut', 'behav']
            pca_of_interest = 'full_cut'
            cut_time = ['before_feedback', 'after_feedback', 'decision'][2]

            PCA_data = transform_data(session_data, neural_data_area, mode=pca_of_interest, cut_time=cut_time)

            y_1 = session_data.target.values
            y_2 = session_data.value_function.values

            ep_times = get_epoch_lens(neural_data)
            ep_times = {k: v/neural_data.attrs['bin_size'] for k, v in ep_times.items()}

            n_pcs = 5
            savedir = f'/Users/zsombi/OneDrive/PoPy/notebooks/decoders/value_modulates_target/{monkey}_{session}_{area}.png'
            fig, axs = plt.subplots(n_pcs, 1, figsize=(10, 5*n_pcs))

            for pc, ax in enumerate(axs):
                # get data for pc
                PCA_data_temp = PCA_data[pc, :]
                # explain variance
                var = PCA_data_temp.expl_var.data

                # create dataset
                pca_data_np = np.array([PCA_data_temp[np.where(PCA_data_temp.trial_id.data == trial)[0]] for trial in shared_trials])

                plot_trajectory_fb_value(pca_data_np, y_1, y_2, ep_times=ep_times, ax=ax, 
                                        title=f'monkey {monkey}, session {session}, area {area}\nPC {pc}, explained variance {var:.2f}',
                                        savedir=savedir)

        except:
            print(f'Error with {monkey}_{session}_{area}')

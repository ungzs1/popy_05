#%% imports
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("C:\ZSOMBI\OneDrive\PoPy")

import pandas as pd

from popy.io_tools import *
from popy.dim_reduction import *
from popy.decoding.population_decoders import *
from popy.plotting.plotting_tools import *
import popy.config as cfg
from sklearn.decomposition import PCA

#%% load data, neural and behavioral
monkey, session = 'ka', '250422'
area = 'LPFC'

# load behavioral data
behav_data = get_behavior(monkey, session)
behav_data = add_value_function(behav_data)
behav_data = add_switch_info(behav_data)
behav_data = add_phase_info(behav_data, exploration_limit=6, transition_limit=1
                            )
# remove nan
behav_data = behav_data.dropna()
behav_data['value_function'] = behav_data['value_function'].shift(-1)
behav_data['switch'] = behav_data['switch'].shift(-1)
behav_data['phase'] = behav_data['phase'].shift(-1)
# replace value function: 0-.25=0, .25-.5=1, .5-.75=2, .75-1=3
behav_data['value_function'] = np.digitize(behav_data['value_function'], [0, .25, .5, .75, 1]) - 1
behav_data = drop_time_fields(behav_data)  # remove time fields
behav_data = behav_data.drop(['block_id', 'best_target'], axis=1)  # drop block_id and best_target

# Load neural data
out_path = os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'processed', 'rates')
floc = os.path.join(out_path , f'neural_data_{monkey}_{session}.nc')
neural_data = xr.open_dataarray(floc)  #load already preprocessed neual data
#neural_data = remove_low_fr_neurons(neural_data, 1)  # remove neurons with firing rate < 1Hz

#%% Preprocess data
# big question: how to normalize data??

#%% create dataset
neural_data_area = neural_data[neural_data.area == area]  # select only PFC

#modes = ['full', 'mean', 'full_cut', 'mean_cut', 'behav']
pca_of_interest = 'behav'
pca = fit_model(behav_data, neural_data_area, mode=pca_of_interest,
                behav_condition='target')

#%% plot variance
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title(f'PCA variance explained, {area} {monkey} {session}')
plt.legend()
plt.show()

#%% transform
n_components = 4

# mean PCA activity
X_cond_temp, y_cond_temp = build_dataset(neural_data_area, behav_data, starts_on='fb-1', ends_on='fb+1')
T_mean = pca.transform(np.mean(X_cond_temp, axis=0).T)  # transform mean activity, this will be subtracted from the conditional activity

# df to collect PCA trajectories for plottingresults per condition
T_all = pd.DataFrame(columns=['feedback',  'target', 'switch', 'value_function'],
                          index=[area])

for condition in T_all.columns:
    X_cond_temp, y_cond_temp = build_dataset(neural_data_area, behav_data, starts_on='fb-1', ends_on='fb+1',
                                             target_name=condition)

    T_current = {}
    for label in np.unique(y_cond_temp):
        X_tmep = np.mean(X_cond_temp[y_cond_temp == label], axis=0)  # mean activity per condition
        T_temp = pca.transform(X_tmep.T)
        #T_temp = pca.transform(X_tmep.T) - T_mean  # subtract full mean activity
        T_current[label] = T_temp[:, :n_components].T  # save only the first n_components
    T_all[condition][area] = T_current

# plot
title = f'Representation of conditions in PC space, {monkey} {session} {area} {pca_of_interest} (cond: {pca_of_interest} PCA)'
show_PCA_trajectories(neural_data, T_all, pca.explained_variance_ratio_, n_components=n_components, title=title)

#%% plot decision space
condition = 'target'
X_cond_temp, y_cond_temp = build_dataset(neural_data_area, behav_data, starts_on='trial_start', ends_on='trial_end',
                                         target_name=condition)
_, y_cond_phase = build_dataset(neural_data_area, behav_data, starts_on='trial_start', ends_on='trial_end',
                                 target_name='phase')
print(X_cond_temp.shape)
# flatten
#X_cond_temp_mean = np.mean(X_cond_temp, axis=2)
#X_cond_temp_mean = X_cond_temp_mean[200:]
#y_cond_temp = y_cond_temp[200:]
T_full = np.empty(X_cond_temp.shape)
for i in range(X_cond_temp.shape[2]):
    T_full[:, :, i] = pca.transform(X_cond_temp[:, :, i])

# print activity by condition
pc = 1
# find time of max separation
max_sep = 0
for t in range(T_full.shape[2]):
    all_activities = [np.mean(T_full[y_cond_temp == label, pc, t]) for label in np.unique(y_cond_temp)]
    max_sep_temp = np.abs(np.max(all_activities) - np.min(all_activities))
    if max_sep_temp > max_sep:
        max_sep = max_sep_temp
        t_max = t
for label in np.unique(y_cond_temp):
    plt.plot(np.mean(T_full[y_cond_temp == label, pc, :], axis=0), label=label)
# vertival line at t_max
plt.axvline(x=t_max, color='k', linestyle='--')
plt.legend()
plt.title(f'PC{pc+1} activity by condition, {monkey} {session} {area}')
plt.grid()
plt.show()

# plot T
pcs_of_interests = [1, 2]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
colors_dict = {label: colors[i] for i, label in enumerate(np.unique(y_cond_temp))}

fig, ax = plt.subplots(figsize=(10, 10))

#data = T_full[:, pcs_of_interests, t_max]  # show activity at time point of max separation
data = np.mean(T_full[:, pcs_of_interests, :], axis=2)  # show mean activity

colors_vector = [colors_dict[y] for y in y_cond_temp]
# show all points
ax.scatter(data[:, 0], data[:, 1], c=colors_vector)
# plot mean
for label in np.unique(y_cond_temp):
    # ids are y==label AND y_phase == exproit
    ids_of_interest = (y_cond_temp == label)  # & (y_cond_phase == 'repeat')
    ax.scatter(np.mean(data[ids_of_interest, 0]),
               np.mean(data[ids_of_interest, 1]),
               c=colors_dict[label], s=200)

# show time as lines
for i in range(len(data)-1):
    ax.plot(data[i:i+2, 0], data[i:i+2, 1], c='black', alpha=.1)

ax.set_xlabel(f'PC {pcs_of_interests[0]+1}')
ax.set_ylabel(f'PC {pcs_of_interests[1]+1}')
ax.set_title(f'Decision space, N trials: {T_mean.shape[0]} {monkey} {session} {area} {pca_of_interest} )')
plt.show()

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
monkey, session = 'ka', '210322'
area = 'MCC'

# load behavioral data
behav_data = get_behavior(monkey, session)
behav_data = add_value_function(behav_data)
behav_data = add_switch_info(behav_data)
# remove nan
behav_data = behav_data.dropna()
behav_data['value_function'] = behav_data['value_function'].shift(-1)
behav_data['switch'] = behav_data['switch'].shift(-1)
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
# should realize it multiple ways:
# 1. full dataset: all neurons X all trials' all timepoints
# 2. mean dataset: all neurons X trial timepoints (mean across trials)
#   2+ behavioral dataset: all neurons X trial timepoints per condition concatenated (mean across trials BY CONDITION, )
neural_data_area = neural_data[neural_data.area == area]  # select only PFC

modes = ['full', 'mean', 'full_cut', 'mean_cut', 'behav']
PCAs = {}
for mode in modes:
    PCAs[mode] = fit_model(behav_data, neural_data_area, mode=mode, behav_condition='target')
    print(f"pca {mode} shape: ", PCAs[mode])

# transforming: projecting the same data onto the PCA axes as we used for decoding (or just the mean, idk if PCA -> mean is the same as mean -> PCA)

#%% plot variance
for i, (key, pca) in enumerate(PCAs.items()):
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color=f'C{i}', label=key, alpha=.5)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title(f'PCA variance explained, {area} {monkey} {session}')
plt.legend()
plt.show()

#%% transform
# simmilar plot as the decoders, but instead of the accuracy under conditions,
# we plot the activities along PCs (mean across trials)
n_components = 5
pca_of_interest = 'full'
pca_curr = PCAs[pca_of_interest]

T_all = pd.DataFrame(columns=['feedback',  'target', 'switch', 'value_function'],
                          index=['both'])
# mean PCA activity
X_cond_temp, y_cond_temp = build_dataset(neural_data_area, behav_data, starts_on='fb-1', ends_on='fb+1')
X_mean = np.mean(X_cond_temp, axis=0)
T_mean = pca_curr.transform(X_mean.T)

for condition in T_all.columns:
    X_cond_temp, y_cond_temp = build_dataset(neural_data_area, behav_data, starts_on='fb-1', ends_on='fb+1',
                                             target_name=condition)

    T_current = {}
    for label in np.unique(y_cond_temp):
        X_tmep = np.mean(X_cond_temp[y_cond_temp == label], axis=0)
        T_temp = pca_curr.transform(X_tmep.T) - T_mean
        T_current[label] = T_temp[:, :n_components].T
    T_all[condition][area] = T_current

#%% plot
title = f'Representation of conditions in PC space, {monkey} {session} {area} {pca_of_interest} (cond: {pca_of_interest} PCA)'
show_PCA_trajectories(neural_data, T_all, PCAs[pca_of_interest].explained_variance_ratio_, n_components=4, title=title)


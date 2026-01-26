from sklearn.decomposition import PCA
import numpy as np
import xarray as xr
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as lines

from popy.decoding.population_decoders import *
    

class PcaCustom:
    def __init__(self, neural_data, behav_data, mode='full', time_of_interest=None, behav_condition=None):
        self.neural_data = neural_data
        self.behav_data = behav_data
        self.mode = mode
        self.time_of_interest = time_of_interest 
        self.behav_condition = behav_condition
        
        self.PCA = PCA()
    
    
    def fit(self):
        # should realize it multiple ways:
        # 1. full dataset: all neurons X all trials' all timepoints
        # 2. mean dataset: all neurons X trial timepoints (mean across trials)
        #   2+ behavioral dataset: all neurons X trial timepoints per condition concatenated (mean across trials BY CONDITION, )

        # 1. full dataset
        if self.mode == 'full':
            data = self.neural_data.copy()  # copy the data to avoid changing the original data
            
            if self.time_of_interest is not None:  # if time of interest is provided, we will use it
                ids_of_interest = (data.time_in_trial >= self.time_of_interest[0]) & (data.time_in_trial < self.time_of_interest[1])  # select the time window of interest
                data = data[:, ids_of_interest]  # select the data of interest

            self.PCA.fit(data.T)

        # 2. mean dataset
        if self.mode == 'mean':
            data = self.neural_data.copy()  # copy the data to avoid changing the original data
            
            if self.time_of_interest is not None:  # if time of interest is provided, we will use it
                ids_of_interest = (data.time_in_trial >= self.time_of_interest[0]) & (data.time_in_trial < self.time_of_interest[1])  # select the time window of interest
                data = data[:, ids_of_interest]  # select the data of interest            
                
            # take mean across trials
            X_full, _ = build_dataset(data, self.behav_data)  # build dataset to get X_full
            X_mean = np.mean(X_full, axis=0)  # mean across trials
            
            self.PCA.fit(X_mean.T)

        # 3 behavioral dataset
        if self.mode == 'behav':
            '''
            In this mode, we will fit PCA on the mean neural activity of each behavioral condition. Eg. if the behavioral condition is 'feedback', we will take the trial averaged 
            neural activity of all trials with positive feedback and all trials with negative feedback and concatenate them. This results in a units X N*trial_timepoints matrix, 
            where N is the number of unique values of the behavioral conditions.
            '''
            if self.behav_condition is None:
                raise ValueError('Please provide a behavioral condition as "behav_condition" argument')
            X_cond_temp, y_cond_temp = build_dataset(self.neural_data, self.behav_data)
            X_cond = []
            for label in np.unique(y_cond_temp):
                X_cond.append(np.mean(X_cond_temp[y_cond_temp == label], axis=0))
            X_cond = np.concatenate(X_cond, axis=1)
            
            self.PCA.fit(X_cond.T)


    def transform(self, data):
        transformed_data = self.PCA.transform(data.T).T
        
        # make transformed data xarray, if data is xarray
        if isinstance(data, xr.DataArray):
            dims = data.dims
            coords = data.mean(dim='unit').coords  # take coords , but remove 'unit' dimension
            transformed_data = xr.DataArray(transformed_data, dims=dims, coords=coords)
            # add new coordinate along 'unit' dimension
            transformed_data['unit'] = [f'PC_{i}' for i in range(transformed_data.shape[0])]
            # add new coordinate along 'unit' dimension, called 'variance_explained'
            transformed_data = transformed_data.assign_coords({'variance_explained': ('unit', self.PCA.explained_variance_ratio_)})    
        
        return transformed_data

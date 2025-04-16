"""
Tools for working with (already processed) neural data. The data can either be spike trains or firing rates. 

The data is stored in xarray data structures.
"""

import pandas as pd
import xarray as xr
import math
import numbers
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.ndimage import convolve1d
import scipy.signal as scs
from tqdm import tqdm
import time


### Firing rate generation


def add_firing_rates(neural_data, method='gauss', std=.050, win_len=.200, drop_spike_trains=False):
    """
    Add firing rate fields calculated from spikes trains.
    
    Parameters
    ----------
    neural_data : xarray.Dataset
        Neural data with spike trains, must have a 'bin_size' attribute and a 'spike_trains' data array
        
    method : str, optional (default='gauss')
        Method to calculate firing rates, either 'gauss' or 'count'. 'gauss' uses a gaussian window for convolution, 'count' uses a moving average window.
        
    std : float, optional (default=.050)
        Standard deviation of the gaussian window in seconds.
        
    win_len : float, optional (default=.200)
        Length of the window in seconds.
        
    drop_spike_trains : bool, optional (default=False)
        If True, the spike trains are removed from the data.
        
    """   
    assert isinstance(neural_data, xr.Dataset), "neural_data must be an xarray Dataset."

    bin_size = neural_data.attrs['bin_size']
    bin_size = float(f'{bin_size:.4f}')  # round to 4 decimals (as it is in a wierd format where it is not exactly 0.01 or something)

    # create window for convolution (either moving average or gaussian)
    if method == "gauss":
        win_len = 10*std  # in seconds. 10 std is a good rule of thumb for window size, capturing all the signal but not too many zeros
        # measures in bins
        std_bins = int(std / bin_size)
        win_bins = int(win_len / bin_size)
        win = scs.windows.gaussian(win_bins, std_bins)
        # normalize to unit area under the curve so that the rate will be in Hz (spikes per second)
        win = (win / np.mean(win)) * (1/win_len)  
    elif method == "count":
        win_bins = int(win_len / bin_size)  # number of bins needed to cover the desired window size
        win = np.ones(win_bins)  # window for convolution to achieve moving averaging
    else:
        raise NotImplementedError(f"Method '{method}' not implemented. It should be 'gauss' or 'count'.")
    
    # calculate rates for every spike field
    firing_rates_np = get_rate(neural_data, win)
    
    # add firing rates to the data
    if method == "gauss":
        new_data_name = 'firing_rates'
    elif method == "count":
        new_data_name = 'spike_counts'
        
    # create xarray
    firing_rates_xr = xr.DataArray(firing_rates_np, 
                                   dims=list(neural_data.spike_trains.dims),
                                   coords={coord: neural_data.spike_trains[coord].data for coord in neural_data.spike_trains.dims})
    firing_rates_xr.attrs['bin_size'] = bin_size  # add bin size to the attributes
    neural_data[new_data_name] = firing_rates_xr
    return neural_data.drop('spike_trains') if drop_spike_trains else neural_data


### trial-by-trial dataset creation


def _concat_to_xr_normalized(neural_data, data_np, n_extra_trials, trials_vector):
    # get dimensions and coords of the data
    dims = ('trial_id', 'unit', 'time')
    coords = {name: ('unit', coord.values) for name, coord in neural_data['unit'].coords.items()}  # get the coords of each units
    coords['trial_id'] = trials_vector  # add trial_id coords (as dimension of the data array)

    # create new 'time' and 'epoch' coords
    time_on_trial = neural_data.where(neural_data.trial_id == neural_data.trial_id.data[0], drop=True).time_in_trial.data

    dt = (time_on_trial[1] - time_on_trial[0]).round(5)  # get the time step
    time_in_past = [time_on_trial[0] - i * dt for i in range(1, np.abs(n_extra_trials[0]) * len(time_on_trial)+1)][::-1]  # get the time of the first time bin
    time_in_future = [time_on_trial[-1] + i * dt for i in range(1, n_extra_trials[1] * len(time_on_trial)+1)]  # get the time of the last time bin
    time_vector = np.concatenate([time_in_past, time_on_trial, time_in_future], axis=0)  # concatenate the time bins
        
    epochs_on_trial = neural_data.where(neural_data.trial_id == neural_data.trial_id.data[0], drop=True).epoch_id.data
    n_trials = np.abs(n_extra_trials[0]) + 1 + n_extra_trials[1]  # number of trials to concatenate
    epochs = np.concatenate([epochs_on_trial for _ in range(n_trials)], axis=0)
    
    # add new coords, named 'time' over new time axis
    coords['time'] = time_vector
    coords['epoch'] = ('time', epochs)

    return xr.DataArray(data_np, coords=coords, dims=dims)  # create dataset


def _concat_to_xr_centered(neural_data, data_np, trials_vector, center_window):
    # get dimensions and coords of the data
    dims = ('trial_id', 'unit', 'time')
    coords = {name: ('unit', coord.values) for name, coord in neural_data['unit'].coords.items()}  # get the coords of each units
    coords['trial_id'] = trials_vector  # add trial_id coords (as dimension of the data array)

    # create new 'time' and 'epoch' coords
    dt = (neural_data.time.data[1] - neural_data.time.data[0]).round(5)  # get the time step
    time = np.arange(center_window[0], center_window[1], step=dt)

    # add new coords, named 'time' over new time axis
    coords['time'] = time

    return xr.DataArray(data_np, coords=coords, dims=dims)  # create dataset


def _get_event_id(neural_data, epoch_start_time):
    """
    Get the time bin index of the event to center the data around.
    """
    event_id = int(np.where(neural_data.time.data == epoch_start_time)[0])
    return event_id


def build_trial_dataset(neural_data, mode='centering', center_on_epoch_start=5, n_extra_trials=(0, 0), center_window=(-1, 1),  monkey=None, session=None):
    """
    Creating a dataset of shape (n_trials, units, time), where 
    - n_trials is the number of trials, matching the number of rows in the session_data dataframe, with matching ids,
    - n_neurons is the number of neurons 
    - time is the number of time bins covering the given interval of trial or epoch.

    The input is the neural data, which is an xarray dataset with dimensions (neurons, time), where
    time covers the whole sessins. There is a 'trial_id' coordinate, which is the same as the 'trial_id' column in the
    session_data dataframe, and an 'epoch' coordinate, which defines the behavioral events.

    The session_data dataframe contains the behavioral data, with a 'trial_id' column, which is the same as the 'trial_id'
    coordinate in the neural data. 

    The target_name is the name of the column in the session_data dataframe, which is the target of the decoding.
    
    Parameters
    ----------
    neural_data : xarray.Dataset
        Neural data.
        
    mode : str, optional (default='centering')
        Mode of operation. Can be 'centering' or 'full_trial'. If 'centering', the data is centered around the event specified in 'center_on'. If 'full_trial', the data is the full trial.
        
    center_on_epoch_start : int, optional (default=5)
        The epoch whos beginning to center the data around. If 'mode' is 'centering', this parameter must be specified. Must be an integer, the epoch id to center the data around its beginning, e.g. 5 centers the data around the feedback event (as epoch_id==5 is the feedback epoch).
    
    center_window : list, optional (default=[-1, 1])
        Window to center the data around. If 'mode' is 'centering', this parameter must be specified. Must be a list of two floats, the start and end of the window in seconds, e.g. [-1, 1] centers the data around the feedback event, from 1 second before to 1 second after.
        
    n_extra_trials : int, optional (default=[0, 0])
        Number of extra trials to include. If 'mode' is 'full_trial', this parameter must be specified. Must be a list of two integers, the number of trials before and after the current trial, e.g. [-1, 1] includes the previous, current and next trials.
        

    Returns
    -------
    X_ds : xarray.Dataset
        The trial dataset.
    """
    # error if not either none or both of the monkey and session are given
    if (monkey is None) != (session is None):
        raise ValueError("Either both monkey and session must be given, or none of them.")

    X_all, trial_ids = [], []  # init lists to store the data and the corresponding trial ids
    all_trial_ids = np.unique(neural_data.trial_id.values)  # get the trial ids, excluding NaNs (before and after session)
    all_trial_ids = all_trial_ids[~np.isnan(all_trial_ids)]  # remove NaNs (before and after session)
    dt = neural_data.attrs['bin_size']  # get the time step
    for trial_id in all_trial_ids:  # loop through all trials in the neural data
        if mode == 'full_trial':
            # get the first and last time bin of the current trial and the next trial
            first_trial_id = int(trial_id + n_extra_trials[0])  # first trial id of the current trial
            last_trial_id = int(trial_id + n_extra_trials[1])  # last trial id of the next trial

            # check if the current trial and the corresponding extra trials are in the neural data:
            if any(trial_to_be_there not in neural_data.trial_id.data for trial_to_be_there in range(first_trial_id, last_trial_id+1)):
                continue

            # Filter the dataset for trial_id between first_trial_id and last_trial_id
            curr_trial_data = neural_data.where((neural_data.trial_id >= first_trial_id) & (neural_data.trial_id <= last_trial_id), drop=True)    

        elif mode == 'centering':
            # Filter the dataset for trial_id between first_trial_id and last_trial_id
            curr_trial_data = neural_data.where(neural_data.trial_id == trial_id, drop=True)
            
            # if there are nans in the epoch vector -> interrupted trial, skip it
            if np.any(np.isnan(curr_trial_data.epoch_id.data)):  
                continue

            # get the time of the event to center the data around
            epoch_start_time = curr_trial_data.where(curr_trial_data.epoch_id == center_on_epoch_start, drop=True).time.data[0]
            event_id = np.where(neural_data.time.data == epoch_start_time)[0][0]  # get the time bin index of the event to center the data around
            n_bins_back = int(center_window[0] / dt)  # number of bins to go back
            n_bins_forward = int(center_window[1] / dt)  # number of bins to go forward
            
            # check indices are not out of bounds
            if epoch_start_time - center_window[0] < neural_data.time.data[0] or epoch_start_time + center_window[1] > neural_data.time.data[-1]:
                continue

            # get the data around the event
            curr_trial_data = neural_data.where(
                (neural_data.time >= neural_data.time[event_id + n_bins_back]) & 
                (neural_data.time < neural_data.time[event_id + n_bins_forward]), 
                drop=True)

        # set the trial_id = trial_id row of the dataframe to the current trial data
        X_all.append(curr_trial_data)
        trial_ids.append(int(trial_id))

    # concatenate the data
    X_ds = {}
    for data_var in neural_data.data_vars:
        X_temp_concat = np.stack([group[data_var].data for group in X_all])
        
        if mode == 'full_trial':
            X_temp_xr = _concat_to_xr_normalized(neural_data, X_temp_concat, n_extra_trials, trial_ids)
            
        elif mode == 'centering':
            X_temp_xr = _concat_to_xr_centered(neural_data, X_temp_concat, trial_ids, center_window)
            epoch_vector = X_all[0].epoch_id.data
            X_temp_xr.coords['epoch'] = ('time', epoch_vector)
            
        X_temp_xr.attrs['bin_size'] = neural_data.attrs['bin_size']  # add bin size to the attributes   
        X_ds[data_var] = X_temp_xr
    X_ds = xr.Dataset(X_ds)  # create dataset
    X_ds.attrs['bin_size'] = neural_data.attrs['bin_size']  # add bin size to the attributes
    
    if monkey is not None and session is not None:
        X_ds = X_ds.assign_coords(unit=('unit', [f'{monkey}_{session}_{unit}' for unit in X_ds.unit.values]))
        X_ds = X_ds.assign_coords(monkey=('unit', [monkey for _ in range(X_ds.dims['unit'])]))
        X_ds = X_ds.assign_coords(session=('unit', [session for _ in range(X_ds.dims['unit'])]))

    return X_ds


def merge_behavior(neural_data_original, behav_original):
    """
    Merges the behavioral data with the trial-by-trial neural data.

    Parameters
    ----------
    neural_data : xarray.Dataset
        Neural data.
        
    behav : pd.DataFrame
        Behavioral data.

    Returns
    -------
    neural_data : xarray.Dataset
        The neural data with the behavioral data added.
    """
    assert len(neural_data_original.dims) == 3, "Neural data must be the trial-by-trial dataset, having 3 dimensions (units, time, trials)."
    
    behav = behav_original.copy()
    neural_data = neural_data_original.copy()
    
    # get trial ids that are in both the neural data and the behavioral data
    shared_trials = np.intersect1d(neural_data.trial_id.data, behav.trial_id.values)
    behav = behav.loc[behav['trial_id'].isin(shared_trials)]
    neural_data = neural_data.sel(trial_id=shared_trials)
    
    # add columns of the behavioral data to the xarray (all but monkey and session)
    colums_to_add = list(behav.columns)
    colums_to_add.remove('session')
    colums_to_add.remove('monkey')
    colums_to_add.remove('trial_id')
    for col in colums_to_add:
        neural_data.coords[col] = ('trial_id', behav[col].values)
            
    return neural_data


def balance_labels(neural_data_original, coords):
    '''
    Balances the neural dataset by undersampling the majority class(es) of the coordinate provided as 'coord'.
      
    Why not simply use sklearn's resample functions? Because it can not hande 3 dimensional X arrays.

    Parameters
    ----------
    neural_data : xarray.Dataset
        Neural data.

    coords : str or list of str
        The coordinate(s) to balance the dataset on. If multiple coordinates are given, the dataset is balanced on the combination of the labels, e.g. if 'coord1' has 2 classes and 'coord2' has 3 classes, the balanced dataset will have 2*3=6 classes.

    Returns
    -------
    neural_data_balanced : xarray.Dataset
        The balanced neural data.
    '''
    # if only one coord is given, convert it to a list
    if isinstance(coords, str):
        coords = [coords]

    # importantly there should not be any nan values in the labels, as we usually do not want to balance considering nan values
    for coord in coords:
        assert not np.isnan(neural_data_original[coord].data).any(), "Label vector contains NaN values - importantly there should not be any nan values in the labels, as we usually do not want to balance considering nan values."
    
    neural_data = neural_data_original.copy()

    if len(coords) > 1:
        # create vector of mixed labels as 'label1:label2:...:labeln'
        coord_vector = []
        n_elements = neural_data[coords[0]].shape[0]
        for i in range(n_elements):
            coord_vector.append(':'.join([str(neural_data[coord].data[i]) for coord in coords]))
        coord_vector = np.array(coord_vector)
        
        y = coord_vector

    else:
        # get vector of the labels
        coord = coords[0]
        y = neural_data[coord].data

    # number of trials per class
    n_trials_per_class = [np.sum(y == i) for i in np.unique(y)]
    min_n_trials_per_class = int(np.min(n_trials_per_class))  # set minimum number of trials per class

    # get indices of trials per class
    subsempled_trial_ids = []
    for i, class_name in enumerate(np.unique(y)):
        trial_ids = neural_data.sel(trial_id=y == class_name).trial_id.data
        subsempled_trial_ids.append(np.random.choice(trial_ids, min_n_trials_per_class, replace=False))

    # concatenate the indices
    subsempled_trial_ids = np.concatenate(subsempled_trial_ids)

    # select the balanced dataset
    neural_data_balanced = neural_data.sel(trial_id=subsempled_trial_ids)

    return neural_data_balanced


def remove_nan_labels(neural_dataset_original, coords):
    """
    Removes trials where the behavioural variable(s) provided as 'coord' is NaN.

    Parameters
    ----------
    neural_dataset : xarray.Dataset
        Neural data.

    coords : str or list of str
        The coordinate(s) to remove the NaN labels from.

    Returns
    -------
    neural_dataset : xarray.Dataset
        The neural data with the NaN labels removed.
    """
    # if only one coord is given, convert it to a list
    if isinstance(coords, str):
        coords = [coords]

    neural_dataset = neural_dataset_original.copy()

    for coord in coords:       
        mask = ~np.isnan(neural_dataset[coord])
        neural_dataset = neural_dataset.where(mask, drop=True)

    return neural_dataset

### time-by-units dataset processing

def downsample_time(data: xr.DataArray, sr_new: float = 100) -> xr.DataArray:
    """
    Downsample xarray data by averaging values within time bins.
    
    Parameters
    ----------
    data : xr.Dataset
        Input data with 'time' dimension in seconds
    sr : float
        Desired sampling rate in Hz
        
    Returns
    -------
    xr.DataArray
        Downsampled data with new time axis
    """    
    # Get original time values
    time = data.time.values
    
    # Create new time bins
    decimals = int(np.log10(sr_new))  # Number of decimals to round to - used to make time rounded

    time_start = np.ceil(time[0] * sr_new) / sr_new  # Start time of first bin is the first time point with enough data around it
    time_end = np.floor(time[-1] * sr_new) / sr_new  # End time of last bin is the last time point with enough data
    time_bins = np.arange(time_start, time_end, 1/sr_new)  # Time bins for new data, e.g. [0.015, 0.025, 0.035, ...], thus with increments of 1/sr_new, shifted by 1/2*1/sr_new so that between them the desired pint is in the middle
    
    # Group by time bins and average
    binned_data = data.sel(time=time_bins, method='nearest')

    # Update time values
    binned_data.attrs['bin_size'] = 1/sr_new
    
    return binned_data


def run_normalization(neural_data_original, print_usr_msg=False):
    """
    Helper function for function 'time_normalize_session()', that normalizes the 
    firing rates of each neuron to the mean firing rate of the neuron across all trials.
    """
    assert isinstance(neural_data_original, xr.Dataset), "neural_data must be an xarray Dataset."
        
    methods = {}
    for data_var in neural_data_original.data_vars:
        # if discrete data, use nearest interpolation, otherwise linear
        data_curr = neural_data_original[data_var].data
        data_curr = data_curr[~np.isnan(data_curr)]
        if np.all(data_curr == np.round(data_curr)):
            methods[data_var] = 'nearest'
        else:
            methods[data_var] = 'linear'
    
    EP_LENS = np.array([1., 1., 0.5, 0.5, 0.5, 4.])
    bin_size = np.round(neural_data_original.time[1].data - neural_data_original.time[0].data, 6)
    ep_lens = (EP_LENS / bin_size.data).astype('int')  # epoch lengths in number of bins
        
    # add attribute: original bin size (bcause it wont be calculatable anymore)
    neural_data_original.attrs['bin_size'] = bin_size    

    neural_data = neural_data_original.copy()
    
    # interpolating each epoch

    trial_ids_all = neural_data.trial_id.values  # all trial ids, including NaNs (before and after session)
    trial_ids_all = np.unique(trial_ids_all[~np.isnan(trial_ids_all)])  # remove NaNs (before and after session)

    epoch_ids_all = neural_data.epoch_id.values  # all epoch ids, including NaNs (before and after session)
    epoch_ids_all = np.unique(epoch_ids_all[~np.isnan(epoch_ids_all)])  # remove NaNs (before and after session)
    
    interrupted_count = 0
    very_long_epoch_count = 0
    all_epochs = np.unique(neural_data.epoch_id.values) # to find shorter epochs 
    all_epochs = all_epochs[~np.isnan(all_epochs)]
    data_normalized = []
    for trial_id in trial_ids_all:  # all trial ids, excluding NaNs (before and after session)
        trial_normalized = []

        # skip interrupted trials and truncated trials (can be at the beginning or at the end of the session je pense)
        epochs_in_trial = np.unique(neural_data.sel(time=neural_data.trial_id==trial_id).epoch_id)
        if np.any(np.isnan(epochs_in_trial)):
            interrupted_count += 1
            continue
        if not np.array_equal(epochs_in_trial, all_epochs):
            interrupted_count += 1
            continue

        for epoch_index in range(len(epoch_ids_all)):
            # get epoch data
            epoch = neural_data.sel(time=(neural_data.trial_id==trial_id) & (neural_data.epoch_id==epoch_index))

            # compute the desired length of the current epoch
            curr_ep_bins_len = ep_lens[epoch_index]
            norm_epoch_time = np.linspace(epoch.time[0], epoch.time[-1], curr_ep_bins_len)  
            
            # time normalization of time
            epoch_norm = xr.Dataset()
            for data_var in epoch.data_vars:
                if not len(epoch.time) > 1:  # if we have at least 2 time points, we can interpolate, but not otherwise TODO: this is ugly code...
                    xxx = xr.concat([epoch[data_var], epoch[data_var]], dim='time')
                    xxx.time.data[1] = xxx.time.data[1]+.0001                
                    norm_epoch_time = np.linspace(epoch.time[0], epoch.time[-1]+.0001, curr_ep_bins_len)  
                    x = xxx.interp(time=norm_epoch_time, method=methods[data_var], assume_sorted=True)
                else:
                    x = epoch[data_var].interp(time=norm_epoch_time, method=methods[data_var], assume_sorted=True)
                epoch_norm[data_var] = x
    
            #epoch_norm = epoch.interp(time=norm_epoch_time, method='linear', assume_sorted=True)
            
            # append to others
            trial_normalized.append(epoch_norm)

            # check if epoch is longer than 10s
            if epoch.time[-1] - epoch.time[0] > 10:  # if epoch is longer than 10s, print warning
                very_long_epoch_count += 1
                
        trial_normalized = xr.concat(trial_normalized, dim='time')
        trial_normalized = trial_normalized.assign_coords({'time_in_trial': ('time', np.linspace(0, EP_LENS.sum()-bin_size, len(trial_normalized.time)))})
        data_normalized.append(trial_normalized)
    
    # concatenate all epochs  -- todo really slow
    data_normalized = xr.concat(data_normalized, dim='time')

    # print info
    if print_usr_msg:
        print(f"Removed {interrupted_count} interrupted trials.")
        print(f"Very long epochs: {very_long_epoch_count} epochs longer than 10s.")
    
    
    data_normalized.attrs['bin_size'] = bin_size  # add bin size to the attributes
    
    return data_normalized


def time_normalize_session(neural_data_original):
    """
    Time normalization that results in uniform trial times. Main function.

    Removes trials with missing epochs (or no epochs in case of interrupted trials).
    """


    neural_data = run_normalization(neural_data_original)

    return neural_data


def scale_neural_data(neural_data_original, method='zscore', data_var='firing_rates'):
    '''
    Z-score scaling of the given data.
    '''
    assert isinstance(neural_data_original, xr.Dataset), "neural_data must be an xarray Dataset."
    
    neural_data = neural_data_original[data_var].copy()
    if method == 'zscore':
        scaler = StandardScaler()
    else:
        raise NotImplementedError(f"Method '{method}' not implemented. It should be 'zscore'.")
    
    # scale data
    neural_data.data = scaler.fit_transform(neural_data.T).T

    neural_data_original[data_var] = neural_data

    return neural_data_original


def remove_low_fr_neurons(neural_data_original, fr_mean=1, print_usr_msg=False):
    """
    Function that removes neurons that has a lower mean firing rate than the given 'limit'.
    
    Mean rates are calculated based on the first data variable in the neural data object.

    :param spike_train: xarray spike train data
    :param limit: the minimum firing rate (in Hz) for a neuron to be kept
    """
    
    # if the data is not in the right format (xarray dataset)
    assert isinstance(neural_data_original, xr.Dataset), "neural_data must be an xarray Dataset."

    neural_data = neural_data_original.copy()

    if 'spike_trains' in neural_data_original.data_vars:
        neural_data = neural_data_original['spike_trains']
        mean_spikes = np.mean(neural_data.data, axis=1)
        bin_size = neural_data_original.attrs['bin_size']
        mean_rates = mean_spikes / bin_size
    elif 'firing_rates' in neural_data_original.data_vars:
        neural_data = neural_data_original['firing_rates']
        mean_rates = np.mean(neural_data.data, axis=1)
    else:
        raise ValueError("Data variable 'spike_trains' or 'firing_rates' not found in the data...")
        
    # remove units with mean firing rate below the given limit
    valid_units = neural_data.unit.data[mean_rates > fr_mean]
    
    if print_usr_msg:
        print(f"Removed {len(neural_data.unit.data) - len(valid_units)}/{len(neural_data.unit.data)} units with mean firing rate lower than {fr_mean} Hz.")
        print(f"removed units: {neural_data.unit.data[mean_rates <= fr_mean]}")
    
    return neural_data_original.sel(unit=valid_units)


def remove_low_varance_neurons(neural_data_original, var_limit=0.1, print_usr_msg=False):
    """
    Function that removes neurons that have a variance lower than the given 'var_limit'.
    """
# if the data is not in the right format (xarray dataset)
    if not isinstance(neural_data_original, xr.Dataset):
        # warning message saying its the old format
        print("Old format detected. Converting to xarray dataset...")
        neural_data = neural_data_original.copy()

    else:
        data_var_name = 'firing_rates'
        if data_var_name not in neural_data_original.data_vars:
            raise ValueError(f"Data variable '{data_var_name}' not found in the data, but must be there to get variance info...")
        neural_data = neural_data_original[data_var_name]
    
    # remove units with variance below the given limit
    variances = np.var(neural_data.data, axis=1)
    valid_units = neural_data.unit.data[variances > var_limit]

    # remove units
    neural_data_original = neural_data_original.sel(unit=valid_units)

    if print_usr_msg:
        print(f"Removed {len(neural_data.unit.data) - len(valid_units)}/{len(neural_data.unit.data)} units with variance lower than {var_limit}.")

    return neural_data_original


def set_off_recording_times_to_nan(neural_data_original):
    # raise deprecated warning
    raise DeprecationWarning("This function is deprecated. Use 'remove_trunctuated_neurons(neural_data, mode='set_nan', delay_limit=5) instead.")


def remove_trunctuated_neurons(neural_data_original, mode='remove', delay_limit=20, print_usr_msg=False):
    """
    Function that handles neurons which arent active all the way through the recording session.
    Criteria: neurons that have a time gap (in s) between the session start/end and the first/last spike that is larger than the given 'delay_limit' are
    either removed completely or set to NaN in the incative time bins.
    
    Must be used on a dataset with 'spike trains'.
    
    Parameters
    ----------
    neural_data : xarray.Dataset
        Neural data with spike trains. Must have 'spike_trains' data var.
        
    mode : str, optional (default='remove')
        Mode of operation. Can be 'remove' or 'set_nan'. If 'remove', neurons with trunctuated recordings are removed completely. If 'set_nan', the inactive time bins are set to NaN (i.e. time bins that are before/after the first/last spike +- delay limit).
    
    delay_limit : float, optional (default=20)
        Time limit in seconds. If the delay between session start/end and the first/last spike is larger than this limit, the neuron is considered trunctuated.
        
    print_usr_msg : bool, optional (default=False)
        If True, prints user messages.
        
    Returns
    -------
    neural_data : xarray.Dataset
        Neural data with neurons with trunctuated recordings removed or set to NaN
    """
    # if the data is not in the right format (xarray dataset)
    assert isinstance(neural_data_original, xr.Dataset), "neural_data must be an xarray Dataset."
    neural_data = neural_data_original.copy()

    neural_data = neural_data_original['spike_trains']
        
    # get times of session start and end
    t_sess_start = neural_data.time[np.where(~np.isnan(neural_data.trial_id))[0][0]].data
    t_sess_end = neural_data.time[np.where(~np.isnan(neural_data.trial_id))[0][-1]].data
    
    # only keep the session times
    neural_data = neural_data.sel(time=slice(t_sess_start, t_sess_end))  # only keep the session times
    
    valid_units = []
    for i, unit_name in enumerate(neural_data.unit.data):
        time_ids_to_set_nan = [] # set of time ids to set to NaN
        
        non_zero_ids = np.where(neural_data.sel(unit=unit_name).data != 0)[0]  # indices of non-zero values

        if len(non_zero_ids) == 0:  # if there are no non-zero values (i.e. no neural activity for this neuron), skip
            continue
        
        first_non_zero_time = neural_data.time.data[non_zero_ids[0]]  # time of first non-zero bin
        last_non_zero_time = neural_data.time.data[non_zero_ids[-1]]  # time of last non-zero bin
        
        if mode == 'set_nan':
            if first_non_zero_time - t_sess_start > delay_limit:  # if there is a gap of more than X sec at the beginning, set to NaN
                time_ids_to_set_nan += list(np.where(neural_data.time.data < first_non_zero_time-delay_limit)[0])
            if t_sess_end - last_non_zero_time > delay_limit:  # if there is a gap of more than X sec at the end, set to NaN
                time_ids_to_set_nan += list(np.where(neural_data.time.data > last_non_zero_time+delay_limit)[0])

            # set to NaN in all data arrays
            if len(time_ids_to_set_nan) > 0:  # if there are times to set to NaN
                for data_var in neural_data_original.data_vars:
                    neural_data_original[data_var][i, time_ids_to_set_nan] = np.nan  # set all times in all deta arrays to NaN
            else:
                valid_units.append(unit_name)

        elif mode == 'remove':
            if np.max([first_non_zero_time - t_sess_start, t_sess_end - last_non_zero_time]) < delay_limit:  # if even the longest gap is less than the delay tolerance, keep the neuron
                valid_units.append(unit_name)
            
    invalid_units = set(neural_data.unit.data) - set(valid_units)
    
    # return
    if mode == 'set_nan':
        if print_usr_msg:
            print(f"Set to NaN {len(invalid_units)}/{len(neural_data.unit.data)} units with trunctuated recordings, delay limit: {delay_limit} s.\n",
                  f"Invalid units: {invalid_units}")

        return neural_data_original
    elif mode == 'remove':
        # remove units with nan values
        neural_data_original = neural_data_original.sel(unit=valid_units)

        n_units_before = len(neural_data.unit.data)
        n_units_after = len(neural_data_original.unit.data)
                
        # print info
        if print_usr_msg:
            print(f"Removed {n_units_before - n_units_after}/{n_units_before} units with trunctuated recordings, delay limit: {delay_limit} s.\n",
                  f"Invalid units: {invalid_units}")
        
        return neural_data_original


def remove_drift_neurons(neural_data_original, corr_limit=.2, print_usr_msg=False):
    """
    Function that removes neurons that have a correlation coefficient between the unit activity and the session time.
    """
    # if the data is not in the right format (xarray dataset)
    if not isinstance(neural_data_original, xr.Dataset):
        # warning message saying its the old format
        print("Old format detected. Converting to xarray dataset...")
        neural_data = neural_data_original.copy()
    else:
        data_var_name = 'firing_rates'
        if data_var_name not in neural_data_original.data_vars:
            raise ValueError(f"Data variable '{data_var_name}' not found in the data, but must be there to get drift info...")
        neural_data = neural_data_original[data_var_name]
        
    valid_units = []
    for i, unit_data in enumerate(neural_data.data):
        corr = np.corrcoef(unit_data, neural_data.time)[0, 1]
        if np.abs(corr) < corr_limit:
            valid_units.append(neural_data.unit.data[i])

    if print_usr_msg:
        print(f"Removed {len(neural_data.unit.data) - len(valid_units)}/{len(neural_data.unit.data)} units with correlation coefficient lower than {corr_limit}.")
    
    return neural_data_original.sel(unit=valid_units)


def get_rate(neural_data, win):
    """
    Calculate firing rate from spike train data.
    Parameters
    ----------
    spikes_xr : xarray.DataArray
        spike train data
    win : np.array
        window for convolution
    Returns
    -------
    rates : xarray.DataArray
        firing rate data
    """
    rates_np = convolve1d(neural_data.spike_trains.data, win, axis=1, output=np.float32) #/ bin_size

    return rates_np


### CV analysis


def calculate_CV(isi_vector):
    CV = np.std(isi_vector) / np.mean(isi_vector)  # ???????
    return CV 

def calculate_lvr(isi_vector):
    """
    Calculate the Local Variation (LVR) of a sequence of inter-spike intervals (ISIs).

    Parameters:
    isi_vector (list or numpy array): Vector of inter-spike intervals.

    Returns:
    float: The calculated LVR value.
    """
    refractory_period = 0.005  # seconds, assumed constant refractory period
    lvr_sum = 0.0  # Initialize sum for LVR calculation

    # Iterate over ISI vector to calculate sum part of LVR
    for i in range(len(isi_vector) - 1):
        isi_sum = isi_vector[i] + isi_vector[i + 1]
        lvr_component = (1 - 4 * isi_vector[i] * isi_vector[i + 1] / isi_sum**2) * \
                        (1 + 4 * refractory_period / isi_sum)
        lvr_sum += lvr_component

    # Calculate final LVR value
    lvr_value = 3 * lvr_sum / (len(isi_vector) - 1)
    return lvr_value
    
def _run_cv(isi_vector, mode):
    if len(isi_vector) < 100:
        return np.nan
    else:
        if mode == 'CV':
            return calculate_CV(isi_vector)
        elif mode == 'LVR':
            return calculate_lvr(isi_vector)
        elif mode == 'mean_rate':
            return 1/np.mean(isi_vector)
    
def _get_epoch_isi(isi_vector, epoch_vector, epoch_ids):    
    # select only the ISIs from the epochs of interest
    isi_vector_curr_epochs = isi_vector[np.isin(epoch_vector, epoch_ids)]
    return isi_vector_curr_epochs
    
'''def _run_cv(neural_data, unit, epoch_ids):
neural_data_unit_temp = neural_data.sel(
        unit=unit,
        time=np.isin(neural_data.epoch_id.values, epoch_ids)  # full trial, but no nan-s
        )
spikes_times = neural_data_unit_temp.time[neural_data_unit_temp.spike_trains.data == 1]
isi_vector = np.diff(spikes_times)
if len(isi_vector) <50:
    return np.nan
else:
    return calculate_CV(isi_vector)'''

def add_CV(neural_data, mode='CV', epochs='all'):
    """
    Adds CV values to the neural data. Adds a new coordinate to the neural data object for each CV value.

    No CV values are computed for units with less than 100 spikes (add NaN instead computing the CV).
    
    If epochs is 'all', computes CV values for each unit in the neural data object and for different epochs:
    - CV_full: CV value for the full trial.
    - CV_st -> Lv: CV value for the start to lever epoch.
    - CV_Lv -> Fb: CV value for the lever to feedback epoch.
    - CV_Fb -> end: CV value for the feedback to end epoch.
    if epochs is 'full_only', computes CV values for the full trial only.
    
    Uses the LVR method to calculate the CV values (from a paper by someone Matteo knowks).

    Parameters:
    neural_data (object): The neural data object.
    mode (str): The CV mode to use. Can be 'CV' or 'LVR'.
    epochs (str): The epochs to compute CV values for. Can be 'all' or 'full_only'.

    Returns:
    object: The updated neural data object with CV values added.
    """
    
    if epochs == 'all':
        cv_s = {f'{mode}_full': [], f'{mode}_st -> Lv': [], f'{mode}_Lv -> Fb': [], f'{mode}_Fb -> end': []}
    elif epochs == 'full_only':
        cv_s = {f'{mode}_full': []}
    else:
        raise ValueError(f"Invalid 'epochs' parameter: {epochs}. It should be 'all' or 'full_only'.")
    
    for unit in neural_data.unit.data:
        neural_data_unit_temp = neural_data.sel(unit=unit,).spike_trains
        spikes = neural_data_unit_temp[neural_data_unit_temp.data == 1]  # get the times of all spikes
        
        # add the ISI vector to the spikes
        isi_vector = np.diff(spikes.time.data)  # get the ISI vector
        epoch_vector = spikes.epoch_id.data[:-1]  # get the epoch vector
        
        # remove ISI out of mean+-5std
        if mode == 'CV':
            ids_to_remove = np.where((isi_vector > np.mean(isi_vector) + 5*np.std(isi_vector) ) | (isi_vector < np.mean(isi_vector) - 5*np.std(isi_vector)))[0]
            isi_vector = np.delete(isi_vector, ids_to_remove)
            epoch_vector = np.delete(epoch_vector, ids_to_remove)
        
        if epochs == 'all':
            cv_s[f'{mode}_full'].append(_run_cv(_get_epoch_isi(isi_vector, epoch_vector,[0, 1, 2, 3, 4, 5]), mode))
            cv_s[f'{mode}_st -> Lv'].append(_run_cv(_get_epoch_isi(isi_vector, epoch_vector, [0, 1]), mode))
            cv_s[f'{mode}_Lv -> Fb'].append(_run_cv(_get_epoch_isi(isi_vector, epoch_vector, [2, 3, 4]), mode))
            cv_s[f'{mode}_Fb -> end'].append(_run_cv(_get_epoch_isi(isi_vector, epoch_vector, [5]), mode))
        elif epochs == 'full_only':
            cv_s[f'{mode}_full'].append(_run_cv(_get_epoch_isi(isi_vector, epoch_vector,[0, 1, 2, 3, 4, 5]), mode))
            
    # assign as coordinate over the units
    for cv_type, cv_value in cv_s.items():
        neural_data.coords[cv_type] = ('unit', cv_value)
        
    return neural_data


### from down here: should be rewieved, cleaned and deleted...

def join_session_info(df1, df2):
    """
    Merging session data info, where the key is the "unit_id" field.
    """

    return df1.join(df2.set_index('trial_id'), on='trial_id')  # merge spikes


def filter_behaviour(behav, to_keep='events'):
    raise NotImplementedError
    """
    Filters behaviour data, i.e. returns either the behavioural times ot the events
    :param behav:
    :param to_keep: str
        "times" or "events"
    :return:
    """
    if to_keep == "times":
        return join_session_info(behav.trial_id, behav.filter(regex='_time$', axis=1))
    elif to_keep == "events":
        return #????


def get_average_epoch_len():
    pass


def get_trial_data(session_data, signal_name, trial_id):
    """
    Returns a single trial or a list of trials, based on trial id. 'trial_id' can be integer or list.
    :param session_data:
    :param signal_name:
    :param trial_id:
    :return:
    """
    if isinstance(trial_id, numbers.Integral):  # check if number is integer (int or np.int8 | np.int32 | ...)
        return session_data[session_data['trial_id'] == trial_id][signal_name].values[0]
    if isinstance(trial_id, list) or isinstance(trial_id, np.ndarray):
        return [get_trial_data(session_data, signal_name, curr_trial_id) for curr_trial_id in trial_id]
    else:
        raise ValueError(f"'trial_id' invalid format... {trial_id}")


def concat_trials(trials, print_usr_msg=False):
    """
    Concatenates trials in time. Checks if trials are consecutive or not. Checks if bin sizes are matching
    :param trials:
    :return:
    """
    if not isinstance(trials, list):
        return trials

    assert len(trials) != 0, 'nothing to concatenate'

    bin_size = trials[0].time[1] - trials[0].time[0]

    #assert np.count_nonzero(np.array([math.isclose((trial.time[1] - trial.time[0]).values, bin_size) for trial in trials])) != 1, f"bin size mismatch"

    consecutives = True
    for i in range(len(trials) - 1):
        if not (math.isclose(trials[i].time_in_session[-1] + bin_size, trials[i + 1].time_in_session[0]) or
                math.isclose(trials[i].time_in_session[-1], trials[i + 1].time_in_session[0]) or
                math.isclose(trials[i].time_in_session[-1] - bin_size, trials[i + 1].time_in_session[0])):
            consecutives = False
            break
    if not consecutives:
        if print_usr_msg:
            print("trials are not consecutive!")

    return xr.concat(trials, dim='time')


def get_area(signal_name):
    return signal_name.split('_')[0]


def get_average_epoch_length(session_data, default_lens=True, return_ep_names=False):
    if default_lens:
        ep_lens = np.array([1., 1., 0.5, 0.5, 0.5, 4.])

        if return_ep_names:
            ep_names = [name[:-len('_time')] for name in session_data.filter(regex='_time$', axis=1).columns]
            return {name: len for len, name in zip(ep_lens, ep_names)}
        else:
            return ep_lens
    else:
        # calculate epoch lengths
        behav = session_data.filter(regex='_time$', axis=1)  # get only time things, i.e. behaviour times

        ep_lens = [(behav.iloc[:, i + 1] - behav.iloc[:, i]).mean() for i in range(len(behav.columns) - 1)]
        return np.round(ep_lens, 1)   # normalised epoch lengths in time


# todo this has to be in the dataframe_tools.py file!!!! or deleted

def merge_areas(session_data, mode='rates'):
    merged_data = []
    for index, row in session_data.iterrows():
        merged_trials = xr.concat([row[f'LPFC_{mode}'], row[f"MCC_{mode}"]], dim='unit')
        merged_data.append(merged_trials)
    session_data[f'both_{mode}'] = merged_data
    return session_data


def join_bins(session_data, signal_name, n_bins):
    new_col = []
    for trial_data in session_data[signal_name]:
        N, T = trial_data.shape
        T = (T // n_bins) * n_bins # throw away last bins

        arr = trial_data[:, :T]
        arr = np.stack([np.mean(arr[:, i*n_bins:(i+1)*n_bins], axis=1) for i in range(T // n_bins)])
        arr = arr.T

        time = trial_data.time[:T]
        time = np.stack([time[i*n_bins] for i in range(T // n_bins)])
        time_in_session = trial_data.time_in_session[:T]
        time_in_session = np.stack([time_in_session[i*n_bins] for i in range(T // n_bins)])

        xr_new = xr.DataArray(arr,
                              dims=['unit', 'time'],
                              coords={
                                  'unit': trial_data.unit.data,
                                  'unit_label': ('unit', trial_data.unit_label.data),
                                  'channel': ('unit', trial_data.channel.data),
                                  'time': time,
                                  'time_in_session': ('time', time_in_session),
                                  'bin_size': trial_data.bin_size.data*n_bins
                              })

        new_col.append(xr_new)

    session_data[signal_name] = new_col
    return session_data


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

def run_normalization(neural_data_original, print_usr_msg=False):
    """
    Helper function for function 'time_normalize_session()', that normalizes the 
    firing rates of each neuron to the mean firing rate of the neuron across all trials.
    """
    if isinstance(neural_data_original, xr.DataArray):  # if it is a single data array, return a single data array
        import warnings
        warnings.warn("Single DataArray is returned, consider using a Dataset instead.")
        
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

    if print_usr_msg:
        print("Normalizing firing rates (in time)...")
        pbar = tqdm(total=len(trial_ids_all))
    
    interrupted_count = 0
    very_long_epoch_count = 0
    data_normalized = []
    for trial_id in trial_ids_all:  # all trial ids, excluding NaNs (before and after session)
        trial_normalized = []
        if print_usr_msg: pbar.update(1)
        if len(np.unique(neural_data.sel(time=neural_data.trial_id==trial_id).epoch_id)) == 1:  # if there is no epoch in this trial (only nan values), skip (interrupoted trials...)
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
    if print_usr_msg: pbar.close()
    
    # concatenate all epochs  -- todo really slow
    data_normalized = xr.concat(data_normalized, dim='time')

    # print info
    if print_usr_msg:
        print(f"Removed {interrupted_count} interrupted trials.")
        print(f"Very long epochs: {very_long_epoch_count} epochs longer than 10s.")
    
    return data_normalized


def time_normalize_session(neural_data):
    """
    Time normalization that results in uniform trial times. Main function.
    """


    neural_data = run_normalization(neural_data)

    return neural_data


def scale_neural_data(neural_data_original, method='zscore'):
    '''
    Z-score scaling of the given data.
    '''

    # if the data is not in the right format (xarray dataset)
    if not isinstance(neural_data_original, xr.Dataset):
        # warning message saying its the old format
        print("Old format detected. Converting to xarray dataset...")
        neural_data = neural_data_original.copy()

    neural_data = neural_data_original.copy()
    if method == 'zscore':
        scaler = StandardScaler()
    else:
        raise NotImplementedError 
    
    # scale data
    for data_var in neural_data.data_vars:
        #neural_data.data = scaler.fit_transform(neural_data.T).T
        neural_data[data_var].data = scaler.fit_transform(neural_data[data_var].data.T).T

    return neural_data


def remove_low_fr_neurons(neural_data_original, fr_mean=1, print_usr_msg=False):
    """
    Function that removes neurons that has a lower mean firing rate than the given 'limit'.
    Removal is done trial-by-trial! This function works on the spike trains only (use a different function for
    firing rates).

    :param spike_train: xarray spike train data
    :param limit: the minimum firing rate (in Hz) for a neuron to be kept
    """
    
    # if the data is not in the right format (xarray dataset)
    if not isinstance(neural_data_original, xr.Dataset):
        # warning message saying its the old format
        print("Old format detected. Converting to xarray dataset...")
        neural_data = neural_data_original.copy()

    else:
        first_data_var_name = list(neural_data_original.data_vars.keys())[0]
        neural_data = neural_data_original[first_data_var_name]
    
    # remove units with mean firing rate below the given limit
    mean_rates = np.sum(neural_data.data, axis=1) / (neural_data.time[-1] - neural_data.time[0]).data
    valid_units = neural_data.unit.data[mean_rates > fr_mean]
    
    if print_usr_msg:
        print(f"Removed {len(neural_data.unit.data) - len(valid_units)}/{len(neural_data.unit.data)} units with mean firing rate lower than {fr_mean} Hz.")
    
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
    """
    # if the data is not in the right format (xarray dataset)
    if not isinstance(neural_data_original, xr.Dataset):
        # warning message saying its the old format
        print("Old format detected. Converting to xarray dataset...")
        neural_data = neural_data_original.copy()

    else:
        first_data_var_name = list(neural_data_original.data_vars.keys())[0]
        neural_data = neural_data_original[first_data_var_name]
        
    # get times of session start and end
    t_sess_start = neural_data.time[np.where(~np.isnan(neural_data.trial_id))[0][0]].data
    t_sess_end = neural_data.time[np.where(~np.isnan(neural_data.trial_id))[0][-1]].data
    
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
                time_ids_to_set_nan += list(np.where(neural_data.time.data < first_non_zero_time)[0])
            if t_sess_end - last_non_zero_time > delay_limit:  # if there is a gap of more than X sec at the end, set to NaN
                time_ids_to_set_nan += list(np.where(neural_data.time.data > last_non_zero_time)[0])

            # set to NaN in all data arrays
            if len(time_ids_to_set_nan) > 0:  # if there are times to set to NaN
                for data_var in neural_data_original.data_vars:
                    neural_data_original[data_var][i, time_ids_to_set_nan] = np.nan  # set all times in all deta arrays to NaN
                    
        elif mode == 'remove':
            if np.max([first_non_zero_time - t_sess_start, t_sess_end - last_non_zero_time]) < delay_limit:  # if the delay is less than N sec, keep the unit, otherwise remove (i.e. skip)
                valid_units.append(unit_name)
            
    if mode == 'set_nan':
        return neural_data_original
    elif mode == 'remove':
        # remove units with nan values
        n_units_before = len(neural_data.unit.data)
        neural_data_original = neural_data_original.sel(unit=valid_units)
                
        # print info
        if print_usr_msg:
            print(f"Removed {n_units_before - len(neural_data_original.unit.data)}/{n_units_before} units with trunctuated recordings, delay limit: {delay_limit} s.")
        
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


### CV calculation
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

###

def add_firing_rates(neural_data, method='gauss', std=.050, win_len=.200, drop_spike_trains=False):
    """
    Add firing rate fields calculated from spikes fields
    Parameters
    ----------
    trial_data : pd.DataFrame
        trial_data dataframe
    method : str
        'moving_avg' or 'gauss'
    std : float (optional)
        standard deviation of the Gaussian window to smooth with
        default 0.05 seconds

    Returns
    -------
    td : pd.DataFrame
        trial_data with '_rates' fields added
    """   
    if isinstance(neural_data, xr.DataArray):  # if it is a single data array, return a single data array
        import warnings
        warnings.warn("Single DataArray is returned, consider using a Dataset instead.")

    bin_size = neural_data.attrs['bin_size'].values
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
    
    neural_data[new_data_name] = firing_rates_xr
    return neural_data.drop('spike_trains') if drop_spike_trains else neural_data

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


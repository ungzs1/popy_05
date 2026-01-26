# imports
import sys
sys.path.append("C:\ZSOMBI\OneDrive\PoPy")
sys.path.append("/Users/zsombi/OneDrive/PoPy")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from popy.io_tools import *
from popy.io_tools import load_metadata
from popy.behavior_data_tools import *
from popy.neural_data_tools import *
from popy.decoding.population_decoders import build_dataset
from popy.plotting.plotting_tools import *


### plotting

def mask_N_consecutives(data, N=4):
    # find where N consecutive values are False
    mask = np.empty(data.shape)
    for u, unit_data in enumerate(data):
        consecutive_count = 0
        candidate_ids = []
        unit_signif_ids = []
        for i, num in enumerate(unit_data):
            # if not nan
            if num == num:
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
    return np.ma.masked_where(mask==0, data)


### data processing

def load_preproc_session_data(monkey, session, area='both'):
    session_data = load_behavior(monkey, session)
    session_data = add_value_function(session_data, monkey=monkey, digitize=False)
    session_data['value_function_continous'] = session_data['value_function']
    '''session_data = add_value_anna(session_data)
    session_data['value_function_continous'] = session_data['fss_01']'''
    #session_data = add_value_function(session_data, monkey=monkey, digitize=True, n_classes=4)
    session_data = add_phase_info(session_data)
    # remove phase == 'search'

    if np.all(np.isnan(session_data.value_function_continous)):
        raise ValueError('No value function found for current session(s).')

    session_data = session_data.dropna()

    # remove phase not 'repeat'
    #session_data = session_data[session_data.phase == 'repeat']

    # Load neural data
    neural_data = load_neural_data(monkey, session, 'rates')

    # get area
    if area == 'both':
        neural_data = neural_data
    else:
        area_ids = np.where(neural_data.area.data == area)[0]
        neural_data = neural_data[area_ids, :]

    # conservative restrictions
    neural_data = remove_trunctuated_neurons(neural_data, delay_limit=10)  # remove neurons with trunctuated activity
    neural_data = remove_drift_neurons(neural_data, corr_limit=.2)  # remove neurons with drift
    neural_data = remove_low_fr_neurons(neural_data, 1)  # remove low_firing units

    # normalize neural data in time
    neural_data = time_normalize_session(neural_data)  # normalize neural data in time to get uniform trial length
    #neural_data = scale_neural_data(neural_data)  # scale neural data (z-score)

    return session_data, neural_data


def data_to_xr(results, significances, unit_names, time_vector, monkey, area, session):
    """
    Initialize the container for the data.

    data must be a numpy array of units x timepoints.
    """

    # set data containers, empty xarray with first dimension named unit_id, second dimension is time
    dr1 = xr.DataArray(
        results,
        coords={
            "unit_id": unit_names,
            "time": time_vector,
            "monkey": ("unit_id", monkey),
            "area": ("unit_id", area),
            "session": ("unit_id", session),
        },
        dims=["unit_id", "time"],
    )

    dr2 = xr.DataArray(
        significances,
        coords={
            "unit_id": unit_names,
            "time": time_vector,
            "monkey": ("unit_id", monkey),
            "area": ("unit_id", area),
            "session": ("unit_id", session),
        },
        dims=["unit_id", "time"],
    )

    # create dataset
    ds = xr.Dataset({"results": dr1, "significances": dr2})

    return ds

def statistics(a, b):
    return np.corrcoef(a, b)[0, 1]

from scipy.stats import permutation_test

def measure_modulation(a, b):
    res = permutation_test((a, b), statistics, n_resamples=1000)

    return res.statistic, res.pvalue


def process_session(monkey, session):
    # load behav data and neural data
    session_data, neural_data = load_preproc_session_data(monkey, session, area='both')

    # create dataset
    neural_data_np, session_data = build_dataset(neural_data, session_data, n_extra_trials=0)

    step_size = 10

    res_sess = np.empty((3, neural_data_np.shape[1], neural_data_np.shape[2]//step_size))
    sig_sess = np.empty((3, neural_data_np.shape[1], neural_data_np.shape[2]//step_size))
    # loop through nuerons
    for neuron in range(neural_data_np.shape[1]):
        neural_data_curr = neural_data_np[:, neuron, :]

        # loop through targets
        for i_target, target in enumerate([1, 2, 3]):            
            target_ids = session_data.target.values == target

            # get data
            X, y = neural_data_curr[target_ids], session_data.value_function_continous.values[target_ids]

            # loop throuch time
            res_temp, sig_temp = [], []
            for time in range(0, X.shape[1], step_size):
                # measure modulation
                '''
                More clean to write like this and erase data manipulation from above in the for loop:
                X = neural_data_np[target_ids, neuron, time]
                y = session_data.value_function_continous.values[target_ids]
                modulation, significance = measure_modulation(X, y)
                '''
                modulation, significance = measure_modulation(X[:, time], y[:])
                res_temp.append(modulation)
                sig_temp.append(significance)
            res_temp = np.array(res_temp)
            sig_temp = np.array(sig_temp)

            res_sess[i_target, neuron, :] = res_temp
            sig_sess[i_target, neuron, :] = sig_temp

    # to xarray
    unit_names, time_vector = neural_data.unit.data, np.linspace(0, neural_data_np.shape[2]*neural_data.attrs['bin_size'], res_sess.shape[2])
    monkey, area, session = [monkey for _ in range(len(unit_names))], neural_data.area.data, [session for _ in range(len(unit_names))]
    xr_t1 = data_to_xr(res_sess[0], sig_sess[0], unit_names, time_vector, monkey, area, session)
    xr_t2 = data_to_xr(res_sess[1], sig_sess[1], unit_names, time_vector, monkey, area, session)
    xr_t3 = data_to_xr(res_sess[2], sig_sess[2], unit_names, time_vector, monkey, area, session)

    return xr_t1, xr_t2, xr_t3


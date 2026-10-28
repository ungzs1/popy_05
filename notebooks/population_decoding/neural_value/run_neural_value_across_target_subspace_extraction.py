##% title Imports
import os

from matplotlib.pyplot import clf

# Limit native-threaded math libs per process (prevents CPU oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import traceback
import concurrent.futures
import datetime
import logging

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.linear_model import LinearRegression, LogisticRegression

import popy.config as cfg
from notebooks.population_decoding.neural_value.helpers import (
    time_resolved_decoder_all_time_target_stratified,
)
from popy.io_tools import load_metadata
from run_neural_value_all_sessions import load_data_custom


PATH = cfg.PROJECT_PATH_LOCAL

def extract_target_subspaces(neural_dataset, fit_subspace_on='R_1'):
    '''
    This is the modified version of the code, where we regress the neural data to the last outcome instead of the model based value. This way we get a more unbiased projection of the neural data, which is not biased towards foraging (as the model based value is).
    '''
    # Get a vector of selected targets [0, 1, 2] for each trial
    target_vector = neural_dataset['target'].values
    trial_id_vector = neural_dataset['trial_id'].values
    target_count = {target: np.sum(target_vector == target) for target in np.unique(target_vector)}
    min_target_count = min(target_count.values())

    # split the neural_dataset into 2 subsets: sample min_target_count/2 trials from each target for training, and the same amount for testing
    trials_in_first_half = []
    trials_in_second_half = []
    for target in np.unique(target_vector):
        trials_for_target = trial_id_vector[np.where(target_vector == target)[0]]
        np.random.shuffle(trials_for_target)
        trials_in_first_half.extend(trials_for_target[:min_target_count//2])
        trials_in_second_half.extend(trials_for_target[min_target_count//2:min_target_count])

    neural_dataset_first = neural_dataset.sel(trial_id=trials_in_first_half)
    neural_dataset_second = neural_dataset.sel(trial_id=trials_in_second_half)

    # write back to xarray, preserve the trial and time dimensions and corresponding coordinates
    time_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'time' in coord.dims}
    trial_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'trial_id' in coord.dims}
    unit_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'unit' in coord.dims}

    # create train and test splits
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)

    # store the weights for each target and time point
    weights_all = np.full((3, 2, len(neural_dataset.unit), len(neural_dataset.time)), np.nan) 

    ## Part 2: project alternative targets: train the model on one target, project the other targets to this subspace
    # scores = np.empty((2, len(neural_dataset.time)))
    for t_id, t in enumerate(neural_dataset.time):
        for fold, target_temp in enumerate(np.unique(target_vector)):
            neural_dataset_first_within = neural_dataset_first.firing_rates.sel(trial_id=neural_dataset_first.target == target_temp, time=t)
            neural_dataset_second_within = neural_dataset_second.firing_rates.sel(trial_id=neural_dataset_second.target == target_temp, time=t)
            #neural_dataset_across = neural_dataset.firing_rates.sel(trial_id=neural_dataset.target != target_temp, time=t)

            # get the firing rates for the current time point
            X_train_first = neural_dataset_first_within.values
            y_train_first = neural_dataset_first_within[fit_subspace_on].values

            X_train_second = neural_dataset_second_within.values
            y_train_second = neural_dataset_second_within[fit_subspace_on].values

            # fit the model
            clf.fit(X_train_first, y_train_first)
            weights_first = clf.coef_.reshape(-1, 1)  # reshape to ensure correct dimensions

            clf.fit(X_train_second, y_train_second)
            weights_second = clf.coef_.reshape(-1, 1)  # reshape to ensure correct dimensions

            # trials_projected = np.dot(neural_dataset_across, weights)
            #weights_norm = np.linalg.norm(weights)
            #trials_projected = (np.dot(neural_dataset_across, weights) + clf.intercept_) / weights_norm

            weights_all[fold, 0, :, t_id] = weights_first.squeeze()  # store the weights for each target and time point
            weights_all[fold, 1, :, t_id] = weights_second.squeeze()  # store the weights for each target and time point

    # Create a DataArray with the projected data
    data_projected_da = xr.DataArray(weights_all, dims=('subspace', 'split', 'unit', 'time'),
                                      coords={'subspace': [f'target_{int(temp)}' for temp in np.unique(target_vector)],
                                              'unit': neural_dataset.unit.values,
                                              'time': neural_dataset.time.values,
                                              'split': ['first', 'second']})
    
    # Add the original coordinates along time and neuron axes
    for name, coord in time_coords.items():
        data_projected_da = data_projected_da.assign_coords({name: coord})
    for name, coord in unit_coords.items():
        data_projected_da = data_projected_da.assign_coords({name: coord})

    return data_projected_da


### Load metadata

def get_all_sessions():
    session_metadata = load_metadata()
    session_metadata = session_metadata[session_metadata['block_len_valid'] == True]  # Only use sessions with valid block length

    monkeys = session_metadata['monkey'].values 
    sessions = session_metadata['session'].values

    return monkeys, sessions


### Configure logging

def end_log():
    # start time is the first log entry
    end_time = datetime.datetime.now()
    logging.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

def init_io(PARAMS):
    os.makedirs(PARAMS['floc'], exist_ok=True)

    # configure logging
    logging.basicConfig(filename=os.path.join(PARAMS['floc'], 'log.txt'),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w')  # 'w' mode will overwrite the log file

    start_time = datetime.datetime.now()
    logging.info("PARAMS:")
    for key, value in PARAMS.items():
        logging.info(f'{key}: {value}')
    logging.info(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

### Save results

def save_results(dfs_all, floc):
    # concatenate all subregions into a single xarray dataset (along the 'unit' dimension)
    dfs_xr = xr.concat(dfs_all, dim='unit')

    # save xarray dataset to a netcdf file
    dfs_xr.to_netcdf(os.path.join(floc, 'weights.nc'))

### Set parameters
 
TIMES_OF_INTEREST = {  # TODO normally we only want the last one
    #'after_fb':   (-4, -2),
    #'it1':        (-2,  0),
    #'it2':        (-.5,  1.5),
    #'tt_centered': (2,  3),    
    #'tt_centered_short': (2.25,  2.75),    
    'before_fb':  (1.5, 3.5), 
}

FLOC_FOLDER = 'across_target_weights_split_'  # TODO
N_FIRST_ONLY = None  # TODO: whether to use only the first trial of each block (to avoid contamination from previous trial). None or an int.

def make_params(key, time_window):
    FLOC_BASE = os.path.join(cfg.PROJECT_PATH_LOCAL, 'notebooks', 'population_decoding', 'neural_value', 'results')
    return {
        'time_of_interest': list(time_window),
        'floc': os.path.join(FLOC_BASE, FLOC_FOLDER+key),
        'n_first_only': N_FIRST_ONLY,  # whether to use only the first trial of each block (to avoid contamination from previous trial) 
    }

### Run

def run(monkey, session, PARAMS):
    print('running: ', monkey, session)
    #monkey, session, subregion = 'ka', '210322', 'vLPFC'

    time_of_interest = PARAMS['time_of_interest']

    neural_dataset = load_data_custom(monkey, session, area=None, n_extra_trials=(-1, 0), n_first_only=PARAMS['n_first_only'])  # load data for the current session, with all areas and one extra trial before the first trial (for projection purposes)

    dfs = []
    for subregion in np.unique(neural_dataset.subregion.data):
        # get data for current subregion and time window of interest
        neural_dataset_temp = neural_dataset.sel(unit=neural_dataset.subregion==subregion)
        neural_dataset_temp = neural_dataset_temp.sel(time=slice(*time_of_interest))

        # project the neural data across targets - from high dimensional space to 3 dimensional space
        weights = extract_target_subspaces(neural_dataset_temp, fit_subspace_on='R_1')  
                
        dfs.append(weights)

    session_log = []

    return dfs, session_log

if __name__ == '__main__':
    # run the same thing for different time windows of interest
    for key, time_window in TIMES_OF_INTEREST.items():
        PARAMS = make_params(key, time_window)  # generate parameters for the current time window
        init_io(PARAMS)  # Initialize logging and create results folder

        monkeys, sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information
        
        n_cores = np.max([1, os.cpu_count()-2])  # get number of cores in the machine TODO set this manually if you want to use less cores
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            # submit jobs
            futures, future_proxy_mapping = [], {}
            for monkey, session in zip(monkeys, sessions):
                future = executor.submit(run, monkey, session, PARAMS)  # Run decoder for each session
                futures.append(future)
                future_proxy_mapping[future] = (monkey, session)

            # wait for results, save them
            count_good = 0
            count_bad = 0

            dfs_all = []  
            for future in concurrent.futures.as_completed(futures):
                try:
                    res, session_log = future.result()
                    monkey_fut, session_fut = future_proxy_mapping[future]

                    # Append results to existing results and save after each session
                    if len(dfs_all) == 0:
                        dfs_all = res
                    else:
                        dfs_all += res

                    # Save results after each session
                    save_results(dfs_all, PARAMS['floc'])  # Save results after each session

                    # Log progress
                    for line in session_log:
                        logging.info(line)
                    logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                    count_good += 1

                except Exception as e:  # Catch exceptions and log them
                    logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                    print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                    traceback.print_exc()  # Print traceback (?)

                    count_bad += 1

                print(f'Progress: {count_good + count_bad}/{len(monkeys)} failed: {count_bad}')


        end_log()
        print(f'Finished all on {datetime.datetime.now()}')

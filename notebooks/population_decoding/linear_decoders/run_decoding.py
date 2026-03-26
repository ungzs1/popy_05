import os

# Limit native-threaded math libs per process (prevents CPU oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import datetime
import concurrent.futures
import traceback
import xarray as xr
import logging

from popy.io_tools import load_metadata
from popy.decoding.population_decoders import run_decoder
import popy.config as cfg


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

def save_results(xr, floc):
    xr.to_netcdf(os.path.join(floc, 'scores.nc'))

### Set parameters
'''
"stay_value_0.05",
        "stay_value_0.10",
        "stay_value_0.15",
        "stay_value_0.20",
        "stay_value_0.25",
        "stay_value_0.30",
        "stay_value_0.35",
        "stay_value_0.40",
        "stay_value_0.45",
        "stay_value_0.50",
        "stay_value_0.55",
        "stay_value_0.60",
        "stay_value_0.65",
        "stay_value_0.70",
        "stay_value_0.75",
        "stay_value_0.80",
        "stay_value_0.85",
        "stay_value_0.90",
        "stay_value_0.95",'''

PARAMS = {
    "conditions": [  # TODO
        'stay_value', 
        'Q_1_inf', 'Q_1_stand', 'Q_2_inf', 'Q_2_stand', 'Q_3_inf', 'Q_3_stand', 
        'Q_chosen_inf', 'Q_chosen_stand'
    ],
    "group_targets": None,  # ['target', 'target_shuffled'],
    "K_fold": 10,
    "step_len": 0.1,
    "n_perm": 100,
    "n_extra_trials": (-1, 0),
    "floc": os.path.join(
        cfg.PROJECT_PATH_LOCAL,
        "notebooks",
        "population_decoding",
        "results",
        "alternative_model_values",
    ),
    "msg": "Running linear decoders for alternative model values (e.g. Q values, RPE) - no grouping of targets, 100 permutations, 10-fold CV, step length of 100ms, and using all trials (n_extra_trials = (-1, 0))",
}

### Run

if __name__ == '__main__':
    init_io(PARAMS)  # Initialize logging and create results folder

    monkeys, sessions = get_all_sessions()  # Get a pandas df containing all sessions' meta information
    
    n_cores = np.min([111, os.cpu_count()-3])  # get number of cores in the machine
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # submit jobs
        futures, future_proxy_mapping = [], {}
        for monkey, session in zip(monkeys, sessions):
            future = executor.submit(run_decoder, monkey, session, PARAMS, load_data=False, save_data=False)  # Run decoder for each session
            futures.append(future)
            future_proxy_mapping[future] = (monkey, session)

        # wait for results, save them
        count = 0
        xrs = []  
        for future in concurrent.futures.as_completed(futures):
            try:
                res, session_log = future.result()
                monkey_fut, session_fut = future_proxy_mapping[future]

                # Append results to existing results and save after each session
                if len(xrs) == 0:  # First result - save directly
                    xrs = res
                else:  # Not first result - concatenate to existing results
                    xrs = xr.concat([xrs, res], dim='session')
            
                # Save results after each session
                save_results(xrs, PARAMS['floc'])  # Save results after each session

                # Log progress
                for line in session_log:
                    logging.info(line)
                logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")

                # print log
                print(f'Progress: {count+1}/{len(monkeys)}')
                count += 1

            except Exception as e:  # Catch exceptions and log them
                logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                traceback.print_exc()  # Print traceback (?)

    end_log()
    print(f'Finished all on {datetime.datetime.now()}')

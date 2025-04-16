import os

import pandas as pd
import xarray as xr
import numpy as np
import concurrent.futures
# mute warnings
import datetime
from itertools import product
import traceback
import logging

from popy.io_tools import load_metadata
import popy.config as cfg
from popy.decoding.glms import SingleUnitAnalysis

import warnings
pd.options.mode.chained_assignment = None  # default='warn', mute warnings
warnings.filterwarnings("ignore")


### Load metadata

def get_all_sessions():
    # get all session info
    session_metadata = load_metadata()
    session_metadata = session_metadata[session_metadata['block_len_valid'] == True]  # remove session with block_len_valid==False
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

PARAMS = {
    'model': 'glm_cpd',  # 'linear_correlation' or 'anova' or 'glm' or 'glm_cpd'
    'glm_predictors': ['feedback', 'value_function'],
    #'cpd_predictors': ['feedback', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6'],
    'cpd_targets': ['feedback', 'value_function'],  # ['feedback', 'R_1', 'R_2', 'R_3', 'R_4'],
    'neural_data_type': 'spike_counts',  # 'firing_rates' or 'spike_counts'
    'value_type': 'continuous',  # 'discrete' or 'continuous'
    'step_time': .05,
    'n_permutations': 500,

    'floc': os.path.join(cfg.PROJECT_PATH_LOCAL, 'notebooks', 'decoders', 'glm', 'results', 'fb_value_cpd'),  # Folder to save results
    'msg': 'Fitting a GLM with feedback and value, computing CPD for feedback and value with permutations',
    }

### Run

if __name__ == '__main__':
    init_io(PARAMS)  # Initialize logging and create results folder

    monkeys, sessions = get_all_sessions()

    n_cores = np.min([11, os.cpu_count()-1])  # get number of cores in the machine
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # submit jobs
        futures, future_proxy_mapping = [], {}  # init lists to store futures and their arguments
        for monkey, session in zip(monkeys, sessions):
            glm = SingleUnitAnalysis(monkey, session)
            for key, value in PARAMS.items(): 
                setattr(glm, key, value)
            future = executor.submit(glm.run)
            futures.append(future)
            future_proxy_mapping[future] = (monkey, session)

        # wait for results, save them
        count = 0
        xrs = []  # Container for results
        for future in concurrent.futures.as_completed(futures):
            try:
                res, session_log = future.result()
                monkey_fut, session_fut = future_proxy_mapping[future]

                # Append results to existing results and save after each session
                if len(xrs) == 0:  # First result - save directly
                    xrs = res
                else:  # Not first result - concatenate to existing results
                    xrs = xr.concat([xrs, res], dim='unit')

                # Save results after each session
                save_results(xrs, PARAMS['floc'])  # Save results after each session

                # Log progress
                for line in session_log:
                    logging.info(line)
                logging.info(f"Finished for monkey {monkey_fut} and session {session_fut}")
                                    
                # print log
                print(f'Progress: {count+1}/{len(monkeys)}')
                count += 1
                
            except Exception as e:
                logging.error(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}")
                print(f"Error occurred for arguments {future_proxy_mapping[future]}: {e}\n")
                traceback.print_exc()  # Print traceback (?)
                    
    end_log()
    print(f'Finished all on {datetime.datetime.now()}')
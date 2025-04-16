"""
This script processes the neural data for all sessions and saves the spike times in a netcdf file.
""" 

import pandas as pd
import os
import logging
import time

from popy.io_tools import process_neural_data, load_behavior
import popy.config as cfg

# Load the full behavior data
behavior_data = load_behavior()

# output path
fname_info = os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'neural_summary.csv')
floc_neural_data = os.path.join(cfg.PROJECT_PATH_LOCAL, 'data', 'processed', 'neural_data', 'spikes')
fname_log = os.path.join(floc_neural_data, 'neural_data_processing.log')

os.makedirs(floc_neural_data, exist_ok=True)

logging.basicConfig(filename=fname_log,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')  # 'w' mode will overwrite the log file

logging.info("Processing started at %s", time.strftime("%Y-%m-%d %H:%M:%S"))

meta_info = []
for (monkey, session), subdf in behavior_data.groupby(['monkey', 'session']):
    print(f'Processing {monkey}_{session}')
    try:
        # get spikes and rates: this function below reads the raw spike data
        # and creates an xarray dataset with the spike times
        spikes_train = process_neural_data(subdf, sr=1000)

        # get the session info
        for unit_name in spikes_train.unit.data:
            meta_info.append({
                'monkey': monkey,
                'session': session,
                'area': spikes_train.sel(unit=unit_name).area.data,
                'subregion': spikes_train.sel(unit=unit_name).subregion.data,
                'channel': spikes_train.sel(unit=unit_name).channel.data,
                'unit': spikes_train.sel(unit=unit_name).unit_id_original.data,
                'unit_zs': unit_name,
            })
        
        # save
        fname_temp = os.path.join(floc_neural_data, f'{monkey}_{session}_spikes.nc')
        spikes_train.to_netcdf(fname_temp, mode='w')
        print(f'Saved {monkey}_{session} spikes at {fname_temp}')

        logging.info(f"No problems for {monkey}_{session}")
    
    except Exception as e:
        print(f'Failed for {monkey}_{session}: {e}')
        logging.error(f'Failed for {monkey}_{session}: {e}')

# save the meta info
meta_info_df = pd.DataFrame(meta_info)
meta_info_df.to_csv(fname_info, index=False)
print(f'Saved meta info at {fname_info}')

logging.info("Processing finished at %s", time.strftime("%Y-%m-%d %H:%M:%S"))
print(f'N units = {len(meta_info_df)}')
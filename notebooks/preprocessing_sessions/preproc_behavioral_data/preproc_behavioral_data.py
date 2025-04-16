"""
This script processes the neural data for all sessions and saves the spike times in a netcdf file.
""" 
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import numpy as np
import os
import logging


from popy.io_tools import get_behavior, load_metadata
import popy.config as cfg

# file access paths
base_path = cfg.PROJECT_PATH_LOCAL
out_path = os.path.join(base_path, 'data', 'processed', 'behavior')

os.makedirs(out_path, exist_ok=True)

logging.basicConfig(filename=os.path.join(out_path, 'behav_processing_log.txt'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')  # 'w' mode will overwrite the log file

logging.info("Processing started")

# load all session metadata
session_metadata = load_metadata()

# process all sessions
log = []  # log errors here
df_long = pd.DataFrame()  # init dataframe
df_short = pd.DataFrame()  # init dataframe

for index, row in session_metadata.iterrows():
    monkey = row['monkey']
    session = row['session']
    block_len_valid = row['block_len_valid']

    try:
        session_data = get_behavior(monkey, session)  # load raw data and preprocess it
        if block_len_valid:
            df_long = pd.concat([df_long, session_data], ignore_index=True)  # collect all sessions in one dataframe
        else:
            df_short = pd.concat([df_short, session_data], ignore_index=True)
    except:
        logging.info(f"Error processing {monkey} {session}")

# save pandas to pickle
df_long.to_pickle(os.path.join(out_path, f'behavior.pkl'))  # save
df_short.to_pickle(os.path.join(out_path, f'behavior_blocklen_25.pkl'))  # save

logging.info(f"Long session behavioral data saved to {os.path.join(out_path, f'behavior.pkl')}")
logging.info(f"Short session bata saved to {os.path.join(out_path, f'behavior_blocklen_25.pkl')}")

#save log
logging.info("Processing finished")
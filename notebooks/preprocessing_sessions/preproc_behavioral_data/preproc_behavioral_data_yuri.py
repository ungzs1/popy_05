"""
This script processes the behavioral data of monkey Yuri (csv files in data/recordings/Yuri_behavior/behavior) for all
sessions and saves them in one pickle file. Load it with popy.io_tools.load_behavior(monkey='yu').
"""
#%%
import glob
import os
import logging

import pandas as pd

from popy.behavior_data_tools import get_behavior_yuri
import popy.config as cfg

# file access paths
pkl_path = cfg.COMBINED_DATA_PATH  # path to the folder containing the behavioral csv files
base_path = cfg.PROJECT_PATH_LOCAL
out_path = os.path.join(base_path, 'data', 'processed', 'behavior')

os.makedirs(out_path, exist_ok=True)

logging.basicConfig(filename=os.path.join(out_path, 'behav_processing_log_yu.txt'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')  # 'w' mode will overwrite the log file

logging.info("Processing started")
#%%
conditions = ['DCZ', 'sham']

for condition in conditions:
    # all sessions in the folder (ddmmyy), in chronological order
    session_flocs = [f for f in glob.glob(os.path.join(pkl_path, condition, '*_events.npz'))]
    #sessions = sorted(sessions, key=lambda s: (s[4:], s[2:4], s[:2]))

#%%
# process all sessions
all_sessions = []
failed = []
for session_floc_temp in session_flocs:
    session = os.path.basename(session_floc_temp).split('_')[0]  # extract session name from file path
    try:
        all_sessions.append(get_behavior_yuri(session_floc_temp, condition=condition))  # load raw data and preprocess it
        logging.info(f"Processed yu {session}")
    except Exception as e:
        failed.append(session)
        logging.info(f"Error processing yu {session}: {type(e).__name__}: {e}")

df = pd.concat(all_sessions, ignore_index=True)  # collect all sessions in one dataframe

# save pandas to pickle
df.to_pickle(os.path.join(out_path, 'behavior_yu_new.pkl'))

logging.info(f"{len(all_sessions)}/{len(session_flocs)} sessions saved to {os.path.join(out_path, 'behavior_yu.pkl')}")
logging.info(f"Failed sessions: {failed}")
logging.info("Processing finished")

print(f"{len(all_sessions)}/{len(session_flocs)} sessions saved to {os.path.join(out_path, 'behavior_yu.pkl')}, failed: {failed}")

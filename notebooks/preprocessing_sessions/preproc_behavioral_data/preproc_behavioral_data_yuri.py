"""
This script processes the behavioral data of monkey Yuri (<session>_events.npz files in cfg.COMBINED_DATA_PATH/<condition>)
for all sessions and saves them in one pickle file. Load it with popy.io_tools.load_behavior(monkey='yu').

The event codes are fixed first with fix_events_yuri.py (the npz files are not changed). Every change is saved in
behavior_yu_event_fixes_log.txt next to the pickle file.
"""
#%%
import glob
import os
import logging

import pandas as pd

from popy.behavior_data_tools import get_behavior_yuri
import popy.config as cfg

import fix_events_yuri
from fix_events_yuri import load_events_yuri, fix_events_yuri as fix_events

# file access paths
npz_path = cfg.COMBINED_DATA_PATH  # path to the folder containing the behavioral npz files (one folder per condition)
base_path = cfg.PROJECT_PATH_LOCAL
out_path = os.path.join(base_path, 'data', 'processed', 'behavior')
out_floc = os.path.join(out_path, 'behavior_yu.pkl')
fix_log_floc = os.path.join(out_path, 'behavior_yu_event_fixes_log.txt')

os.makedirs(out_path, exist_ok=True)

logging.basicConfig(filename=os.path.join(out_path, 'behav_processing_log_yu.txt'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')  # 'w' mode will overwrite the log file

logging.info("Processing started")
#%%
conditions = ['DCZ', 'sham']

# all sessions of all conditions: (condition, session, file path)
session_flocs = []
for condition in conditions:
    for session_floc in sorted(glob.glob(os.path.join(npz_path, condition, '*_events.npz'))):
        session_flocs.append((condition, os.path.basename(session_floc).split('_')[0], session_floc))

#%%
# process all sessions
all_sessions = []
failed = []
fix_logs = {}  # changes made to the event codes, per session
for condition, session, session_floc in session_flocs:
    try:
        events, fix_logs[session] = fix_events(load_events_yuri(session_floc))  # load raw events and fix them
        all_sessions.append(get_behavior_yuri(events, session, condition=condition))  # extract trials
        logging.info(f"Processed yu {condition} {session} ({len(fix_logs[session])} event fixes)")
    except Exception as e:
        failed.append(session)
        logging.info(f"Error processing yu {condition} {session}: {type(e).__name__}: {e}")

df = pd.concat(all_sessions, ignore_index=True)  # collect all sessions in one dataframe

# save pandas to pickle
df.to_pickle(out_floc)

# save the changes made to the event codes
with open(fix_log_floc, 'w') as f:
    f.write(f"CHANGES MADE TO THE BEHAVIORAL EVENT CODES OF YURI BEFORE EXTRACTING THE TRIALS ({os.path.basename(out_floc)})\n")
    f.write(f"Made by preproc_behavioral_data_yuri.py with fix_events_yuri.py (notebooks/preprocessing_sessions/"
            f"preproc_behavioral_data/). The npz files in {npz_path} are not changed.\n")
    f.write(f"Times are in s, on the time axis of the npz files ('time' column).\n\n")
    f.write("RULES (docstring of fix_events_yuri.py)\n")
    f.write(fix_events_yuri.__doc__)
    f.write("\nSUMMARY\n")
    rules = sorted({line.split(':')[0] for lines in fix_logs.values() for line in lines})
    for rule in rules:
        n = [sum(line.startswith(rule + ':') for line in lines) for lines in fix_logs.values()]
        f.write(f"  {rule}: {sum(n)} changes in {sum(x > 0 for x in n)} sessions\n")
    f.write(f"  failed sessions: {failed}\n")
    for condition, session, session_floc in session_flocs:
        if session not in fix_logs:
            continue
        f.write(f"\n=== {condition} {session} ({len(fix_logs[session])} changes)\n")
        for line in fix_logs[session]:
            f.write(f"  {line}\n")

logging.info(f"{len(all_sessions)}/{len(session_flocs)} sessions saved to {out_floc}")
logging.info(f"Event code fixes saved to {fix_log_floc}")
logging.info(f"Failed sessions: {failed}")
logging.info("Processing finished")

print(f"{len(all_sessions)}/{len(session_flocs)} sessions saved to {out_floc}, failed: {failed}")
print(f"Event code fixes saved to {fix_log_floc}")

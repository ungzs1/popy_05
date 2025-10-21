"""
Functions for extracting data from raw recording files. Mainly (or more like only) used by the io_tools functions.
"""

import pandas as pd
import numpy as np
import xarray as xr
import math
import warnings


### Behavioral data loading helper functions

def chop_trunctuated_blocks(session_data_base):
    """
    Chops the trunctuated blocks from the raw behaviour data, from the beginning and the end.
    :param raw_behav:
        numpy array, raw behaviour data
    :return:
        numpy array, raw behaviour data without trunctuated blocks
    """

    # copy the data
    session_data = session_data_base.copy()

    # block ids
    block_ids = session_data.block_id.unique()
    first_block, last_block = block_ids[0], block_ids[-1]

    # count lines per first and last block
    first_block_trials = len(session_data[session_data.block_id == first_block])
    last_block_trials = len(session_data[session_data.block_id == last_block])

    # remove block if trials are less then 30
    if first_block_trials < 30:
        session_data = session_data[session_data.block_id != first_block]
    if last_block_trials < 30:
        session_data = session_data[session_data.block_id != last_block]

    # reindex blocks and trials to count from zero
    block_ids = session_data.block_id.unique()
    trial_ids = session_data.trial_id.values
    if block_ids[0] != 0:
        #del session_data["block_id"]
        session_data["block_id"] -= block_ids[0]
        session_data["trial_id"] -= trial_ids[0]
 
    # rebase indexing 
    session_data.index = range(len(session_data))

    return session_data


def extract_behav_timestamps(raw_behav):
    """
    Extracts key behavioural event times.
    :return:
        pandas DataFrame, where the first column is the 'trial_id'.
    """

    # split behav by trials
    behav_pd = reformat_behav(raw_behav)

    # get keypoints for each trials
    behav_times = []  # dict to save trial timestamps of each trial temporarily
    for trial_id, curr_behav in behav_pd['behav_events'].items():
        # deriving basic times of key events  # ugly formattiong, repair sometimes..
        curr_trial = {
            'trial_id': trial_id,
            'trial_start_time': curr_behav[np.where(curr_behav == 100)[0], 1][0]
            }  # to save times and ids

        # if the trial is not interrupted, we can extract the other times
        if 252 not in curr_behav[:, 0]:
            curr_trial['lever_touch_time'] = get_lever_touch(curr_behav)
            curr_trial['lever_validation_time'] = curr_behav[np.where(curr_behav == 62)[0], 1][0]
            curr_trial['lever_release_time'] = curr_behav[np.where(curr_behav == 64)[0], 1][0]
            # 'lever_release_time': curr_behav[np.where(curr_behav == 64)[0], 1][0]
            curr_trial['target_touch_time'], curr_trial['n_target_touches'] = get_target_touch(curr_behav)
            curr_trial['target_validation_time'] = curr_behav[np.where(curr_behav == 125)[0], 1][0]
            curr_trial['feedback_time'] = curr_behav[np.where((curr_behav[:, 0] == 65) | (curr_behav[:, 0] == 66))[0], 1][0]
        else:  # if the trial is interrupted, we can't (won't) extract the other times
            curr_trial['lever_touch_time'] = np.nan
            curr_trial['lever_validation_time'] = np.nan
            curr_trial['lever_release_time'] = np.nan
            curr_trial['target_touch_time'] = np.nan
            curr_trial['target_validation_time'] = np.nan
            curr_trial['feedback_time'] = np.nan

        # trial end is next trial start. it would be possible to use the trial end 
        # marker (101) to get the trial end time, but it creates a small gap between trials. better to use next trial start
        # (except for last trial, where its the trial end event)
        if not trial_id == behav_pd.trial_id.max():
            behav_next = behav_pd.iloc[trial_id + 1]['behav_events']
            curr_trial['trial_end_time'] = behav_next[np.where(behav_next[:, 0] == 100)[0], 1][0]
        else:
            curr_trial['trial_end_time'] = curr_behav[np.where(curr_behav[:, 0] == 101)[0], 1][0]

        # appendng times together with previous trials
        behav_times.append(curr_trial)

    # create dataframe and sort columns to follow behav sequence order in trials
    behav_keypoints = pd.DataFrame(behav_times).sort_values(0, axis=1)

    # downsampling to 10000 Hz (more precisely, to 4 decimals)
    behav_keypoints.loc[:, behav_keypoints.columns != 'trial_id'] = behav_keypoints.loc[:, behav_keypoints.columns != 'trial_id'].round(4)

    return behav_keypoints


def extract_behav_events(raw_behav):
    """
    Extracts behavioural metadata for each trial. Exact measures are derived here. More complex measures defined later.
    :return:
        pandas DataFrame, where the first column is the 'trial_id'.
    """

    # split behav by trials
    behav_pd = reformat_behav(raw_behav)

    # get keypoints for each trials
    behav_all = []  # dict to save trial timestamps of each trial temporarily
    for trial_id, curr_behav in behav_pd['behav_events'].items():
        # extract basic times of key events  # ugly formattiong, repair sometimes..
        curr_trial = dict()  # to save times and ids

        # extract best target
        curr_trial['trial_id'] = trial_id
        curr_trial['block_id'], curr_trial['best_target'] = get_block_info(trial_id, raw_behav)
        
        # if the trial is not interrupted, we can extract the other times
        if 252 not in curr_behav[:, 0]:
            curr_trial['target'] = get_selected_target(curr_behav)  # extract selected target
            curr_trial['feedback'] = get_reward_info(curr_behav)  # extract target info
        else:  # if the trial is interrupted, we can't (won't) extract the other times
            curr_trial['target'] = np.nan
            curr_trial['feedback'] = np.nan

        # appendng times together with previous trials
        behav_all.append(curr_trial)

    # create dataframe and sort columns to follow behav sequence order in trials
    behav_meta = pd.DataFrame(behav_all)
    
    '''# set datatypes
    behav_meta['block_id'] = behav_meta['block_id'].astype(int)
    behav_meta['best_target'] = behav_meta['best_target'].astype(int)
    behav_meta['target'] = behav_meta['target'].astype(int)
    behav_meta['feedback'] = behav_meta['feedback'].astype(int)'''

    return behav_meta


def reformat_behav(raw_behav):
    """
    Splits the raw behav file shape=(#events, 2) by trials. All trial starts with a behav code 100 and ens with a code
    of 101.
    :param raw_behav:
    :return:
        pandas DataFrame, where the keys are the 'trial_id' (first column), and the data is in the 'behav_events' column
        with shape=(#events_in_trial, 2)
    """
    trial_start_ids = np.where(raw_behav[:, 0] == 100)[0]
    trial_end_ids = np.where(raw_behav[:, 0] == 101)[0]

    all_trials = []
    for trial_id, (trial_start, trial_end) in enumerate(zip(trial_start_ids, trial_end_ids)):
        curr_trial = {
            'trial_id': trial_id,
            'behav_events': raw_behav[trial_start:trial_end + 1, :]
        }
        all_trials.append(curr_trial)

    return pd.DataFrame(all_trials)


def get_lever_touch(behav):
    """
    Lever touch event is the last lever touch (code 61) before lever validation (code 62), so we need the last 61 before
    the only 62 event.
    """

    id_62 = np.argwhere(behav[:, 0] == 62)
    ids_61 = np.argwhere(behav[:, 0] == 61)
    last_61_before_62 = np.max(ids_61[ids_61 < id_62])

    return behav[last_61_before_62, 1]


def get_target_touch(behav):
    """
    Target touch event is the last target touch (code 121/122/123) before targegt validation (code 125),
    so we need the last 121/122/123 before the only behav event.
    """

    id_125 = np.argwhere(behav[:, 0] == 125)
    ids_target_touch = np.argwhere((behav[:, 0] == 121) | (behav[:, 0] == 122) | (behav[:, 0] == 123))
    last_traget_touch_id = np.max(ids_target_touch[ids_target_touch < id_125])
    n_target_touches = len(ids_target_touch[ids_target_touch < id_125])

    return behav[last_traget_touch_id, 1], n_target_touches


def get_block_info(trial_id, behav):
    """
    Returns block id and best target. Block id is derived simply by counting block start events (7). Best target is
    encoded in the second event as 51/52/53 for target 1/2/3 respectively, and the code appears after the 7 event
    and after the trial start (100) event. But many times it is missing. So we collect all 51/52/53 events in a set and
    return the one unique value for all trials in this block.
    """

    block_start_index = np.argwhere(behav[:, 0] == 7)
    session_end_index = np.argwhere(behav[:, 0] == 102)
    if len(session_end_index) == 0:
        session_end_index = behav.shape[0]  # if the session end signal is missing, we assume it is the last event

    block_border_index = np.append(block_start_index, session_end_index)

    trial_count = 0
    for block_id, (block_start, block_end) in enumerate(zip(block_border_index[:-1], block_border_index[1:])):
        curr_block = behav[block_start:block_end]
        all_best_targets_found = np.concatenate((curr_block[curr_block[:, 0] == 51, 0],
                                                 curr_block[curr_block[:, 0] == 52, 0],
                                                 curr_block[curr_block[:, 0] == 53, 0])
                                                )
        best_target = set(all_best_targets_found)

        N_trials_in_block = np.count_nonzero(curr_block[:, 0] == 100)
        trial_count += N_trials_in_block

        if trial_id < trial_count:
            # return code if it appears, NaN if it doesn't
            if len(best_target) == 1:
                return block_id, int(list(best_target)[0] - 50)
            else:
                print(f'Warning: no best target of multiple best targets in block ({len(best_target)}).')
                print(f'block_start: {block_start}, block_end: {block_end}')
                return np.nan, np.nan
            
    raise ValueError(f'Trial id {trial_id} not found in any block.')


def get_reward_info(behav):
    """
    Rewarded / Non-rewarded trials has a code of 65/66, resp.
    """

    best_target_code = behav[((behav[:, 0] == 65) | (behav[:, 0] == 66)), 0]

    # return code if it appears, NaN if it doesn't
    if len(best_target_code) == 1:
        return not bool(best_target_code - 65)
    else:
        return np.nan


def get_selected_target(behav):
    """
    Target selection events can occure multiple times. We only want to see here what the validated target was. This is
    the last target selection (121/122/123) before target validation (125).
    """

    id_125 = np.argwhere(behav[:, 0] == 125)
    ids_target_touch = np.argwhere((behav[:, 0] == 121) | (behav[:, 0] == 122) | (behav[:, 0] == 123))

    last_traget_touch_id = np.max(ids_target_touch[ids_target_touch < id_125])
    last_target_touch_code = behav[last_traget_touch_id, 0]

    return int(last_target_touch_code - 120)

### Neural data loading helper functions

def convert_firing(behav, spikes_raw, area, sr=1000):
    """
    Converts the raw spike times to spike trains for each unit. Spike trains are vectors of 0's and 1's, where 1's
    represent the time of a spike.

    The time resolution of the spike trains is determined by the sampling rate (sr, 1/sec).

    The data is in an xarray format, where the dimensions are:
        - unit: the unit id
        - time: the time of the spike

    """

    # set sampling rate for the spike train generation 
    decimals = int(math.log10(sr))  # number of decimals to round to

    # resample spike times and behav times
    spikes_raw.time = spikes_raw.time.round(decimals)
    behav.loc[:, behav.columns != 'trial_id'] = behav.loc[:, behav.columns != 'trial_id'].round(decimals)

    # list all units in this session and corresponding meta info (e.g. recording channel)
    unit_info = get_unit_info(spikes_raw, area)

    # init xarray with all zeros
    session_start, session_end = behav.trial_start_time.min()-2, behav.trial_end_time.max()+2  # add 2 seconds to the beginning and end of the session
    first_spike, last_spike = spikes_raw.time.min(), spikes_raw.time.max()
    if first_spike > session_start or last_spike < session_end:  # check if the session is fully covered by the spikes
        warnings.warn(f"Session not fully covered by spikes. First spike: {first_spike}, last spike: {last_spike}, session start: {session_start}, session end: {session_end}")
        if first_spike > session_start:
            session_start = first_spike -1
        if last_spike < session_end:
            session_end = last_spike +1
            
    spike_train = init_spike_train(unit_info, session_start, session_end, sr)

    # fill xarray spike trains unit-by-unit
    for _, curr_unit_info in unit_info.iterrows():
        unit_id = curr_unit_info['unit_id']
        unit_id_original = curr_unit_info['unit_id_original']

        # select spikes for this unit and format to numpy
        spike_times_np = spikes_raw.loc[spikes_raw['id'] == unit_id_original, 'time'].to_numpy()
        # extract the session interwal for this unit
        spike_times_np = spike_times_np[(spike_times_np >= session_start) & (spike_times_np <= session_end-(1/sr))]  # last bin removed to avoid index out of bounds
        #spike_times_np = spike_times_np[:-1]
        
        # converting absolute spike times to bin id's
        spike_ids = (spike_times_np - session_start) * sr
        spike_ids = spike_ids.astype(int)

        # change bins corresponding to spikes to value 1 (all others store a value of 0)
        spike_train.sel(unit=unit_id)[spike_ids] += 1  # normally always a value of one, but this method handles abnormally close spikes

    return spike_train


def get_unit_info(spikes_raw, area):
    """
    A pandas DataFrame containing all units and the corresponding information (i.e. channel, etc...). Also redefines
    'unit id' so that it is meaningful (i.e. ordered by depth and the unit id describes the channel). E.g. unit 0403
    if the 3rd unit on channel 4.
    """

    unit_info = []
    unit_count_by_channel = {channel:1 for channel in range(1, 17)}
    for unit_label in spikes_raw.id.unique():
        sample_row = spikes_raw[spikes_raw.id == unit_label].iloc[0]
        curr_unit_info = {
            'unit_id': f"{area}_{str(sample_row.ch).zfill(2)}_{str(unit_count_by_channel[sample_row.ch]).zfill(2)}",  # new indexing
            'unit_id_original': unit_label,
            'channel': sample_row.ch,
            'Amplitude': sample_row.Amplitude,
            'amp': sample_row.amp,
            'ContamPct': sample_row.ContamPct
        }
        # increase unit count on given channel
        unit_count_by_channel[sample_row.ch] += 1
        unit_info.append(curr_unit_info)

    unit_info = pd.DataFrame(unit_info).sort_values(['channel'])

    return unit_info.reset_index(drop=True)


def init_spike_train(unit_info, trial_start, trial_end, sr):
    # init xarray with all zeros
    #trial_start = curr_trial_behav.trial_start_time
    #trial_end = curr_trial_behav.trial_end_time
    
    time = np.arange(trial_start, trial_end, 1/sr)
    #time_in_trial = time_in_session - trial_start
    #time_event = get_event_borders(curr_trial_behav)

    N_units = len(unit_info)  # number of units in the session
    N_bins = len(time)  # number of units in the session

    spike_train = xr.DataArray(np.zeros((N_units, N_bins), dtype='int8'),
                               coords={
                                   "unit": unit_info.unit_id.tolist(),
                                   "time": time,
                                   "unit_id_original": ("unit", unit_info.unit_id_original.tolist()),
                                   "channel": ("unit", unit_info.channel.tolist()),
                                   #"area": ("unit", unit_info.area.tolist()),
                                   #"LPFC_subregion": ("unit", unit_info.LPFC_subregion.tolist()),
                                   #"time_in_session": ("time", time_in_session),
                               },
                               dims=["unit", "time"])
    #spike_train['bin_size'] = 1/sr
    #spike_train.attrs['behav_info'] = curr_trial_behav.filter(regex='_time$').to_dict()
    return spike_train

### Behav + Neural data loading helper functions

def add_behav_info(neural_data, session_data):   
    """
    Add trial and epoch info to spike train or firing rate data.
    """
    # get trial and epoch info
    trial_vector = np.full(len(neural_data.time), np.nan)
    epoch_vector = np.full(len(neural_data.time), np.nan)

    # fill trial and epoch info
    for trial_id in session_data.trial_id.unique():
        session_data_temp = session_data[session_data.trial_id == trial_id]

        # epoch borders
        trial_start = session_data_temp.trial_start_time.values[0]
        lever_touch = session_data_temp.lever_touch_time.values[0]
        lever_validation = session_data_temp.lever_validation_time.values[0]
        target_touch = session_data_temp.target_touch_time.values[0]
        target_validation = session_data_temp.target_validation_time.values[0]
        feedback = session_data_temp.feedback_time.values[0]
        trial_end = session_data_temp.trial_end_time.values[0]

        # fill in trial info
        time_ids_trial = np.where((neural_data.time >= trial_start) & (neural_data.time < trial_end))[0]
        trial_vector[time_ids_trial] = trial_id

        if neural_data.time[0] > trial_start:
            warnings.warn(f"Trial {trial_id} starts at {trial_start}, but the first time point in neural data is {neural_data.time[0]}. "
                          f"Recording starts in the middle of a trial, thus filling the epoch info to nans and passing to the next trial.")
            continue
        if trial_end >= neural_data.time[-1]:
            warnings.warn(f"Trial {trial_id} ends at {trial_end}, but the last time point in neural data is {neural_data.time[-1]}. "
                          f"Recording end in the middle of a trial, thus filling the epoch infor to nans and ending the session here.")
            break
        else:
            # fill in epoch info
            time_ids = {
                0: np.where((neural_data.time >= trial_start) & (neural_data.time < lever_touch))[0],
                1: np.where((neural_data.time >= lever_touch) & (neural_data.time < lever_validation))[0],
                2: np.where((neural_data.time >= lever_validation) & (neural_data.time < target_touch))[0],
                3: np.where((neural_data.time >= target_touch) & (neural_data.time < target_validation))[0],
                4: np.where((neural_data.time >= target_validation) & (neural_data.time < feedback))[0],
                5: np.where((neural_data.time >= feedback) & (neural_data.time < trial_end))[0]
            }

            epoch_vector[time_ids[0]] = 0
            epoch_vector[time_ids[1]] = 1
            epoch_vector[time_ids[2]] = 2
            epoch_vector[time_ids[3]] = 3
            epoch_vector[time_ids[4]] = 4
            epoch_vector[time_ids[5]] = 5

    # add to neural_data
    neural_data = neural_data.assign_coords(trial_id=("time", trial_vector))
    neural_data = neural_data.assign_coords(epoch_id=("time", epoch_vector))

    return neural_data

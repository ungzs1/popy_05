"""
These functions are used to load the raw data (behavior and neural) from the folder data/recordings and to preprocess the data.

Output functions should also go here.
"""
import os
import scipy.io

import popy.config as config # project config file
from popy.io_tools_helpers import *  # behavior preprocessing functions


### loading functions for RAW DATA - these are used to generate the standard neural and behavioral data formats (i.e. for during datapreprocessing), but not during the analysis!!! ###

def load_raw_behav_mat(monkey, session, folder_path):
    """
    Load the raw behav file, and return a pandas DataFrame with 2 columns: behav event and corresponding time.
    Also checks if the behav file is corrupt, and raises error if it is (so that no downstream processing can
    be done on corrupt data)! Note that corrupt data can be processed (in many cases), but the results will be
    misleading.

    :param monkey: 'ka' or 'po'
    :param session: sesssion number
    :param folder_path:
    :return:
    """
    prefix = f'{monkey}{session}_'
    prefixed_fnames = [filename for filename in os.listdir(folder_path) if
                       filename[:len(prefix)] == prefix and filename[-4:] == '.mat']

    # if more than one found
    if len(prefixed_fnames) != 1:
        raise ValueError(f"zero or more than one behav file found for monkey {monkey} and session {session}:\n"
                         f"files: {prefixed_fnames}")

    # load the behav file from .mat
    behav_raw = scipy.io.loadmat(os.path.join(folder_path, prefixed_fnames[0]))

    return behav_raw['behav']


def load_raw_spikes_txt(monkey, session, area, folder_path):
    """
    Load the raw spikes file, and return a pandas DataFrame with 13 columns, the ones that Clem defined in his code.
    Most of the columns are completely nonsense.

    Raises error if zero or more than one file is found for the given monkey, session and area.
    """

    prefix = f'spk_dataset_{area}_{monkey}{session}_'
    prefixed_fnames = [filename for filename in os.listdir(folder_path) if filename[:len(prefix)] == prefix and filename[-4:] == '.txt']

    # if more than one found
    if len(prefixed_fnames) != 1:
        raise ValueError(f"({len(prefixed_fnames)}) spike file found for area: {area}, monkey: {monkey}, session: {session}:\n"
                         f"files: {prefixed_fnames}")
    else:
        spikes = pd.read_csv(os.path.join(folder_path, prefixed_fnames[0]))
        return spikes


def get_behavior(monkey, session):
    """
    This function loads the raw behaviour (.mat file) and extracts the timestamps and the events. Returns a formatted pandas dataframe.
    """

    # get the path to the raw data folder
    RAW_DATA_PATH = os.path.join(config.PROJECT_PATH_LOCAL, 'data', 'recordings', 'behavior')

    raw_behav = load_raw_behav_mat(monkey, session, folder_path=RAW_DATA_PATH)  # load raw behav data

    # chop trunctuated blocks in the beginning and the end
    #raw_behav = chop_trunctuated_blocks(raw_behav)

    times_df = extract_behav_timestamps(raw_behav)  # extract timestamps
    events_df = extract_behav_events(raw_behav)   # extract events

    # concatenate times and events and return
    joint_df = times_df.join(events_df.set_index('trial_id'), on='trial_id')

    # add monkey and session info, replace it to the first 2 cols
    joint_df['monkey'] = monkey
    joint_df['session'] = session 
    # rearrange: put monkey and session info in the front.
    cols = joint_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    joint_df = joint_df[cols]

    # remove too short blocks from beginning and end of session
    # a block is too short if it is shorter than 30 trials
    joint_df = chop_trunctuated_blocks(joint_df)

    return joint_df


def process_neural_data(behav_data, sr=1000, *args, **kwargs):
    """
    Process the neural data for one session.

    This function loads the raw neural data (.txt file) and extracts the timestamps and the events. Returns a formatted xarray dataset.

    :session_data: pandas dataframe with behavior data (including monkey and session name)
    :param mode: 'spikes' or 'rates'
    :param sr: sampling rate for the spike train in Hz
    """
    assert len(args) == 0, f"Unknown positional arguments: {args} - do not provide 'mode' parameter, it is deprecated. It will return spike trains by default, and you can generate rates from them."
    assert len(kwargs) == 0, f"Unknown keyword arguments: {kwargs} - do not provide 'mode' parameter, it is deprecated. It will return spike trains by default, and you can generate rates from them."

    # get the path to the raw data folder
    RAW_DATA_PATH = os.path.join(config.PROJECT_PATH_LOCAL, 'data', 'recordings', 'neural_data')

    # get monkey and session info
    monkey = behav_data['monkey'].iloc[0]
    session = behav_data['session'].iloc[0]

    # load metadata, and check if there is neural data for mcc and lpfc 
    metadata = load_metadata()
    metadata = metadata.loc[(metadata['monkey'] == monkey) & (metadata['session'] == session)]

    lpfc_available = metadata['LPFC_spikes_exist'].iloc[0]
    mcc_available = metadata['MCC_spikes_exist'].iloc[0]
    lpfc_subregion = metadata['LPFC_subregion'].iloc[0]
    # load behav data
    # load raw spike train
    xr_both = []
    if lpfc_available:
        spikes_raw_lpfc = load_raw_spikes_txt(monkey, session, "LPFC", folder_path=RAW_DATA_PATH)
        xr_lpfc = convert_firing(behav_data, spikes_raw_lpfc, area='LPFC', sr=sr)  # generate spike train dataset (xr.DataArray)
        # add monkey, session, area and subregion info to the dataset, ac coordinates along the unit axis
        xr_lpfc = xr_lpfc.assign_coords({'monkey': ('unit', [monkey] * xr_lpfc.sizes['unit']),
                                          'session': ('unit', [session] * xr_lpfc.sizes['unit']),
                                          'area': ('unit', ['LPFC'] * xr_lpfc.sizes['unit']),
                                          'subregion': ('unit', [lpfc_subregion] * xr_lpfc.sizes['unit'])})
        xr_both.append(xr_lpfc)

    if mcc_available:
        spikes_raw_mcc = load_raw_spikes_txt(monkey, session, "MCC", folder_path=RAW_DATA_PATH)
        xr_mcc = convert_firing(behav_data, spikes_raw_mcc, area='MCC', sr=sr)  # generate spike train dataset (xr.DataArray)
        xr_mcc = xr_mcc.assign_coords({'monkey': ('unit', [monkey] * xr_mcc.sizes['unit']),
                                          'session': ('unit', [session] * xr_mcc.sizes['unit']),
                                          'area': ('unit', ['MCC'] * xr_mcc.sizes['unit']),
                                          'subregion': ('unit', ['MCC'] * xr_mcc.sizes['unit'])})
        xr_both.append(xr_mcc)

    # concatenate spike trains on time axis (if only one exists, it is supposed to work as well)
    if len(xr_both) == 2:
        xr_both = xr.align(xr_both[0], xr_both[1], join="inner", exclude=['unit'])  # Align the two datasets to keep only the overlapping parts
    xr_spikes = xr.concat(xr_both, dim='unit')

    #  append trial and epoch info
    xr_spikes = add_behav_info(xr_spikes, behav_data)

    # add sampling rate info
    xr_spikes.attrs['bin_size'] = 1/sr  # in seconds
    
    return xr_spikes


# LOADING FUNCTIONS FOR PROCESSED DATA #

def load_simulation(monkey=None):
    """
    Simply loads the simulation dataset so that its not necessary to load it every time from path.
    """

    base_path = config.PROJECT_PATH_LOCAL
    data_path = os.path.join(base_path, 'data', 'processed', 'behavior', f'simulation_{monkey}.pkl')
    return pd.read_pickle(data_path)


def load_behavior(monkey=None, session=None):
    """
    Simply loads the full behavior dataset so that its not necessary to load it every time from path.
    """

    base_path = config.PROJECT_PATH_LOCAL
    data_path = os.path.join(base_path, 'data', 'processed', 'behavior', 'behavior.pkl')
    concat_data = pd.read_pickle(data_path)
    
    # set 'feedback' to float
    concat_data['feedback'] = concat_data['feedback'].astype(float)
    
    if (monkey is not None) and (session is not None):
        assert session in concat_data.loc[concat_data['monkey'] == monkey, 'session'].values, f"Session {session} not found for monkey {monkey}"
        return concat_data.loc[(concat_data['monkey'] == monkey) & (concat_data['session'] == session)]
    elif (monkey is not None) and (session is None):
        return concat_data.loc[(concat_data['monkey'] == monkey)]
    elif (monkey is None) and (session is not None):
        raise ValueError("provide monkey name if you want to filter by session")    
    elif (monkey is None) and (session is None):
        return concat_data
    

def load_neural_data(monkey, session, hz=100, *args, **kwargs):
    """
    Loading function for one session's neural data in xarray format.

    Returns the spike data in xarray.Dataset format.

    Spike trains, as xarray.DataArray, are stored in the 'spike_trains' variable of the dataset and can be accessed as:
    `neural_data['spike_trains']`.

    Parameters:
        monkey (str): 'ka' or 'po'
        session (str): Session number

    Returns:
        xarray.Dataset: The neural data in xarray.Dataset format
    """
    import warnings

    if hz == 100:
        raise ValueError("100 Hz data is not supported anymore, use 1000 Hz data and downsample after firing rate generation with: .")

    # SANITY: check if there is anything in args and kwargs -> older versions of the function had 'mode' parameter and 'return_dataset_format' parameter
    if len(args) > 0:
        raise ValueError(f"'mode' and 'return_dataset_format' parameters are deprecated, use 'neural_data = load_neural_data(monkey, session)' instead, that will return xarray.Dataset.\n\
            To get xarray.DataArray of spikes, use the 'spike_trains' variable of the dataset as 'neural_data.spike_trains'.\n\
            To get firing rates, use the 'neural_data = add_firing_rates(neural_data)' function from popy.neural_data_tools, then you can access firing rates as 'neural_data.firing_rates'.")
    if len(kwargs) > 0:
        if 'mode' in kwargs:
            kwargs.pop('mode')
            raise ValueError("mode parameter is deprecated, instead of loading rates, generate it from the spike trains -> Load spike trains without ```mode``` param, \n\
                and then generate rates with 'add_firing_rates()' function from popy.neural_data_tools.")
        if 'return_dataset_format' in kwargs:
            kwargs.pop('return_dataset_format')
            raise ValueError("old convention is depricated, function will return xarray.Dataset. To get xarray.DataArray of spikes, use the 'spike_trains' variable of the dataset as 'neural_data.spike_trains'.\n\
                          To get firing rates, use the 'neural_data = add_firing_rates(neural_data)' function from popy.neural_data_tools, then you can access firing rates as 'neural_data.firing_rates'.")
        if len(kwargs) > 0:
            raise ValueError(f"Unknown parameters: {kwargs.keys()}")
        
    # Load neural data
    '''out_path = os.path.join(config.PROJECT_PATH_LOCAL, 'data', 'processed', 'neural_data')
    floc = os.path.join(out_path, f'{monkey}_{session}_spikes.nc')'''
    if hz == 1000:
        out_path = os.path.join(config.PROJECT_PATH_LOCAL, 'data', 'processed', 'neural_data', 'spikes')

    elif hz == 100:    
        out_path = os.path.join(config.PROJECT_PATH_LOCAL, 'data', 'processed', 'neural_data_old_100hz', 'spikes')
    floc = os.path.join(out_path, f'{monkey}_{session}_spikes.nc')

    neural_data = xr.open_dataarray(floc)
    neural_data = neural_data.astype(float)
    neural_data.attrs['bin_size'] = round(neural_data.time[1].values - neural_data.time[0].values, 4)

    # make it a dataset
    neural_data = neural_data.to_dataset(name='spike_trains')
    neural_data.attrs['bin_size'] = round(neural_data.time[1].values - neural_data.time[0].values, 4)

    # Load the data into memory
    neural_data.load()

    return neural_data


def load_metadata():
    # load metadata
    metadata = pd.read_csv(os.path.join(config.PROJECT_PATH_LOCAL, 'data', 'recordings_summary.csv'), dtype={'session': str})
    return metadata

def load_neural_metadata():
    # load metadata
    metadata = pd.read_csv(os.path.join(config.PROJECT_PATH_LOCAL, 'data', 'neural_summary.csv'), dtype={'session': str})
    return metadata

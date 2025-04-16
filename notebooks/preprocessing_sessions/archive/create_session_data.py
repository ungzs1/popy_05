"""
This script can be used to create a session data dataframe, with behav and spike data, and to save it so that the preprocessing does not need to be done every time.

The code is to be rewieved and updated, but it is a good starting point.
"""

from popy.io_tools import PoPy
from popy.decoding.population_decoders import *
from popy.neural_data_tools import *

#%%
sessions = pd.read_csv('/Users/zsombi/OneDrive/PoPy/data/recordings_summary.csv', dtype={'session': str})
all_sessions = sessions.loc[((sessions.behav_valid == True) &
                         (sessions['LPFC_spikes_sorted?'] == True) &
                         (sessions['MCC_spikes_sorted?'] == True) &
                         (sessions.monkey == 'ka')), 'session']
monkey, session = 'ka', '050620'

# load behav data
behav = PoPy(monkey, session).get_data('behaviour')
behav = add_switch_info(behav)
behav = add_phase_info(behav)

# load spike data
spikes = PoPy(monkey, session).get_data('spikes')
session_data = join_session_info(behav, spikes)

# process MCC
area = "MCC"
session_data = remove_low_fr_units(session_data, f"{area}_spikes", limit=.0)
session_data = add_firing_rates(session_data, f"{area}_spikes", method="gauss", std=.050)
session_data = remove_missing_neurons(session_data, f"{area}_rates")
session_data = time_normalize_trials(session_data, f"{area}_rates")
session_data = join_bins(session_data, f"MCC_rates", 3)
session_data = scale_data(session_data, f'MCC_rates')

# process LPFC
area = "LPFC"
session_data = remove_low_fr_units(session_data, f"{area}_spikes", limit=.0)
session_data = add_firing_rates(session_data, f"{area}_spikes", method="gauss", std=.050)
session_data = remove_missing_neurons(session_data, f"{area}_rates")
session_data = time_normalize_trials(session_data, f"{area}_rates")
session_data = join_bins(session_data, f"LPFC_rates", 3)
session_data = scale_data(session_data, f'LPFC_rates')

# merge areas
session_data = merge_areas(session_data)

session_data.to_pickle(f'/Users/zsombi/OneDrive/PoPy/data/processed/session_data_{monkey}{session}.pickle')
print('saved')




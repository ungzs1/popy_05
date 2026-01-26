import sys
sys.path.append("C:\ZSOMBI\OneDrive\PoPy")
sys.path.append("/Users/zsombi/OneDrive/PoPy")

import pandas as pd
from tqdm import tqdm

from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.population_decoders import *
from popy.plotting.plotting_tools import show_target_selection
import popy.config as cfg


project_folder = cfg.PROJECT_PATH_LOCAL

# sessions info
sessions = pd.read_pickle(os.path.join(project_folder, 'data', 'recordings_summary.pickle'))
out_path = os.path.join('data', 'processed', 'rates')

save = False
load = True

do_in_time_decoder = True
do_across_time_decoder = False
do_weights_decoder = False

error_sessions = []

# loop through sessions
for i, row in sessions.iterrows():
    if not row.behav_valid:
        continue
    if row.interrupted_trials != 0:
        continue

    monkey, session = row.monkey, row.session
    print(f"Monkey: {monkey}, session: {session}")
    try:
        ## load behavior data
        session_data = get_behavior(monkey, session)
        if len(session_data) < 200:
            print('Session too short')
            continue
        # add necessary behav. variables
        #session_data = add_phase_info(session_data, exploration_limit=1, transition_limit=6)  # add double feedback label
        session_data = add_value_function(session_data)  # add action value
        #session_data = add_switch_info(session_data)  # add switch info
        #session_data = add_RPE(session_data)  # add reward prediction error

        # clean up data
        session_data = drop_time_fields(session_data)  # remove time fields
        session_data = session_data.drop(['block_id', 'best_target'], axis=1)  # drop block_id and best_target
        
        ## Load or generate neural data
        floc = os.path.join(out_path, f'neural_data_{monkey}_{session}.nc')
        if load:
            neural_data = xr.open_dataarray(floc)
        else:
            neural_data = get_neural_data(monkey, session, 'rates', sr=100)
            # remove trunctuated units
            neural_data = remove_trunctuated_neurons(neural_data, delay_limit=10)
            # remove low_firing units
            neural_data = remove_low_fr_neurons(neural_data, 1)
            # z-score neural data
            neural_data = scale_neural_data(neural_data)
            # normalize neural data in time
            neural_data = time_normalize_session(neural_data)

        if save:
            neural_data.to_netcdf(floc)

        ## Prepare data for decoding
        # in order to decode Q_t+1, we need to shift the value_function column by 1
        session_data['value_function'] = session_data['value_function'].shift(-1)
        #session_data['switch'] = session_data['switch'].shift(-1)
        session_data['target'] = session_data['target']
        session_data = session_data.dropna()

        # make it binary: search or repeat only, drop transition (in fact change it to nan)
        #session_data.loc[session_data.phase == 'transition', 'phase'] = np.nan

        # df to store results
        all_scores = pd.DataFrame(columns=['value_function'],  # , 'phase'
                                  index=['MCC', 'LPFC'])
        '''all_weights = pd.DataFrame(columns=['feedback', 'value_function', 'RPE', 'switch'],  # , 'phase'
                                   index=['MCC', 'LPFC'])
        all_scores_across = pd.DataFrame(columns=['feedback', 'value_function', 'RPE', 'switch', 'target'],  # , 'phase'
                                         index=['MCC', 'LPFC'])'''

        # run decoding for each condition and store results in df
        window_len = .2
        step_len = .1
        # add new column for rpe
        
        if do_in_time_decoder:
            for target_name in all_scores.columns:
                for area in all_scores.index:
                    print(f"time-resolved - {target_name} in {area} area")
                    all_scores[target_name][area] = linear_decoding(neural_data, session_data, target_name, area=area, K_fold=5, starts_on='trial_start', ends_on='next_trial_end', window_len=window_len, step_len=step_len, return_p_values=False)
        
        '''elif do_weights_decoder:
            for target_name in all_weights.columns:
                for area in all_weights.index:
                    print(f"weights - {target_name} in {area} area")
                    all_weights[target_name][area] = linear_decoding(neural_data, session_data, target_name, mode='weights', area=area, K_fold=5, window_len=window_len, step_len=step_len)
        elif do_across_time_decoder:
            for target_name in all_scores_across.columns:
                for area in all_scores_across.index:
                    print(f"across time - {target_name} in {area} area")
                    all_scores_across[target_name][area] = linear_decoding(neural_data, session_data, target_name, mode='across_time', area=area, window_len=window_len, step_len=step_len) '''

        # save results
        if do_in_time_decoder:
            all_scores['#units'] = [np.sum(neural_data.area.values == "MCC"),
                                    np.sum(neural_data.area.values == "LPFC")]
            all_scores.to_pickle(os.path.join(project_folder, 'results', 'stoll', f'neurofrance_decoders_{monkey}_{session}.pickle'))
        '''elif do_weights_decoder:
            all_weights['#units'] = [np.sum(neural_data.area.values == "MCC"),
                                    np.sum(neural_data.area.values == "LPFC")]
            all_weights.to_pickle(project_folder + '\\results\\stoll' + f'\\all_weights_{monkey}_{session}.pickle')
        elif do_across_time_decoder:
            all_scores_across['#units'] = [np.sum(neural_data.area.values == "MCC"),
                                       np.sum(neural_data.area.values == "LPFC")]
            all_scores_across.to_pickle(project_folder + '\\results\\stoll' + f'\\all_scores_across_{monkey}_{session}.pickle')'''
    except:
        error_sessions.append((monkey, session))
        print(f'Error in {monkey} {session}')

print('Done')
print('Error sessions:')
print(error_sessions)

from popy.io_tools import PoPy
from popy.decoding.population_decoders import *
from popy.neural_data_tools import *
from popy.behavior_data_tools import *


#%% Creating population activity

sessions = pd.read_csv('/data/recordings_summary.csv', dtype={'session': str})
all_sessions = sessions.loc[((sessions.behav_valid == True) &
                         (sessions['LPFC_spikes_sorted?'] == True) &
                         (sessions['MCC_spikes_sorted?'] == True) &
                         (sessions.monkey == 'ka')), 'session']

area = "both"
condition = 'phase'
monkey = 'ka'

all_scores = {}
for session in all_sessions:
    if len(session)==5: session = '0'+session
    print(session)
    try:
        try:
            session_data = pd.read_pickle(f'/Users/zsombi/OneDrive/PoPy/tmep/session_temp_{monkey}{session}_full.pickle')
            print(session_data.both_rates[0].shape)
        except:
            behav = PoPy(monkey, session).get_data('behaviour')
            behav = add_before_switch_info(behav)
            behav = add_phase_info(behav)

            spikes = PoPy(monkey, session).get_data('spikes')
            session_data = join_session_info(behav, spikes)

            area = "MCC"
            print(session_data.MCC_spikes[0].shape)
            session_data = remove_low_fr_units(session_data, f"{area}_spikes", limit=.0)
            session_data = add_firing_rates(session_data, f"{area}_spikes", method="gauss", std=.050)
            session_data = remove_missing_neurons(session_data, f"{area}_rates")
            print(session_data.MCC_rates[0].shape)
            session_data = time_normalize_trials(session_data, f"{area}_rates")
            print(session_data.MCC_rates[0].shape)

            area = "LPFC"
            print(session_data.LPFC_spikes[0].shape)
            session_data = remove_low_fr_units(session_data, f"{area}_spikes", limit=.0)
            session_data = add_firing_rates(session_data, f"{area}_spikes", method="gauss", std=.050)
            session_data = remove_missing_neurons(session_data, f"{area}_rates")
            print(session_data.LPFC_rates[0].shape)
            session_data = time_normalize_trials(session_data, f"{area}_rates")
            print(session_data.MCC_rates[0].shape)

            session_data = merge_areas(session_data)
            session_data = join_bins(session_data, "both_rates", 3)

            session_data.to_pickle(f'/Users/zsombi/OneDrive/PoPy/tmep/session_temp_{monkey}{session}_full.pickle')
            print(session_data.both_rates[0].shape)

        session_data_binary_phase = session_data.copy()
        session_data_binary_phase['phase'] = session_data_binary_phase['phase'].replace('transition', np.nan)

        X, y = build_dataset(session_data_binary_phase, f'{area}_rates', condition, trials_before=0, trials_after=0)

        # creating data matrix and labels

        # balance conditions
        X_balanced, y_balanced = balance_conditions(X, y)

        # classification parameters
        window_len = 10
        K_fold = 5
        # classification
        scores = classification_in_time(X_balanced, y_balanced, window_len=window_len, K_fold=K_fold)
        '''baseline_scores = classification_in_time(X_balanced, y_balanced[np.random.choice(len(y_balanced), size=len(y_balanced), replace=False)],
                                                 window_len=window_len, K_fold=K_fold)'''

        all_scores[session] = scores

    except:
        continue
        # Create firing rate dataset

#%% plotting
fig, ax = plt.subplots(figsize=(10, 6))
#plt.suptitle(f"{monkey}{session} {area} area, condition: {condition}, window: {window_len*session_data.LPFC_rates[0].bin_size.data}s, #units: {X_balanced.shape[1]}")
plt.suptitle(f"#sessions: {len(all_scores)}, area: {area}, condition: {condition}, window: {window_len*session_data.LPFC_rates[0].bin_size.data}")
#plt.suptitle(f"{measurement}")

scores = np.mean(np.stack([x for x in all_scores.values()]), axis=0)
num_trials = 1
T = len(scores)
t_max = session_data.both_rates[0].bin_size * (len(session_data.both_rates[0].time) - window_len)
t_max = t_max*num_trials
t_scores = np.linspace(0, t_max, T)
ax.plot(t_scores, scores, label='accuracy')

#ax.title.set_text(f"window_len: {window_len} ({window_len*20}ms), K-fold: {K_fold}, #samples: {len(X_balanced)*window_len}, #units: {X_balanced.shape[1]}")
ax.set_ylim([-0.01, 1.1])
ax.set(xlabel='time (s)', ylabel='accuracy %')
ax.hlines(.5, 0, t_max, linestyle='dashed', color='black', label='chance level%')
#ax.hlines(np.array(baseline_scores).mean()+2*np.array(baseline_scores).std(), 0, t_max, color='grey', label='+-2 std')
#ax.hlines(np.array(baseline_scores).mean()-2*np.array(baseline_scores).std(), 0, t_max, color='grey')
#ax.grid()

def draw_keypoints(session_data, ax):
    # ax.set_xticks(np.arange(0, T - window_len, 50), np.array(time[::50], dtype=int))

    keypoint_bins = get_average_epoch_length(session_data)
    bin_size = session_data[f'both_rates'][0].bin_size
    keypoint_bins = (keypoint_bins / bin_size.data).astype('int')  # epoch lengths in number of bins

    keypoint_bins = np.cumsum(keypoint_bins) - 1
    keypoint_bins = np.insert(keypoint_bins, 0, 0, axis=0) * session_data.both_rates[0].bin_size.data

    keypoint_events = list(session_data[f"MCC_rates"][0].attrs['behav_info'].keys())
    markers = ["^", "s", "P", "X", "*", "o", "^"]

    for trial_no in range(num_trials):
        for j, time in enumerate(keypoint_bins):
            offset = trial_no * keypoint_bins[-1]
            ax.axvline(offset + time, c='tab:grey')
            ax.scatter(offset + time, .1, marker=markers[j], s=50, label=keypoint_events[j])

draw_keypoints(session_data, ax)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
#plt.savefig(f'C:/Users/eproc/OneDrive/PoPy/tmep/figs/session_temp_{session}_{condition}.png')
plt.show()
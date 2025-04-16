from popy.decoding.population_decoders import *
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *


# get data
monkey, session = 'ka', '210322'
area = "both"
condition = 'before_switch'

session_data = pd.read_pickle(f'/Users/zsombi/OneDrive/PoPy/data/processed/session_data_{monkey}{session}.pickle')

# add double feedback label
session_data = add_double_feedback(session_data)
session_data = add_phase_info(session_data, threshold=20)
session_data = add_target_value(session_data, num_trials=5)

session_data_binary_phase = session_data.copy()
session_data_binary_phase['phase'] = session_data_binary_phase['phase'].replace('transition', np.nan)

#%% double feedback analysis
condition_temp = 'double_feedback'
trials_before, trials_after = 1, 1
X_double, y_double = build_dataset(session_data, f'both_rates', condition_temp,
                                   trials_before=trials_before, trials_after=trials_after)
keypoints = get_average_epoch_length(session_data, return_ep_names=True)
ep_lens = np.cumsum(np.array([0] + [val for val in keypoints.values()]))
time_fb = ep_lens[np.argwhere(np.array(list(keypoints.keys())) == 'feedback')][0][0]
bin_size = session_data.both_rates[0].bin_size.data

N, T = X_double.shape[1], X_double.shape[2]/3

fb_ids = [int(time_fb // bin_size),
          int(T+time_fb // bin_size),
          int(2*T+time_fb // bin_size)]

# subtract 'trial n-1'from 'trial n'
X_subtracted = []
for X_trial in X_double:
    X_0 = X_trial[:, fb_ids[0]:fb_ids[1]]
    X_1 = X_trial[:, fb_ids[1]:fb_ids[2]]
    X_subtracted.append(np.subtract(X_0, X_1))
X_subtracted = np.array(X_subtracted)

# balance conditions
X, y = balance_conditions(X_subtracted, y_double)

# classification parameters
window_len = 10
K_fold = 5
# classification
scores = classification_in_time(X, y, window_len=window_len, K_fold=K_fold)

t = np.linspace(0, 7.5, 250)
fig, ax = plt.subplots()
ax.plot(t[:len(scores)], scores)
#draw_keypoints(session_data, 1, ax)
plt.show()

# avg over conditions
X_pp = np.mean(X_subtracted[y_double==True], axis=0)
X_nn = np.mean(X_subtracted[y_double==False], axis=0)

fig, axs = plt.subplots(N, 1, figsize=(10, 50))
unit_names = session_data.both_rates[0].unit.data

for i, ax in enumerate(axs):
    ax.plot(X_pp[i])
    ax.plot(X_nn[i])
    ax.axhline(0, color='grey', alpha=.4)
    ax.set(ylabel=unit_names[i], ylim=[-.5, .5])
plt.tight_layout()
plt.show()

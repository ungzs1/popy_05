'''
Analysing the phase shift, is it progressive or step-like? what sets the phase?
'''
import matplotlib.pyplot as plt
import numpy as np

from popy.decoding.population_decoders import *
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *
import scipy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# get M measure for each trial
def get_measure(X, decoder):
    trial_measures = []
    for trial_idx in range(X.shape[0]):
        trial_measures.append(decoder.decision_function(X[trial_idx, :, t_max].reshape(1, -1)))
    return np.array(trial_measures)

# get data
monkey, session = 'ka', '210322'
area = "both"
condition = 'phase'

session_data = pd.read_pickle(f'/Users/zsombi/OneDrive/PoPy/data/processed/session_data_{monkey}{session}.pickle')

# add phase label
exploit_threshold = 6
session_data = add_phase_info(session_data, threshold=exploit_threshold)
# remove transition trials
session_data_binary_phase = session_data.copy()
session_data_binary_phase['phase'] = session_data_binary_phase['phase'].replace('transition', np.nan)

#%% get measure
trials_before, trials_after = 0, 0

# creating data matrix and labels
X_unbalanced, y_unbalanced = build_dataset(session_data_binary_phase, f'{area}_rates', condition,
                                           trials_before=trials_before, trials_after=trials_after)

# balance conditions
X, y = balance_conditions(X_unbalanced, y_unbalanced)

# classification parameters
window_len = 10
K_fold = 5
# classification
scores = classification_in_time(X, y, window_len=window_len, K_fold=K_fold)
t_max = np.argmax(scores)

# plot results
fig, ax = plt.subplots()

t = np.arange(0, (len(scores)+window_len) * .03, .03)
chance_level = 1 / len(np.unique(y))

# plot time resolved decoding accuracy
ax.plot(t[:-window_len], scores, label='decoding accuracy')
# plot time of max decodability
ax.axvline(t[t_max], linestyle='dashed', color='tab:red', alpha=.5, label='time of best decoding')
# plot chance level
ax.axhline(chance_level, linestyle='dashed', color='black', alpha=.5, label='chance level')
# draw vertical lines for behavioral events
draw_keypoints(session_data, num_trials=1 + trials_before + trials_after, ax=ax)

ax.set(xlabel='time (s)', ylabel='accuracy (%)', ylim=[.2, 1])
ax.title.set_text(f'monkey: {monkey}, session: {session}\ndecoding of: {condition}')
ax.legend()
plt.show()

# Train decoder at t_max data
X_temp = np.concatenate([X[:, :, t_max + w] for w in range(window_len)], axis=0)
y_temp = np.concatenate([y for w in range(window_len)], axis=0)

clf = LogisticRegression()
decoder = clf.fit(X_temp, y_temp)

# all trial data
X_full, _ = build_dataset(session_data, f'{area}_rates', condition=None, trials_before=0, trials_after=0, reorder=False)
# M measure for trials
M_trials = get_measure(X_full, decoder)
session_data['measure'] = M_trials

#%% mapping measure to behaviour
background = 'measure'
show_target_selection(session_data, monkey, session,
                      background_value=background)

#%% correlation

fig, ax = plt.subplots()

session_data = add_reward_rate(session_data, num_trials=5, weighting='exponential', exp_decay_rate=100)

# ids where the reward rate is not nan
valid_measure_ids = np.argwhere(session_data.reward_rate.to_numpy() == session_data.reward_rate.to_numpy()).squeeze()
"""transition_ids = np.argwhere(session_data.phase.to_numpy() == 'transition').squeeze()
# elements present both in valid_measure_ids and transition_ids
transition_ids = np.intersect1d(valid_measure_ids, transition_ids)"""

# correlation between reward rate and M measure
measure = M_trials.squeeze()
measure = measure[valid_measure_ids]

reward_rate = session_data.reward_rate[valid_measure_ids].to_numpy()

# cross correlation between measure and reward rate
corr = scipy.stats.pearsonr(measure, reward_rate)
print(f'correlation between measure and reward rate: {np.round(corr[0], 5)}'
      f' with p-value: {np.round(corr[1], 5)}')

# plot measure vs reward rate
ax.scatter(measure, reward_rate)
ax.axvline(0, color='grey', linestyle='dashed', alpha=.3, label='decision axis')
ax.set(xlabel='measure', ylabel='reward rate',
        title=f'monkey: {monkey}, session: {session}\ncorrelation: {np.round(corr[0], 2)}, p-value: {np.round(corr[1], 5)}')
ax.legend()

plt.tight_layout()
plt.show()

#%% 1d projection, time resolved
phases = session_data.phase.to_numpy()
feedbacks = session_data.feedback.to_numpy()
#labels = np.unique(phases_full)
labels = ['search', 'transition', 'repeat']

# plotting
fig, ax = plt.subplots(figsize=(40, 7))
c_map = {'search': 'tab:red',
         'transition': 'darkorange',
         'repeat': 'tab:purple'}

# scatter points and lines
# add legend, red is 'search', orange is 'transition', purple is 'repeat'
# dot is correct, x is incorrect
for label in labels:
    label_ids = np.argwhere(phases == label).squeeze()
    feedbacks_label = feedbacks[label_ids]
    correct_ids = np.argwhere(feedbacks_label == 1).squeeze()
    incorrect_ids = np.argwhere(feedbacks_label == 0).squeeze()
    ax.scatter(label_ids[correct_ids], M_trials[label_ids[correct_ids]], s=80, color=c_map[label], marker='o')
    ax.scatter(label_ids[incorrect_ids], M_trials[label_ids[incorrect_ids]], s=80, color=c_map[label], marker='x')
ax.plot(M_trials, alpha=.3)

"""for i, trial in enumerate(M_trials):
    phase = phases[i]
    fb = feedbacks[i]
    if fb: marker='o'
    else: marker='x'
    ax.scatter(i, trial, marker=marker, s=80, c=c_map[phase])
"""

# labels, ticks, etc
ax.axhline(0, linestyle='dashed', color='black', label='decision boundary')
ax.set(xlabel='trials', ylabel='projection to decision axis')

# count label accuracies
accs = pd.DataFrame(columns=labels, index=['+', '-'])
for label in labels:
    phase_M = M_trials[phases==label]
    accs[label]['-'] = np.round(np.count_nonzero(phase_M < 0) / len(phase_M), 2)
    accs[label]['+'] = np.round(np.count_nonzero(0 < phase_M) / len(phase_M), 2)

# manually add label; dot is correct, x is incorrect
handles, labels = plt.gca().get_legend_handles_labels()
unrew_patch = Line2D([0], [0], marker='x', color='black', label='unrewarded')
rew_patch = Line2D([0], [0], marker='o', color='black', label='rewarded')
patch_search = Patch(color='red', label='search')
patch_trans = Patch(color='darkorange', label='transition')
patch_repeat = Patch(color='purple', label='repeat')

handles.extend([rew_patch,unrew_patch, patch_search, patch_trans, patch_repeat])

ax.legend(handles=handles)
plt.suptitle(f"monkey: {monkey} session: {session}\n{accs.to_string()}")
plt.tight_layout()
plt.show()


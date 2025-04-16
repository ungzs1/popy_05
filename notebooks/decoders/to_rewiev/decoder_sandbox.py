from popy.decoding.population_decoders import *
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *

monkey, session = 'ka', '210322'
area = "both"
condition = 'before_switch'

session_data = pd.read_pickle(f'/Users/zsombi/OneDrive/PoPy/data/processed/session_data_{monkey}{session}.pickle')
# add reward rate
session_data = add_reward_rate(session_data, num_trials=5, weighting='exponential')

#%% distribution of reward rates
rr = session_data.reward_rate
# plot distribution of reward rates
fig, ax = plt.subplots()
ax.hist(rr, bins=30)
ax.set(xlabel='reward rate', ylabel='count', title='reward rate')
plt.show()


#%% Conclusion 1: different cognitive context has different corresponding dynamics

# df to store results
all_scores = pd.DataFrame(columns=['reward_rate'],
                          index=['MCC', 'LPFC', 'both'])
# set number of trials to use for decoding (before and after the given trial)
trials_before, trials_after = 1, 1

# run decoding for each condition and store results in df
for condition in all_scores.columns:
    all_scores[condition] = [
        run_decoder(session_data, area, condition, trials_before, trials_after)
        for area in all_scores.index
    ]

### plot time resolved decoding accuracy
win=10

fig, axs = plt.subplots(len(all_scores.index), 1, figsize=(10, 15))
colors = {'MCC':'black', 'LPFC':'tab:blue', 'both':'tab:green'}
alphas = {'MCC': 1, 'LPFC': 1, 'both': .3}
baselines = {'feedback': .5, 'phase': .5, 'before_switch': .5}

bin_size = float(session_data.both_rates[0].bin_size.data)

for ax, condition in zip(axs, all_scores.columns):
    for area in all_scores.index:
        scores = all_scores[condition][area]
        t = np.arange(0, (len(scores)+win) * bin_size, bin_size)
        t_max = np.argmax(scores)

        ax.plot(t[:len(scores)], scores, label=area, color=colors[area], alpha=alphas[area])
    ax.axhline(baselines[condition], color='grey', alpha=.5, linestyle='dashed', label='chance level')
    draw_keypoints(session_data, num_trials=1+trials_before+trials_after, ax=ax)
    ax.set(xlabel='time (s)', ylabel='accuracy (%)', ylim=[.2, 1])
    ax.title.set_text(condition)
    ax.legend()
plt.suptitle(f'{monkey}{session}, \n#MCC units: {len(session_data.MCC_rates[0].unit)}, #LPFC units: {len(session_data.LPFC_rates[0].unit)}')
plt.tight_layout()
plt.show()

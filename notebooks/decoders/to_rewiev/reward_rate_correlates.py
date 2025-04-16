#%%
import sys
sys.path.append("C:\ZSOMBI\OneDrive\PoPy")

from popy.decoding.population_decoders import *
from popy.behavior_data_tools import *
from popy.plotting.plotting_tools import *

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.patches as mpatches

monkey, session = 'ka', '210322'
area = "both"
condition = 'before_switch'

session_data = pd.read_pickle(f'C:\ZSOMBI\OneDrive\PoPy\data/processed/session_data_{monkey}{session}.pickle')

# add reward rate
session_data = add_reward_rate(session_data, num_trials=5, weighting='exponential')

# %% time resolved correlation between reward rate and neural activity
def time_resolved_reward_rate_correlation(session_data, monkey, session, show=True):
    corrs = []
    weights = []

    # rime resolved correlation between reward rate and neural activity
    for t in range(len(session_data.both_rates[0].time)):
        print(t)
        # vector of reward rates
        y = session_data.reward_rate.values

        # neural activity matrix in numpy, shape: (trials, neurons, time)
        X = []
        for i in range(len(session_data)):
            X.append(session_data.both_rates[i][:, t].data)
        X = np.array(X)

        # remove nen values from y and corresponding X
        X = X[~np.isnan(y)]
        y = y[~np.isnan(y)]

        # linear regression  
        clf = LinearRegression().fit(X, y)

        # project all points to legression line, plot by coloring by reward rate
        projection_vector = X @ clf.coef_.T + clf.intercept_

        # calculate pearson correlation coefficient
        corrs.append(pearsonr(projection_vector, y)[0])
        weights.append(clf.coef_.T)

    corrs = np.array(corrs)
    weights = np.array(weights)

    # plot corrs
    if show:
        bin_size = float(session_data.both_rates[0].bin_size.data)
        t = np.arange(0, (len(corrs)) * bin_size, bin_size)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(t, corrs)
        draw_keypoints(session_data, ax=ax)
        plt.suptitle(f'monkey {monkey}, session {session}, \nreward rate correlates with neural activity')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Pearson correlation coefficient')
        plt.show()

    return corrs, weights

corrs, weights = time_resolved_reward_rate_correlation(session_data, monkey, session, show=True)

#%% correlation between reward rate and neural activity AT A GIVEN TIME POINT
t = np.argmax(corrs) # time

# vector of reward rates
y = session_data.reward_rate.values

# neural activity matrix in numpy, shape: (trials, neurons, time)
X = []
for i in range(len(session_data)):
    X.append(session_data.both_rates[i][:, t].data)
X = np.array(X)

# remove nen values from y and corresponding X
X = X[~np.isnan(y)]
y = y[~np.isnan(y)]

# linear regression  
clf = LinearRegression().fit(X, y)

# project all points to legression line, plot by coloring by reward rate
projection_vector = X @ clf.coef_.T + clf.intercept_

# plotting
plt.scatter(projection_vector, y)

# calculate pearson correlation coefficient
corr  = pearsonr(projection_vector, y)

bin_size = float(session_data.both_rates[0].bin_size.data)
time_vector = np.arange(0, (len(corrs)) * bin_size, bin_size)
plt.title(f'monkey {monkey}, session {session}, time {np.round(time_vector[t], 1)} s, \nPearson correlation coefficient: ' + str(round(corr[0], 2)))
plt.xlabel('projection to linear regression line')
plt.ylabel('reward rate')
plt.xlim(0, 1)
plt.show()

# %% which neurons are most correlated with reward rate?

t = np.argmax(corrs) # time
weights = np.abs(weights)
curr_weights = weights[t, :]

# df of area channel cell and weight
units = session_data.both_rates[0].unit.values
df = pd.DataFrame({'area': [unit[:-4] for unit in units], 
                   'channel': [int(unit[-4:-2]) for unit in units], 
                   'cell': [int(unit[-2:]) for unit in units],
                   'weight': curr_weights})

# plot 
fig, ax = plt.subplots(figsize=(10, 5))
# barplot vertically
for index, unit in df.iterrows():
    if unit.area == 'LPFC':
        color = 'tab:blue'
        ax.barh(unit.channel, unit.weight, color=color, alpha=0.5)
    elif unit.area == 'MCC':
        weight = -unit.weight
        color = 'black'
        ax.barh(unit.channel, weight, color=color, alpha=0.5)

# gray line at 0
ax.axvline(0, color='gray', linestyle='--')

# set ticks from 1 to 16 in y axis, decreasing
ax.set_yticks(np.arange(0, 18))
ax.set_yticklabels(np.arange(17, -1, -1))
#hide labels 0 and 17 in y axis
ax.get_yticklabels()[0].set_visible(False)
ax.get_yticklabels()[-1].set_visible(False)

# set xlim so that 0 in in the center
ax.set_xlim(-1.1*np.max(curr_weights), 1.1*np.max(curr_weights))

# set legend, balck for MCC, blue for LPFC  
black_patch = mpatches.Patch(color='black', label='MCC')
blue_patch = mpatches.Patch(color='tab:blue', label='LPFC')
plt.legend(handles=[black_patch, blue_patch])

plt.xlabel('absolute weight of linear regression coefficient')
plt.ylabel('channel')

bin_size = float(session_data.both_rates[0].bin_size.data)
time_vector = np.arange(0, (len(corrs)) * bin_size, bin_size)
plt.title(f'monkey {monkey}, session {session}, time {np.round(time_vector[t], 2)} s, \nwhich neurons are most correlated with reward rate?')
plt.grid()

plt.show()


# %%

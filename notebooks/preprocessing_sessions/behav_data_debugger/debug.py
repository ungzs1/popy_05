#%%
import sys
sys.path.append("C:\ZSOMBI\OneDrive\PoPy")

import matplotlib.pyplot as plt

from popy.io_tools import get_data
from popy.neural_data_tools import *

def plotting(session_data, trials, signal_name):
    test_trials = get_trial_data(session_data, signal_name, trials)
    test_concat = concat_trials(test_trials)

    fig, ax = plt.subplots(figsize=(7, 4))
    plt.suptitle(signal_name)

    im = ax.imshow(test_concat, aspect='auto')
    units = test_concat.unit.data
    time = test_concat.time_in_session.data
    #time_sec = test_concat.time_in_session[-1] - test_concat.time_in_session[0]
    bin_size = test_concat.time_in_session[1] - test_concat.time_in_session[0]
    time_conv = int(1/bin_size)
    ax.set_yticks(np.arange(len(units)), units)
    ax.set_xticks(np.arange(0, len(time), time_conv), np.round(time[::time_conv], 1))
    ax.set(xlabel='time', ylabel='unit_id')
    #plot_behav_times(ax, test_concat)

    plt.colorbar(im)
    plt.show()
    
#%%
# example session
monkey, session = 'ka', '210322'

# get behavior
behav_data = get_data(monkey, session, 'behaviour')

# get spikes
spike_data = get_data(monkey, session, 'spikes')

# join behavior and spikes
#session_data = join_session_info(behav_data, spike_data)

#%%

# raster plot of t=10 to t=20
t0, t1 = 0, 10
data = spike_data.sel(time=slice(t0, t1))

fig, ax = plt.subplots(figsize=(7, 4))
plt.suptitle('Raster plot')

im = ax.imshow(data, aspect='auto')
units = data.unit.data
time = data.time.data
time_sec = spike_data.time[-1] - spike_data.time[0]
bin_size = data.time[1] - data.time[0]
time_conv = int(1/bin_size)
ax.set_yticks(np.arange(len(units)), units)
ax.set_xticks(np.arange(0, len(time), time_conv), np.round(time[::time_conv], 1))
ax.set(xlabel='time', ylabel='unit_id')
#plot_behav_times(ax, test_concat)

# make yticks smaller
ax.tick_params(axis='y', which='major', labelsize=5)


plt.colorbar(im)
plt.show()

# %%
# Generating spike trains with gaussian smoothing
firing_rates = add_firing_rates(spike_data, method="gauss", std=.050)

#%%
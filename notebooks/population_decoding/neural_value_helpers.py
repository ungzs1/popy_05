##% title Imports
import os
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, Lasso
from scipy import stats
import matplotlib.colors as mcolors
from scipy import ndimage

from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.decoder_tools import *
from popy.plotting.plotting_tools import *
#from popy.plotting.plot_behavior import show_target_selection 
import popy.config as cfg
from popy.plotting.plot_cortical_grid import plot_on_cortical_grid

PATH = cfg.PROJECT_PATH_LOCAL

##% data loading
def load_data_custom(monkey, session, area=None, subregion=None, n_extra_trials=(0, 1)):
    # load data (meta session)
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)
    behav = add_stay_value(behav)
    behav = add_history_of_feedback(behav, num_trials=8, one_column=False, add_history_of_targets=False)
    behav['fb_sequence'] = [r3 + 2*r2 + 4*r1 for (r1, r2, r3) in zip(behav['R_1'], behav['R_2'], behav['R_3'])]
    behav = behav.dropna()
    
    neural_data = load_neural_data(monkey, session, hz=1000)
    neural_data = remove_low_fr_neurons(neural_data, 1, print_usr_msg=False)
    neural_data = remove_trunctuated_neurons(neural_data, mode='remove', delay_limit=10, print_usr_msg=False)
    neural_data = add_firing_rates(neural_data, drop_spike_trains=True, method='gauss', std=.05)
    neural_data = downsample_time(neural_data, 100)
    neural_data = scale_neural_data(neural_data)

    # 3. build neural dataset and merge with behavior
    neural_data = time_normalize_session(neural_data)
    neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=n_extra_trials)
    neural_dataset = merge_behavior(neural_dataset, behav)

    return neural_dataset


# Sample any number of colors
def sample_colors(n_colors):
    extended_colors = [
        '#4A6B9A',  # Deeper blue (12.5% position)
        '#779ECC',  # Your original dark pastel blue (25% position)
        '#9FC0DE',  # Pale Cerulean (37.5% position)
        '#F2C894',  # Peach-Orange (50% position)
        '#FFB347',  # Pastel Orange (62.5% position)
        '#FF985A',  # Atomic Tangerine (75% position)
        '#E8642A'   # Deeper orange-red (87.5% position)
    ][::-1]

    # Create continuous colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("extended_custom", extended_colors, N=256)
    return [cmap(i / (n_colors - 1)) for i in range(n_colors)]



# @title Helper Functions


def get_weights_per_area(weights, t=3.5):
    '''
    Get the weights of the PCA components per area
    '''

    # to df, where columns are monkey, session, area, unit, weight
    data_to_df = []
    for unit_temp in weights.unit.values:
        data_to_df.append(
            {'monkey': unit_temp.split('_')[0],
            'session': unit_temp.split('_')[1],
            'area': unit_temp.split('_')[2],
            'unit': "_".join(unit_temp.split('_')[3:]),
            'weight': weights.sel(time=t, method='nearest').sel(unit=unit_temp).values,
            }
        )
    df = pd.DataFrame(data_to_df)

    df['weight_ratio'] =np.abs(df['weight'])/np.sum(np.abs(df['weight']))

    # get best weight per monkey, session, area (drop unit)
    df_abs = df.copy()
    df_abs['weight'] = df_abs['weight'].abs()

    df_best = df_abs.groupby(['monkey', 'session', 'area']).agg({'weight': 'max'}).reset_index()
    df_best = df_best.rename(columns={'weight': 'best_weight'})
    df_best['best_weight_ratio'] = (df_best['best_weight'] / df_abs['weight'].sum()) * 100

    df_sum = df_abs.groupby(['monkey', 'session', 'area']).agg({'weight': 'sum'}).reset_index()
    df_sum = df_sum.rename(columns={'weight': 'sum_weight'})
    df_sum['sum_weight_ratio'] = (df_sum['sum_weight'] / df_abs['weight'].sum()) * 100
    df_best = df_best.merge(df_sum, on=['monkey', 'session', 'area'], how='left')

    df_mean = df_abs.groupby(['monkey', 'session', 'area']).agg({'weight': 'mean'}).reset_index()
    df_mean = df_mean.rename(columns={'weight': 'mean_weight'})
    df_best = df_best.merge(df_mean, on=['monkey', 'session', 'area'], how='left')

    return df_best


def time_resolved_decoder(neural_dataset, target='R_1', group=None, t_project=None):
    trial_ids = neural_dataset.trial_id.values
    labels = neural_dataset[target].values  
    cv = 10

    # create train and test splits
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    clf = LogisticRegression()

    coeffs_all = []
    scores_all = []
    projected_data = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(trial_ids, labels)):
        # get labels of this fold
        y_train = neural_dataset[target].sel(trial_id=trial_ids[train_idx]).values
        
        ## 1. Fit and score the model at each time point

        '''y_test = neural_dataset[target].sel(trial_id=trial_ids[test_idx]).values
        coeffs_fold = []
        scores_fold = []
        for t in neural_dataset.time.values:
            # get the firing rates for the current time point
            X_train_temp = neural_dataset.firing_rates.sel(trial_id=trial_ids[train_idx], time=t).values
            X_test_temp = neural_dataset.firing_rates.sel(trial_id=trial_ids[test_idx], time=t).values

            # fit the model
            clf.fit(X_train_temp, y_train)
            score = clf.score(X_test_temp, y_test)

            # Store coefficients and R² score
            coeffs_fold.append(clf.coef_)
            scores_fold.append(score)

        coeffs_all.append(coeffs_fold)
        scores_all.append(scores_fold)'''


        ## 2. project test data to decision boundary at t=3.5
        neural_dataset_train = neural_dataset.sel(trial_id=trial_ids[train_idx])
        neural_dataset_test = neural_dataset.sel(trial_id=trial_ids[test_idx])

        # fit the model for decision boundary
        X_train_temp = neural_dataset_train.firing_rates.sel(time=t_project, method='nearest').values
        clf.fit(X_train_temp, y_train)

        # project test data
        X_test_temp = neural_dataset_test.firing_rates.values
        data_projected = np.array([X_test_temp[trial, :, :].T @ clf.coef_.squeeze() for trial in range(X_test_temp.shape[0])])


        ## 3. write back to xarray, preserve the trial and time dimensions and corresponding coordinates
        time_coords = {name: coord for name, coord in neural_dataset_test.coords.items() if 'time' in coord.dims}
        trial_coords = {name: coord for name, coord in neural_dataset_test.coords.items() if 'trial_id' in coord.dims}
        # Create a DataArray with the projected data
        data_projected_da = xr.DataArray(data_projected, dims=('trial_id', 'time'), coords={**trial_coords, **time_coords})
        # add to list
        projected_data.append(data_projected_da.copy())

    '''# convert to numpy array
    coeffs_all = np.array(coeffs_all).squeeze()
    scores_all = np.array(scores_all).squeeze()

    # average over folds
    coeffs_all = np.mean(coeffs_all, axis=0)
    scores_all = np.mean(scores_all, axis=0)

    ## 4. evalueate fit at time of interest
    data_at_time = neural_dataset.firing_rates.sel(time=t_project, method='nearest')
    X, y = data_at_time.values, data_at_time[target].values
    score_perm, permutation_scores, pvalue = permutation_test_score(clf, X, y, n_permutations=100, n_jobs=-1)
    print(f"Sanity: score of perm at time {t_project} is {score_perm}")

    # create an xarray
    results = xr.DataArray(scores_all, dims=('time'), coords={'time': neural_dataset.time.values})
    weights = xr.DataArray(coeffs_all, dims=('time', 'unit'), coords={'time': neural_dataset.time.values, 'unit': neural_dataset.unit.values})
    '''
    # concatenate projected data
    projected_data = xr.concat(projected_data, dim='trial_id')
    projected_data = projected_data.sortby('trial_id')

    '''return results, weights, projected_data, pvalue'''
    return projected_data

from imblearn.under_sampling import RandomUnderSampler

def time_resolved_decoder_all_time(neural_dataset, target='R_1', group=None):
    trial_ids = neural_dataset.trial_id.values
    labels = neural_dataset[target].values  
    cv = 10

    # create train and test splits
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    clf = LogisticRegression()
    rus = RandomUnderSampler(random_state=42)


    coeffs_all = []  # (n_units, timebins, folds)
    scores_all = []
    projected_data = []  # (n_trials_per_test * folds, timebins)
    for fold, (train_idx, test_idx) in enumerate(skf.split(trial_ids, labels)):
        neural_dataset_train = neural_dataset.sel(trial_id=trial_ids[train_idx])
        neural_dataset_test = neural_dataset.sel(trial_id=trial_ids[test_idx])

        # get labels of this fold
        y_train = neural_dataset_train[target].values
        y_test = neural_dataset_test[target].values

        ## 1. Fit and score the model at each time point

        coeffs_fold = []  # (n_units, timebins)
        scores_fold = []
        projections_fold = []
        for t in neural_dataset.time.values:
            # get the firing rates for the current time point
            X_train_temp = neural_dataset_train.firing_rates.sel(time=t).values
            X_test_temp = neural_dataset_test.firing_rates.sel(time=t).values

            # balance dataset
            X_train_temp, y_train_temp = rus.fit_resample(X_train_temp, y_train)

            # fit the model on train data
            clf.fit(X_train_temp, y_train_temp)

            # evaluate on test set
            score = clf.score(X_test_temp, y_test)

            # project test set 
            data_projected = clf.intercept_ + X_test_temp@ clf.coef_.squeeze()

            # Store coefficients and R² score
            coeffs_fold.append(clf.coef_.squeeze()) 
            scores_fold.append(score)
            projections_fold.append(data_projected)


        coeffs_all.append(coeffs_fold)  # (n_units, )
        scores_all.append(scores_fold)

        # Create a DataArray with the projected data (to preserve trials)
        projections_fold = np.array(projections_fold)
        time_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'time' in coord.dims}
        trial_coords = {name: coord for name, coord in neural_dataset_test.coords.items() if 'trial_id' in coord.dims}
        projections_fold = xr.DataArray(projections_fold.T, dims=('trial_id', 'time'), coords={**trial_coords, **time_coords})
        
        # add to list
        projected_data.append(projections_fold)

    projected_data = xr.concat(projected_data, dim='trial_id')
    projected_data = projected_data.sortby('trial_id')

    # convert to numpy array
    coeffs_all = np.array(coeffs_all).squeeze()
    scores_all = np.array(scores_all)
    # average over folds
    coeffs_all = np.mean(coeffs_all, axis=0)
    scores_all = np.mean(scores_all, axis=0)

    # create an xarray
    projected_data = projected_data.assign_coords(decodability=('time', scores_all))
    weights = xr.DataArray(coeffs_all, dims=('time', 'unit'), coords={'time': neural_dataset.time.values, 'unit': neural_dataset.unit.values})
    
    return weights, projected_data


def add_neural_value_coord(data_projected, t_interest=[2.5, 3.5]):
    trial_ids = data_projected.trial_id.values
    V_ts = []
    V_t_p1s = []
    dVs = []
    for trial_id in trial_ids:
        V_t = data_projected.sel(trial_id=trial_id, time=slice(t_interest[0], t_interest[1])).mean('time').values
        # if next trial is part of the same session, get the next time point
        if trial_id + 1 in data_projected.trial_id.values:
            V_t_p1 = data_projected.sel(trial_id=trial_id + 1, time=slice(t_interest[0], t_interest[1])).mean('time').values
            dV = V_t_p1 - V_t
        else:
            # if next trial is not part of the same session, set dV to NaN
            V_t_p1 = np.nan
            dV = np.nan

        V_ts.append(V_t)
        V_t_p1s.append(V_t_p1)
        dVs.append(dV)
    # convert to numpy arrays
    V_ts = np.array(V_ts)   
    V_t_p1s = np.array(V_t_p1s)
    dVs = np.array(dVs)

    data_projected = data_projected.assign_coords(V_t=('trial_id', V_ts))
    data_projected = data_projected.assign_coords(V_t_p1=('trial_id', V_t_p1s))
    data_projected = data_projected.assign_coords(dV=('trial_id', dVs))

    return data_projected


##% Plotting Functions


def plot_decoder_results(results, weights, n_extra_trials):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))

    ax = axs[0]


    ax.plot(results.time, np.convolve(results, np.ones(5)/5, 'same'))
    ax.axhline(.5, color='k', linestyle='--')
    
    plot_keypoints(ax, n_extra_trials, fontsize=8)
    ax.grid(axis='x')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('R2')
    ax.set_xlabel('Time')
    ax.set_title('R2 score')
    # add colorbar
    ax.set_title('decoder performance (cross val)')

    ax = axs[1]
    # plot bar below
    pbar = ax.imshow(weights.data.T, aspect='auto', cmap='RdBu', extent=[weights.time.min(), weights.time.max(), 0-.5, len(weights.unit)-.5], origin='lower', vmin=-np.max(np.abs(weights.data)*.5), vmax=np.max(np.abs(weights.data)*.5))
    plt.colorbar(pbar, ax=ax, pad=0.005, fraction=.05)
    
    plot_keypoints(ax, n_extra_trials, fontsize=8)
    ax.grid(axis='x', alpha=.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time')
    ax.set_ylabel('Unit')
    ax.set_title('Weights')

    ax = axs[2]
    df_weights = get_weights_per_area(weights, t=3.5)
    plot_on_cortical_grid(df_weights, 'sum_weight_ratio', 
                            bar_title='percentage of weights given by area (%)',
                            ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig, axs


def plot_projected_data(data_projected, t_interest_value, normalize=False, n_extra_trials=(0, 1), xlim=None, paper_format=False):
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.2  # in cm
        w = 1.5
        linewidth = 0.7
    else:
        plt.rcParams.update({'font.size': 12})
        h = 4  # in cm
        w = 10  # in cm
        linewidth = 1.5

    colors = sample_colors(8)

    fig, ax = plt.subplots(figsize=(w, h))

    time_vector = data_projected.time.values

    unique_fb_sequences = np.sort(np.unique(data_projected.fb_sequence.data))

    # bwr colormap, n=8 sampples RdYlGn
    alphas = [1-i for i in np.linspace(0, 1, len(unique_fb_sequences))]
    labels = ["[-, -, -]", "[+, -, -]", "[-, +, -]", "[+, +, -]", "[-, -, +]", "[+, -, +]", "[-, +, +]", "[+, +, +]"]
        
    for i, label in enumerate(unique_fb_sequences):
        class_mean = np.mean(data_projected.where(data_projected.fb_sequence == label), axis=0)
        class_mean_smoothed = ndimage.convolve(class_mean, np.ones(50)/50, mode='nearest')
        ax.plot(time_vector, class_mean_smoothed, color=colors[i], label=labels[i], alpha=.7, linewidth=linewidth)

    ax.axvspan(t_interest_value[0], t_interest_value[1], color='grey', alpha=0.2, label='eval window')

    # add behav keyboints
    plot_keypoints(ax, n_extra_trials)
    # y grid only
    ax.grid(axis='x', alpha=.5, linewidth=linewidth*.75)
    ax.axhline(0, color='k', linestyle='-', linewidth=linewidth*.75, zorder=0)

    # remove left and top spines
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    # move legemd outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # save as svg
    ax.set_ylabel('$V_t$')
    '''if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Neural value')'''
    
    if xlim is not None:    
        ax.set_xlim(xlim)

    return fig, ax


def projection_timepoint(behav_new, paper_format=False, ylim=None, show_datapoints=True):
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.2  # in cm
        w = 2
        s = 5
    else:
        plt.rcParams.update({'font.size': 18})
        h = 3  # in cm
        w = 6  # in cm
        s = 10

    # Example: get 10 colors from your scale
    colors = sample_colors(8)
        
    df = pd.DataFrame({
        'trial_id': behav_new.trial_id.values,
        'V_t': behav_new.V_t.values,
        #'V_t_p1': data_projected.V_t_p1.values,
        'dV': behav_new.dV.values,
        'feedback': behav_new.feedback.values,
        'fb_sequence': behav_new.fb_sequence.values,
        'fb_sequence_last_2': behav_new.fb_sequence.values % 4,
    })

    label_mapping_long = {0: '0\n0\n0', 1: '0\n0\n1', 2:'0\n1\n0', 3:'0\n1\n1', 4: '1\n0\n0', 5: '1\n0\n1', 6:'1\n1\n0', 7:'1\n1\n1'}
    label_mapping_short = {0: '0\n0', 1: '0\n1', 2:'1\n0', 3:'1\n1'}
    df['fb_sequence'] = df['fb_sequence'].map(label_mapping_long)
    df['fb_sequence_last_2'] = df['fb_sequence_last_2'].map(label_mapping_short)

    print(df)

    # distribution of pos and neg feedback along neural value
    fig, axs = plt.subplots(1, 2, figsize=(2*w, h), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.1}) 

    # First subplot: boxplot (unchanged)
    ax = axs[0]

    sns.boxplot(df.sort_values('fb_sequence'), x='fb_sequence', y='V_t', palette=colors, ax=ax, showfliers=False, boxprops={'linewidth': 0.5})
    if show_datapoints:
        sns.stripplot(df.sort_values('fb_sequence'), x='fb_sequence', y='V_t', color='black', size=3, alpha=0.5, ax=ax)
    ax.axhline(0, color='k', linestyle='--', alpha=0.7, linewidth=0.75, zorder=0)

    # second subplot
    ax = axs[1]

    sns.boxplot(df.sort_values('fb_sequence_last_2'), x='fb_sequence_last_2', y='dV', hue='feedback', palette=COLORS, ax=ax, showfliers=False, boxprops={'linewidth': 0.5, 'alpha':.6})
    if show_datapoints:
        sns.stripplot(df.sort_values('fb_sequence_last_2'), x='fb_sequence_last_2', y='dV', hue='feedback', palette=['black', 'black'], size=3, alpha=0.5, ax=ax, dodge=True, legend=False)
    ax.axhline(0, color='k', linestyle='--', alpha=0.7, linewidth=0.75, zorder=0)

    ax = axs[0]
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    #ax.set_xlabel('Feedback sequence')
    ax.set_ylabel('$V_t$')
    #ax.set_yticks([0, .5, 1])   
    #ax.set_yticklabels([0, .5, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax = axs[1]
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    #ax.set_xlabel('Feedback sequence')
    ax.set_ylabel('$\Delta V$')
    '''ax.set_yticks([0, 1, 2, 3])   
    ax.set_yticklabels('''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #plt.tight_layout()
    return fig, ax


def green_red_plot(behav_new, paper_format=False, xlim=None, ylim=None):
    from matplotlib.gridspec import GridSpec

    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.3  # in cm
        w = 3.5
        s = 5
        linewidth = 0.7
    else:
        plt.rcParams.update({'font.size': 18})
        h = 6  # in cm
        w = 15  # in cm
        s = 30
        linewidth = 1.5

    # Method 1: Use GridSpec directly (recommended)
    fig = plt.figure(figsize=(w, h))
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05, 
                left=0.55, right=0.95, bottom=0.15, top=0.85)

    ax_main = fig.add_subplot(gs[0])  # main plot
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)  # Y-axis KDE (rotated)
    
    # extract datapoints
    V_t = behav_new.V_t.values
    V_t_p1 = behav_new.V_t_p1.values
    fb_vector = behav_new.feedback.values

    dV = V_t_p1 - V_t

    V_t_behav = behav_new.V_t_behav.values
    fb_sequence = behav_new.fb_sequence.values

    dV_min, dV_max = np.min(dV), np.max(dV)
    hist_bins = np.linspace(dV_min, dV_max, 20)
    
    # Scatter positive and negative feedback with fit lines
    for fb_curr in [0, 1]:
        # get pos/neg trial
        V_t_curr = V_t[fb_vector == fb_curr]
        dV_curr = dV[fb_vector == fb_curr]

        # scatter trials
        ax_main.scatter(V_t_curr, dV_curr, color=COLORS[fb_curr], alpha=.5, s=s, edgecolor='none')

        # fit and plot slopes
        slope, intercept, r_value, p_value, std_err = stats.linregress(V_t_curr, dV_curr)
        fit_line = slope * V_t_curr + intercept
        ax_main.plot(V_t_curr, fit_line, color=COLORS[fb_curr], alpha=.5)

        # Create KDE plots for each group
        '''sns.histplot(
            y=dV_curr,
            color=COLORS[fb_curr],
            alpha=0.6,
            bins=hist_bins,
            kde=True,
            stat='count',
            label='fb positive',
            fill=True,
            edgecolor=None,  # This removes the line around the bars
            ax=ax_hist
        )'''

        sns.boxplot(
            x=fb_curr,
            y=dV_curr,
            color=COLORS[fb_curr],
            #alpha=0.6,            
            ax=ax_hist,
            showfliers=False,
        )

        # mean line for each group
        #mean_dV = np.mean(dV_curr)
        #ax_hist.axhline(mean_dV, color=COLORS[fb_curr], linestyle='--', alpha=0.7, linewidth=.7, label=f'mean fb {["negative", "positive"][fb_curr]}')

    # t-stat
    t_stat, p_val = stats.ttest_ind(dV[fb_vector == 1], dV[fb_vector == 0])
    print(f't-statistic: {t_stat}, p-value: {p_val}')
    # print significance over boxplot (* if p < 0.05, ** if p < 0.01, *** if p < 0.001)
    if p_val < 0.001:
        significance = '***'
    elif p_val < 0.01:
        significance = '**'
    elif p_val < 0.05:
        significance = '*'
    else:
        significance = 'ns'
    ymax = dV.max() * 1.1
    ax_hist.text(0.5, ymax, f'{significance}', ha='center', va='bottom', color='k')
    ax_hist.plot([0, 1], [ymax, ymax], color='black', linestyle='-', linewidth=linewidth)  # horizontal line for significance

    # Reference lines
    mean_val = 0
    ax_main.axvline(mean_val, color='k', linestyle='--', alpha=0.7, linewidth=linewidth)
    ax_main.axhline(0, color='k', linestyle='-', alpha=0.7, linewidth=linewidth)

    # Add horizontal reference lines
    #ax_hist.axhline(0, color='k', linestyle='-', alpha=0.7, linewidth=.7)

    # Styling and labels
    ax_main.scatter([], [], color=COLORS[1], label='Unrewarded')
    ax_main.scatter([], [], color=COLORS[0], label='Rewarded')

    ax_main.legend(loc='upper center')

    ax_main.set_xlabel('$V_t$')
    ax_main.set_ylabel('$\Delta V = V_{t+1} - V_t$')

    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    if xlim is not None:
        ax_main.set_xlim(xlim)    
    if ylim is not None:
        ax_main.set_ylim(ylim)

    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['left'].set_visible(False)

    ax_hist.tick_params(left=False, labelleft=False)

    ax_hist.set_xticks([0, 1])
    ax_hist.set_xticklabels(['Unrewarded', 'Rewarded'], rotation=90)

    #plt.tight_layout()


    return fig, [ax_main, ax_hist]



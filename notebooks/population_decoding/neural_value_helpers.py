##% title Imports
import os
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, Lasso
from scipy import stats
import matplotlib.colors as mcolors
from scipy import ndimage
from joblib import Parallel, delayed
import statsmodels.api as sm

from matplotlib.collections import LineCollection

from collections import defaultdict


from popy.io_tools import *
from popy.behavior_data_tools import *
from popy.neural_data_tools import time_normalize_session, scale_neural_data, remove_low_fr_neurons, remove_trunctuated_neurons
from popy.decoding.decoder_tools import *
from popy.plotting.plotting_tools import *
#from popy.plotting.plot_behavior import show_target_selection 
import popy.config as cfg
from popy.config import value_gradient
from popy.plotting.plot_cortical_grid import plot_on_cortical_grid
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import axes
import statsmodels.api as sm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

PATH = cfg.PROJECT_PATH_LOCAL

##% data loading
def load_data_custom(monkey, session, area=None, subregion=None, n_extra_trials=(0, 1), sr=100):
    # load data (meta session)
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)
    behav = add_switch_info(behav)
    behav = add_stay_value(behav)
    behav = add_history_of_feedback(behav, num_trials=8, one_column=False, add_history_of_targets=False)
    behav['fb_sequence'] = [r3 + 2*r2 + 4*r1 for (r1, r2, r3) in zip(behav['R_1'], behav['R_2'], behav['R_3'])]
    behav['target_1'] = behav['target'] == 1
    behav = behav.dropna()
    
    neural_data = load_neural_data(monkey, session, hz=1000)
    neural_data = remove_low_fr_neurons(neural_data, 1, print_usr_msg=False)
    neural_data = remove_trunctuated_neurons(neural_data, mode='remove', delay_limit=10, print_usr_msg=False)
    neural_data = add_firing_rates(neural_data, drop_spike_trains=True, method='gauss', std=.05)
    neural_data = downsample_time(neural_data, sr)
    neural_data = scale_neural_data(neural_data)

    # 3. build neural dataset and merge with behavior
    neural_data = time_normalize_session(neural_data)
    neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=n_extra_trials)
    neural_dataset = merge_behavior(neural_dataset, behav)

    return neural_dataset


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


def process_fold(fold_data):
    """Process a single fold - to be run in parallel"""
    neural_dataset, fold_id, train_idx, test_idx, target, across_time = fold_data

    # Initialize models for this fold
    clf = LogisticRegression()
    rus = RandomUnderSampler(random_state=42)
    
    neural_dataset_train = neural_dataset.sel(trial_id=neural_dataset.trial_id.values[train_idx])
    neural_dataset_test = neural_dataset.sel(trial_id=neural_dataset.trial_id.values[test_idx])

    # get labels of this fold
    y_train = neural_dataset_train[target].values
    y_test = neural_dataset_test[target].values

    n_timebins = len(neural_dataset.time.values)
    scores_fold = np.full(n_timebins, np.nan)
    across_time_scores_fold = np.full((n_timebins, n_timebins), np.nan)
    projections_fold = []
    
    for train_t_id, t in enumerate(neural_dataset.time.values):
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
        data_projected = clf.intercept_ + X_test_temp @ clf.coef_.ravel()

        # Store score and projections
        scores_fold[train_t_id] = score
        projections_fold.append(data_projected)

        # Test this decoder across all time points
        if across_time:
            for test_t_id, test_t in enumerate(neural_dataset.time.values):
                X_test_all_times = neural_dataset_test.firing_rates.sel(time=test_t).values
                score_across_time = clf.score(X_test_all_times, y_test)
                across_time_scores_fold[train_t_id, test_t_id] = score_across_time

    # Create a DataArray with the projected data (to preserve trials)
    projections_fold = np.array(projections_fold)
    time_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'time' in coord.dims}
    trial_coords_fold = {name: coord for name, coord in neural_dataset_test.coords.items() if 'trial_id' in coord.dims}
    projections_fold = xr.DataArray(projections_fold.T, dims=('trial_id', 'time'), 
                                   coords={**trial_coords_fold, **time_coords})
    
    return fold_id, scores_fold, across_time_scores_fold, projections_fold


def time_resolved_decoder_all_time(neural_dataset, target='R_1', group=None, across_time=False, n_jobs=-1):
    trial_ids = neural_dataset.trial_id.values
    labels = neural_dataset[target].values  
    cv = 10

    # create train and test splits
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    n_timebins = len(neural_dataset.time.values)
    time_coords = {name: coord for name, coord in neural_dataset.coords.items() if 'time' in coord.dims}

    # Prepare data for parallel processing
    fold_data = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(trial_ids, labels)):
        fold_data.append((neural_dataset.copy('deep'), fold_id, train_idx, test_idx, target, across_time))

    # Process folds in parallel - xarray can not be pickled...
    '''print(f"Processing {cv} folds in parallel with {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_fold)(data) for data in fold_data
    )'''
    results = []
    for data in fold_data:
        results.append(process_fold(data))

    # Collect results
    scores_all = np.full((n_timebins, cv), np.nan)  # (timebins, folds)
    across_time_scores = np.full((n_timebins, n_timebins, cv), np.nan)  # (folds, timebins)
    projected_data = []

    for fold_id, scores_fold, across_time_scores_fold, projections_fold in results:
        scores_all[:, fold_id] = scores_fold
        if across_time:
            across_time_scores[:, :, fold_id] = across_time_scores_fold
        projected_data.append(projections_fold)

    # Concatenate projected data
    projected_data = xr.concat(projected_data, dim='trial_id')
    projected_data = projected_data.sortby('trial_id')

    # average over folds
    #coeffs_all = np.mean(coeffs_all, axis=0)
    scores_all = np.mean(scores_all, axis=-1)  # average over folds
    across_time_scores = np.mean(across_time_scores, axis=-1)  # average over folds

    # create an xarray
    projected_data = projected_data.assign_coords(decodability=('time', scores_all))
    
    # NEW: Create xarray for across-time decodability matrix
    decodability_matrix = xr.DataArray(
        across_time_scores,
        dims=('train_time', 'test_time'),
        coords={'train_time': neural_dataset.time.values, 'test_time': neural_dataset.time.values}
    )

    return decodability_matrix, projected_data


def add_neural_value_coord(data_projected, t_interest=[2.5, 3.5]):
    trial_ids = data_projected.trial_id.values
    V_ts = []
    V_t_p1s = []
    dVs = []

    V_ts_behav = []
    V_t_p1s_behav = []
    dVs_behav = []
    for trial_id in trial_ids:
        V_t = data_projected.sel(trial_id=trial_id, time=slice(t_interest[0], t_interest[1])).mean('time').values
        V_t_behav = data_projected.sel(trial_id=trial_id).stay_value.values

        # if next trial is part of the same session, get the next time point
        if trial_id + 1 in data_projected.trial_id.values:
            V_t_p1 = data_projected.sel(trial_id=trial_id + 1, time=slice(t_interest[0], t_interest[1])).mean('time').values
            dV = V_t_p1 - V_t

            V_t_p1_behav = data_projected.sel(trial_id=trial_id + 1).stay_value.values
            dV_behav = V_t_p1_behav - V_t_behav
        else:
            # if next trial is not part of the same session, set dV to NaN
            V_t_p1 = np.nan
            dV = np.nan

            V_t_p1_behav = np.nan
            dV_behav = np.nan


        V_ts.append(V_t)
        V_t_p1s.append(V_t_p1)
        dVs.append(dV)

        V_ts_behav.append(V_t_behav)
        V_t_p1s_behav.append(V_t_p1_behav)
        dVs_behav.append(dV_behav)

    # convert to numpy arrays
    V_ts = np.array(V_ts)   
    V_t_p1s = np.array(V_t_p1s)
    dVs = np.array(dVs)

    V_ts_behav = np.array(V_ts_behav)
    V_t_p1s_behav = np.array(V_t_p1s_behav)
    dVs_behav = np.array(dVs_behav)

    data_projected = data_projected.assign_coords(V_t=('trial_id', V_ts))
    data_projected = data_projected.assign_coords(V_t_p1=('trial_id', V_t_p1s))
    data_projected = data_projected.assign_coords(dV=('trial_id', dVs))
    
    data_projected = data_projected.assign_coords(V_t_behav=('trial_id', V_ts_behav))
    data_projected = data_projected.assign_coords(V_t_p1_behav=('trial_id', V_t_p1s_behav))
    data_projected = data_projected.assign_coords(dV_behav=('trial_id', dVs_behav))

    return data_projected



def simple_glm_weights(behav_new):
    # Data setup
    weights_all = []
    for monkey in behav_new.monkey.unique():
        for subregion in behav_new.subregion.unique():
            behav_temp = behav_new.loc[(behav_new.monkey == monkey) & (behav_new.subregion == subregion)]

            for session in behav_temp.session.unique():
                behav = behav_temp.loc[behav_temp.session == session].copy()

                y = behav.V_t.values
                X = np.array([behav[f'R_{i}'].values for i in range(1, 9)]).T
                # add one more column for the randomly shuffled values

                # Add intercept for statsmodels
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm)
                results = model.fit()

                # Extract weights and p-values (skip intercept)
                weights = results.params[1:]
                pvals = results.pvalues[1:]
                
                res_temp = {f'R_{i}': results.params[i] for i in range(1, len(weights) + 1)}
                res_temp['monkey'] = monkey
                res_temp['session'] = session
                res_temp['subregion'] = subregion
                weights_all.append(res_temp)

    weights_all = pd.DataFrame(weights_all)
    return weights_all

def single_permutation_cpd(y, X_full, i, seed=None):
    """Calculate CPD for a single permutation"""
    if seed is not None:
        np.random.seed(seed)
    
    X_perm = X_full.copy()
    X_perm[:, i] = np.random.permutation(X_perm[:, i])
    
    # Full model with permuted data
    X_perm_sm = sm.add_constant(X_perm)
    model_perm_full = sm.OLS(y, X_perm_sm).fit()
    r2_perm_full = model_perm_full.rsquared
    
    # Reduced model with permuted data
    X_perm_reduced = np.delete(X_perm, i, axis=1)
    X_perm_reduced_sm = sm.add_constant(X_perm_reduced)
    model_perm_reduced = sm.OLS(y, X_perm_reduced_sm).fit()
    r2_perm_reduced = model_perm_reduced.rsquared
    
    # Calculate CPD for this permutation
    cpd_perm = ((r2_perm_full - r2_perm_reduced) / r2_perm_full) * 100
    return cpd_perm

def process_single_session(behav, n_perms=100):
    y = behav.V_t.values
    X_full = np.array([behav[f'R_{i}'].values for i in range(1, 9)]).T
    
    # Calculate full model R²
    X_full_sm = sm.add_constant(X_full)
    model_full = sm.OLS(y, X_full_sm).fit()
    r2_full = model_full.rsquared
    
    cpd_temp = {}
    #pvals_temp = {}
    
    # Process each regressor
    for i in range(X_full.shape[1]):                
        # Calculate actual CPD
        X_reduced = np.delete(X_full, i, axis=1)
        X_reduced_sm = sm.add_constant(X_reduced)
        model_reduced = sm.OLS(y, X_reduced_sm).fit()
        r2_reduced = model_reduced.rsquared
        cpd_actual = ((r2_full - r2_reduced) / r2_full) * 100
        
        # Parallel permutation testing with different seeds for reproducibility
        seeds = np.random.randint(0, 2**31, n_perms)
        cpd_perm = []
        for n_i in range(n_perms):
            cpd_perm.append(single_permutation_cpd(y, X_full, i, seed=seeds[n_i]))

        cpd_perm = np.array(cpd_perm)
        pval = (np.sum(cpd_perm >= cpd_actual) + 1) / (n_perms + 1)
        
        key = f'R_{i+1}'
        cpd_temp[f'cpd_{key}'] = cpd_actual
        cpd_temp[f'pval_{key}'] = pval
        
    #cpd_temp.update(pvals_temp)
    cpd_temp['monkey'] = behav['monkey'].unique()[0]  # Assuming single monkey per session
    cpd_temp['session'] = behav['session'].unique()[0]  # Assuming single session per behavior
    cpd_temp['subregion'] = behav['subregion'].unique()[0]  # Assuming single subregion per behavior

    return cpd_temp

def calculate_cpds(y, X):
    """Calculate CPDs (Change in Explained Variance) for a given dependent variable y and independent variables X."""
    import statsmodels.api as sm

    # add random column to X_full
    random_col = np.random.permutation(X[:, -1])
    X_full_random = np.column_stack((X, random_col))
    # Add intercept for statsmodels
    X_full_random = sm.add_constant(X_full_random)  # Add intercept for statsmodels

    model_full = sm.OLS(y, X_full_random).fit()
    r2_full = model_full.rsquared

    # Leave out random column for reduced model, i.e. use only the original regressors
    X_reduced = X
    X_reduced = sm.add_constant(X_reduced)  # Add intercept for statsmodels
    model_reduced = sm.OLS(y, X_reduced).fit()
    r2_reduced = model_reduced.rsquared
    # Calculate change in explained variance, in percentage points
    cpd = ((r2_full - r2_reduced) / r2_full) * 100

    return cpd

def create_permutation_null(behav_temp, n_perms=1000):
    """Create null by permuting the dependent variable"""
    

    perm_cpds = []
    for perm in range(n_perms):
        
        session_cpds = []
        for session in behav_temp.session.unique():
            behav = behav_temp.loc[behav_temp.session == session].copy()
            
            # PERMUTE THE DEPENDENT VARIABLE
            y = behav.V_t.values  # This breaks real relationships
            X = np.array([behav[f'R_{i}'].values for i in range(1, 9)]).T
            
            # Calculate CPDs with permuted y
            cpd = calculate_cpds(y, X)  # your existing CPD function
            session_cpds.append(cpd)
        
        # Average across sessions for this group
        group_mean_cpds = np.mean(session_cpds, axis=0)
        perm_cpds.append(group_mean_cpds)

    return np.array(perm_cpds)

def add_corrected_pvals(cpd_all):
    from statsmodels.stats.multitest import multipletests

    # Get p-value columns
    pval_cols = [col for col in cpd_all.columns if col.startswith('pval_') and not col.endswith('corrected')]

    # Create new columns for corrected p-values
    for col in pval_cols:
        cpd_all[f'{col}_corrected'] = np.nan

    # Apply correction within each group (monkey × subregion)
    for monkey in cpd_all.monkey.unique():
        for subregion in cpd_all.subregion.unique():
            # Get mask for this group
            group_mask = (cpd_all.monkey == monkey) & (cpd_all.subregion == subregion)
            
            # Extract p-values for this group
            group_pvals = cpd_all.loc[group_mask, pval_cols].values
            
            # Apply correction within this group
            rejected, pvals_corrected, _, _ = multipletests(
                group_pvals.flatten(), method='fdr_bh'
            )
            
            # Reshape back to original shape
            pvals_corrected = pvals_corrected.reshape(group_pvals.shape)
            
            # Write back corrected p-values
            for i, col in enumerate(pval_cols):
                cpd_all.loc[group_mask, f'{col}_corrected'] = pvals_corrected[:, i]

    ''' # Now use the corrected p-values
    pval_cols_corrected = [f'{col}_corrected' for col in pval_cols]'''

    return cpd_all

def get_cpd_all(behav_new, n_perms):

    # Get all unique monkey-subregion combinations
    # Get all unique monkey-subregion-session combinations
    session_behavs = []
    for monkey in behav_new.monkey.unique():
        for subregion in behav_new.subregion.unique():
            behav_temp = behav_new.loc[(behav_new.monkey == monkey) & (behav_new.subregion == subregion)]
            for session in behav_temp.session.unique():
                behav_temp = behav_new.loc[(behav_new.monkey == monkey) & (behav_new.subregion == subregion)]
                behav = behav_temp.loc[behav_temp.session == session].copy()

                session_behavs.append(behav)

    # Process all combinations in parallel

    print(f"Using all available CPU cores for parallel processing")
    cpd_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_single_session)(behav, n_perms)
        for behav in session_behavs
    )
    '''
    cpd_results = []
    for behav in session_behavs:
        cpd = process_single_session(behav, n_perms)
        cpd_results.append(cpd)
    '''

    cpd_all = pd.DataFrame(cpd_results)
    #cols_new = ['monkey', 'subregion', 'session'] + [col for col in cpd_all.columns if col not in ['monkey', 'subregion', 'session']]
    #cpd_all = cpd_all[cols_new]

    print("Analysis complete!")

    # correct pvals per group with FDR
    cpd_all = add_corrected_pvals(cpd_all)

    # save results
    #cpd_all.to_pickle('results/cpd_all.pkl')

    return cpd_all

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


def plot_projected_data(data_projected, t_interest_value=None, normalize=False, n_extra_trials=(0, 1), xlim=None, paper_format=False, ax=None):
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.7  # in cm
        w = 2.5
        linewidth = 2
    else:
        plt.rcParams.update({'font.size': 12})
        h = 6  # in cm
        w = 8  # in cm
        linewidth = 1.5

    unique_fb_sequences = np.sort(np.unique(data_projected.fb_sequence.data))

    colors = value_gradient(len(unique_fb_sequences))

    if ax is None:
        fig, ax = plt.subplots(figsize=(w, h))

    # 500 ms for convolution
    dt = data_projected.time.values[1] - data_projected.time.values[0]
    n_convolution_bins = int(0.5 / dt)
    window = np.ones(n_convolution_bins) / n_convolution_bins

    time_vector = data_projected.time.values


    # bwr colormap, n=8 sampples RdYlGn
    alphas = [1-i for i in np.linspace(0, 1, len(unique_fb_sequences))]
    labels = ["[-, -, -]", "[+, -, -]", "[-, +, -]", "[+, +, -]", "[-, -, +]", "[+, -, +]", "[-, +, +]", "[+, +, +]"]
        
    for i, label in enumerate(unique_fb_sequences):
        class_mean = np.mean(data_projected.where(data_projected.fb_sequence == label), axis=0)
        class_mean_smoothed = ndimage.convolve(class_mean, window, mode='nearest')
        ax.plot(time_vector, class_mean_smoothed, color=colors[i], label=labels[i], linewidth=linewidth, zorder=10)

    if t_interest_value is not None:
        ax.axvspan(t_interest_value[0], t_interest_value[1], color='grey', alpha=0.2, label='eval window')

    # add behav keyboints
    plot_keypoints(ax, n_extra_trials, mark_event='Fb', xlabels='events')
    # y grid only
    ax.grid(axis='x', alpha=.5, linewidth=1, linestyle='--')
    ax.axhline(0, color='k', linestyle='-', linewidth=linewidth*.75, zorder=0)

    # remove left and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # move legend outside
    if not paper_format:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # save as svg
    ax.set_ylabel('$V_t$', fontsize=10)
    '''if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Neural value')'''
    
    if xlim is not None:    
        ax.set_xlim(xlim)

    return fig, ax


def plot_projected_data_switch(data_projected, t_interest_value=None, normalize=False, n_extra_trials=(0, 1), xlim=None, paper_format=False, ax=None):
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 2  # in cm
        w = 2.5
        linewidth = 2
    else:
        plt.rcParams.update({'font.size': 12})
        h = 6  # in cm
        w = 8  # in cm
        linewidth = 1.5

    r1s = [0, 1]
    r2s = [0, 1]
    shifts = [0, 1]  # 0: no switch, 1: switch

    colors = value_gradient(4)
    linestyles = {1: '--', 0: '-'}
    labels = {1: 'switch', 0: 'stay'}

    if ax is None:
        fig, ax = plt.subplots(figsize=(w, h))

    # 500 ms for convolution
    dt = data_projected.time.values[1] - data_projected.time.values[0]
    n_convolution_bins = int(0.5 / dt)
    window = np.ones(n_convolution_bins) / n_convolution_bins

    time_vector = data_projected.time.values

    # bwr colormap, n=8 sampples RdYlGn
        
    for i_s, shift in enumerate(shifts):
        for i_past, (r1, r2) in enumerate(itertools.product(r1s, r2s)):
            data_temp = data_projected.sel(
                trial_id=(data_projected.switch == shift) & (data_projected.R_1 == r1) & (data_projected.R_2 == r2)
                )

            class_mean = data_temp.mean(dim='trial_id').values
            class_mean_smoothed = ndimage.convolve(class_mean, window, mode='nearest')

            ax.plot(time_vector, class_mean_smoothed, color=colors[i_past], label=f'{labels[shift]}: t-1: {r1}, t-2: {r2}', linewidth=linewidth, zorder=10, linestyle=linestyles[shift])

    if t_interest_value is not None:
        ax.axvspan(t_interest_value[0], t_interest_value[1], color='grey', alpha=0.2, label='eval window')

    # add behav keyboints
    plot_keypoints(ax, n_extra_trials, mark_event='Fb', xlabels='events')
    # y grid only
    ax.grid(axis='x', alpha=.5, linewidth=1, linestyle='--')
    ax.axhline(0, color='k', linestyle='-', linewidth=linewidth*.75, zorder=0)

    # remove left and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # move legend outside
    if not paper_format:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # save as svg
    ax.set_ylabel('$V_t$', fontsize=10)
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
    colors = value_gradient(8)
        
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



def create_r_style_plot(ax=None, data=None, x_col=None, y_col=None, title=None, session_col='session', paper_format=False):
    """Create R-style plot with violin, box, and jittered points
    
    Args:
        ax: matplotlib axis
        data: DataFrame with the data
        x_col: column name for x-axis categories
        y_col: column name for y-axis values
        title: plot title
        session_col: column name for session identifier (optional, for connecting points)
    """
    palette = COLORS
    axis = ax
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 2 
        w = 2
        s = 3

    else:
        plt.rcParams.update({'font.size': 18})
        h = 3  # in cm
        w = 4  # in cm
        s = 10

    
    # distribution of pos and neg feedback along neural value
    if axis is None:
        fig, ax = plt.subplots(1, 1, figsize=(w, h))


    # Get unique categories
    # sort by x_col
    data = data.copy().sort_values(by=x_col)
    categories = data[x_col].unique()
    x_positions = np.arange(len(categories))
    
    x_offset = .2
    s = 10
    lw = .4 
    
    # Check if we'll be drawing connecting lines
    will_connect_lines = False
    if session_col is not None and len(categories) == 2:
        sessions_cat1 = set(data.loc[data[x_col] == categories[0], session_col])
        sessions_cat2 = set(data.loc[data[x_col] == categories[1], session_col])
        common_sessions = sessions_cat1.intersection(sessions_cat2)
        
        valid_sessions = []
        for session in common_sessions:
            count_cat1 = len(data.loc[(data[x_col] == categories[0]) & (data[session_col] == session)])
            count_cat2 = len(data.loc[(data[x_col] == categories[1]) & (data[session_col] == session)])
            if count_cat1 == 1 and count_cat2 == 1:
                valid_sessions.append(session)
        
        will_connect_lines = len(valid_sessions) > 0
    
    # Create violin plots (distributions)
    violin_positions = []
    for i in range(len(categories)):
        if will_connect_lines and i == 0:
            # First group: distribution on left
            violin_positions.append(x_positions[i] - x_offset)
        else:
            # Default: distribution on right
            violin_positions.append(x_positions[i] + x_offset)
    
    # Create half violin plots (distributions) - flat side towards boxplot
    for i, cat in enumerate(categories):
        y_data = data.loc[data[x_col] == cat, y_col].values
        
        # Create violin plot data
        density = stats.gaussian_kde(y_data)
        
        # Create y range for the distribution
        y_min, y_max = y_data.min(), y_data.max()
        y_range = np.linspace(y_min - 0.1 * (y_max - y_min), 
                             y_max + 0.1 * (y_max - y_min), 200)
        density_values = density(y_range)
        
        # Normalize density values to desired width
        density_values = density_values / density_values.max() * 0.15
        
        # Determine which side the distribution should be on
        if will_connect_lines and i == 0:
            # First group: distribution on left (away from points/lines)
            x_dist = violin_positions[i] - density_values
            x_null = violin_positions[i]  + .05
        else:
            # Default: distribution on right (away from points)
            x_dist = violin_positions[i] + density_values
            x_null = violin_positions[i] - .05
        
        # Fill the half violin
        ax.fill_betweenx(y_range, x_null, x_dist, 
                        color=palette[cat], alpha=0.3, 
                        edgecolor='black', linewidth=2*lw)
    
    # Create boxplots - centered
    box_data = [data.loc[data[x_col] == cat, y_col].values for cat in categories]
    bp = ax.boxplot(box_data, positions=x_positions, widths=0.2, patch_artist=True,
                   boxprops=dict(alpha=0.7),
                   showfliers=False,
                   medianprops=dict(color='black'),
                   #whiskerprops=dict(linewidth=1.5),
                   #capprops=dict(linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='black', markersize=3, alpha=0.5))
    
    # Color the boxes
    colors = [palette[cat] for cat in categories]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Store jitter positions for connecting lines
    jitter_positions = {}
    
    # Add jittered points
    for i, cat in enumerate(categories):
        y_data = data.loc[data[x_col] == cat, y_col].values
        if 'signif' in data.columns:
            signifs = data.loc[data[x_col] == cat, 'signif'].values
            colors = [palette[cat] if signif else 'white' for signif in signifs]
        else:
            colors = [palette[cat]] * len(y_data)
        
        # Determine jitter position based on whether we're connecting lines
        if will_connect_lines and i == 0:
            # First group: points on right when connecting lines
            x_jitter = np.random.normal(x_positions[i] + x_offset, 0.02, size=len(y_data))
        else:
            # Default: points on left
            x_jitter = np.random.normal(x_positions[i] - x_offset, 0.02, size=len(y_data))
        
        ax.scatter(x_jitter, y_data, alpha=0.6, s=s, c=colors,
                   edgecolor='black', linewidth=lw)
        
        # Store jitter positions if we need to connect points
        if session_col is not None:
            session_data = data.loc[data[x_col] == cat, session_col].values
            jitter_positions[cat] = dict(zip(session_data, zip(x_jitter, y_data)))
    
    # Connect points between categories if session_col is provided and we have exactly 2 categories
    if will_connect_lines:
        # Draw connecting lines for valid sessions
        for session in valid_sessions:
            if session in jitter_positions[categories[0]] and session in jitter_positions[categories[1]]:
                x1, y1 = jitter_positions[categories[0]][session]
                x2, y2 = jitter_positions[categories[1]][session]
                ax.plot([x1, x2], [y1, y2], '-', color='gray', alpha=0.2, zorder=0, lw=lw)
    
    # Perform t-test and add significance
    if len(categories) == 2:
        group1_data = data.loc[data[x_col] == categories[0], y_col]
        group2_data = data.loc[data[x_col] == categories[1], y_col]
        #_, p_value = stats.ttest_ind(group1_data, group2_data)
        U, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')


        # Position for significance bar
        y_max = data[y_col].max()
        y_pos = y_max + 0.1 * abs(y_max)
        
        # Add significance annotation
        if p_value < 0.001:
            significance = '***, U={:.2f}, p={}'.format(U, p_value)
        elif p_value < 0.01:
            significance = '**, U={:.2f}, p={}'.format(U, p_value)
        elif p_value < 0.05:
            significance = '*, U={:.2f}, p={}'.format(U, p_value)
        else:
            significance = 'n.s., U={:.2f}, p={}'.format(U, p_value)
        
        # Add the bar and text for p-value
        ax.plot([0, 1], [y_pos, y_pos], '-k', lw=2*lw)
        ax.text(0.5, y_pos + 0.02 * abs(y_max), f'{significance} (n={len(group1_data)})', 
                ha='center', va='bottom', fontsize=8)
    
    # Formatting
    ax.axhline(0, linestyle='--', alpha=0.7, color='black', lw=2*lw, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Unrewarded', 'Rewarded'], rotation=20)
    ax.set_ylabel('Change in neural value \n$\Delta V = V_t - V_{t-1}$', fontsize=8)
    ax.set_xlabel('')
    ax.set_title(title, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if axis is None:
        return fig, ax
    else:
        return ax


def plot_dV_per_fb(behav_new, paper_format=False, ylim=None, show_datapoints=False, ax=None):
    axis = ax
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.2  # in cm
        w = 2
        s = 3
    else:
        plt.rcParams.update({'font.size': 18})
        h = 3  # in cm
        w = 4  # in cm
        s = 10

    # Example: get 10 colors from your scale

    # remove 1% outliers in dV
    dV_threshold = np.percentile(np.abs(behav_new.dV.values), 99)
    behav_new = behav_new[np.abs(behav_new.dV.values) < dV_threshold]

        
    df = pd.DataFrame({
        #'trial_id': behav_new.trial_id.values,
        #'V_t': behav_new.V_t.values,
        #'V_t_p1': data_projected.V_t_p1.values,
        'session': behav_new.session.values,
        'dV': behav_new.dV.values,
        'feedback': behav_new.feedback.values,
        'signif': behav_new.signif.values,
        #'fb_sequence': behav_new.fb_sequence.values,
        #'fb_sequence_last_2': behav_new.fb_sequence.values % 4,
    })


    # distribution of pos and neg feedback along neural value
    if axis is None:
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    create_r_style_plot(ax, df, 'feedback', 'dV')

    # Add the horizontal line at y=0
    ax.axhline(0, color='k', linestyle='--', alpha=0.7, linewidth=0.75, zorder=0)

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    #ax.set_xlabel('Feedback sequence')
    ax.set_ylabel('$\Delta V$')
    '''ax.set_yticks([0, 1, 2, 3])   
    ax.set_yticklabels('''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if axis is None:
        return fig, ax
    

def plot_Vt_per_sequence(behav_new, paper_format=False, ylim=None, show_datapoints=True, ax=None, showfliers=True):
    axis_original = ax
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 2 
        w = 2.5
        s = 3
    else:
        plt.rcParams.update({'font.size': 18})
        h = 3  # in cm
        w = 4  # in cm
        s = 10

    
    # distribution of pos and neg feedback along neural value
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    # Example: get 10 colors from your scale
    colors = value_gradient(8)

    # remove 1% outliers in V_t
    '''V_t_threshold = np.percentile(np.abs(behav_new.V_t.values), 99)
    behav_new = behav_new[np.abs(behav_new.V_t.values) < V_t_threshold]'''

    df = pd.DataFrame({
        #'trial_id': behav_new.trial_id.values,
        'V_t': behav_new.V_t.values,
        'fb_sequence': behav_new.fb_sequence.values,
        'fb_sequence_last_2': behav_new.fb_sequence.values % 4,
    })

    label_mapping_long = {0: '0\n0\n0', 1: '0\n0\n1', 2:'0\n1\n0', 3:'0\n1\n1', 4: '1\n0\n0', 5: '1\n0\n1', 6:'1\n1\n0', 7:'1\n1\n1'}
    label_mapping_short = {0: '0\n0', 1: '0\n1', 2:'1\n0', 3:'1\n1'}
    df['fb_sequence_str'] = df['fb_sequence'].map(label_mapping_long)
    df['fb_sequence_last_2_str'] = df['fb_sequence_last_2'].map(label_mapping_short)

    ## NEW: statistical test

    # Compute Spearman correlation between numeric sequence order and V_t
    rho, pval = stats.spearmanr(df['fb_sequence'], df['V_t'])  # The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship between two datasets. 

    print(f"Spearman correlation (rho): {rho:.3f}, p-value: {pval:.1e}")

    # Optional: annotate plot with significance result
    ax.text(0.5, 0.95, f"rho = {rho:.2f}, p = {pval:.1e}",
            ha='center', va='top', transform=ax.transAxes,
            fontsize=6 if paper_format else 10)

    sns.boxplot(df.sort_values('fb_sequence_str'), x='fb_sequence_str', hue='fb_sequence_str', y='V_t', palette=colors, ax=ax, showfliers=showfliers, boxprops={'linewidth': 1}, legend=False, width=0.5)
    if show_datapoints:
        if show_datapoints:
            # Get unique groups and their positions
            unique_groups = df['fb_sequence_str'].unique()
            for i, group in enumerate(sorted(unique_groups)):
                group_data = df[df['fb_sequence_str'] == group]
                x_pos = np.full(len(group_data), i - 0.3)  # Shift left by 0.15
                x_pos += np.random.normal(0, 0.05, size=len(group_data))  # Add jitter
                ax.scatter(x_pos, group_data['V_t'], color=colors[i], s=s, alpha=0.5, edgecolor='none', linewidth=0.5, label=group, zorder=0)

    ax.axhline(0, color='k', linestyle='--', alpha=0.7, linewidth=0.75, zorder=0)

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    #ax.set_xlabel('Feedback sequence')
    ax.set_ylabel('Mean projections\nfrom $Lt(t+1)$ to $Fb(t+1)$')
    # set ticks at every 1
    y_min = df['V_t'].min()
    y_max = df['V_t'].max()
    ax.set_ylim(-np.abs(y_max), np.abs(y_max))
    y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ylim is not None:
        ax.set_ylim(ylim)

    #plt.tight_layout()
    if axis_original is None:
        return fig, ax

def plot_projection_timepoint(behav_new, paper_format=False, ylim=None, show_datapoints=False):
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 1.5  # in cm
        w = 4
        s = 5
    else:
        plt.rcParams.update({'font.size': 18})
        h = 6  # in cm
        w = 8  # in cm
        s = 10

    # Example: get 10 colors from your scale
    #     
    # distribution of pos and neg feedback along neural value
    fig, axs = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={'width_ratios': [2, 1]})

    plot_dV_per_fb(behav_new, paper_format=paper_format, ylim=ylim, show_datapoints=show_datapoints, ax=axs[1])
    plot_Vt_per_sequence(behav_new, paper_format=paper_format, ylim=ylim, show_datapoints=show_datapoints, ax=axs[0])

    return fig, axs

def plot_prop_signif_sessions(cpd_temp, ax=None):
    axis_original = ax
    # Initialize container
    plt.rcParams.update({'font.size': 8})

    monkey = cpd_temp.monkey.unique()[0]
    subregion = cpd_temp.subregion.unique()[0]
    
    regressors = ["_".join(col.split('_')[1:]) for col in cpd_temp.columns if col.startswith('cpd_R_')]

    counts = {}
    for reg in regressors:
        #count = (cpd_temp[f'pval_{reg}'] < 0.05).sum()
        count_corrected = (cpd_temp[f'pval_{reg}_corrected'] < 0.05).sum()
        counts[reg] = count_corrected
        #row[f'{reg}_corrected'] = count_corrected

    # small gap
    if axis_original is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))

    n_sessions = len(cpd_temp)

    # Get row from sig_counts_df

    props = np.array([counts[reg] / n_sessions for reg in regressors])

    ax.bar(range(1, len(regressors)+1), props * 100, color=COLORS[subregion], alpha=0.8)
    ax.set_xticks(range(1, len(regressors)+1))
    ax.set_xticklabels(regressors, rotation=45, ha='right')
    #ax.set_ylim(0, session_n + 1)
    ax.set_ylabel('% Significant \nSessions')
    ax.set_title(f'{monkey} {subregion} \n(n={n_sessions})')
    #ax.axhline(y=5, color='black', linestyle='--', alpha=0.3)  # Optional threshold line

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #plt.tight_layout()

    if axis_original is None:
        return fig, ax

def green_red_plot(behav_new, paper_format=False, ax=None):
    from matplotlib.gridspec import GridSpec

    axis_original = ax
    
    # project to axis
    if paper_format:
        plt.rcParams.update({'font.size': 8})
        h = 2  # in cm
        w = 2
        s = 10
        linewidth = 0.7
    else:
        plt.rcParams.update({'font.size': 12})
        h = 6  # in cm
        w = 6  # in cm
        s = 10
        linewidth = 1

    if ax is None:
        # Method 1: Use GridSpec directly (recommended)
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    # extract datapoints
    V_t = behav_new.V_t.values
    V_t_p1 = behav_new.V_t_p1.values
    fb_vector = behav_new.feedback.values

    V_min, V_max = np.min(V_t), np.max(V_t)
    x_lines = np.arange(V_min, V_max, 0.01)

    V_t_p1_min, V_t_p1_max = np.min(V_t_p1), np.max(V_t_p1)
    # remove outliers (5%)

    axis_extreme = np.max(np.abs([V_t.min(), V_t.max(), V_t_p1.min(), V_t_p1.max()]))

    # Scatter positive and negative feedback with fit lines
    for fb_curr in [0, 1]:
        # get pos/neg trial
        V_t_curr = V_t[fb_vector == fb_curr]
        V_t_p1_curr = V_t_p1[fb_vector == fb_curr]

        # scatter trials
        ax.scatter(V_t_curr, V_t_p1_curr, color=COLORS[fb_curr], alpha=.5, s=s, edgecolor='none')

        # fit and plot slopes
        slope, intercept, r_value, p_value, std_err = stats.linregress(V_t_curr, V_t_p1_curr)
        fit_line = slope * x_lines + intercept
        ax.plot(x_lines, fit_line, color=COLORS[fb_curr], alpha=.7, lw=2, label=f'slope {slope:.2f} p={p_value:.2f}', zorder=10)

        # mean line for each group
        #mean_dV = np.mean(dV_curr)
        #ax_hist.axhline(mean_dV, color=COLORS[fb_curr], linestyle='--', alpha=0.7, linewidth=.7, label=f'mean fb {["negative", "positive"][fb_curr]}')

    # Diagonal line negative slope
    ax.plot(x_lines, x_lines, color='k', alpha=0.7, lw=2, zorder=8)

    # Reference lines
    mean_V_t = 0
    mean_V_t_p1 = 0
    ax.axvline(mean_V_t, color='k', linestyle='--', alpha=0.7, linewidth=linewidth)
    ax.axhline(mean_V_t_p1, color='k', linestyle='--', alpha=0.7, linewidth=linewidth)

    # Add vertical reference lines
    ax.legend(loc='upper left', fontsize=8 if paper_format else 10)

    # Add horizontal reference lines
    #ax_hist.axhline(0, color='k', linestyle='-', alpha=0.7, linewidth=.7)

    # Styling and labels
    ax.scatter([], [], color=COLORS[1], label='Unrewarded')
    ax.scatter([], [], color=COLORS[0], label='Rewarded')

    #ax.legend(loc='upper center')

    ax.set_xlabel('$V_t$', fontsize=10 if paper_format else 12)
    ax.set_ylabel('$V_{t+1}$', fontsize=10 if paper_format else 12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim([-axis_extreme, axis_extreme])
    ax.set_ylim([-axis_extreme, axis_extreme])

    # Create a second axis for the histogram
    if axis_original is None:
        return fig, ax

def plot_simple_glm_weights(weights_all, ax=None, show_data=True):
    axis_original = ax
    plt.rcParams.update({'font.size': 8})

    if ax is None:
        fig, ax = plt.subplots(1, 1 , figsize=(2, 1.5))

    # font size
    weight_cols = [col for col in weights_all.columns if col.startswith('R_')]
    subregion = weights_all.subregion.unique()[0]
    monkey = weights_all.monkey.unique()[0]
    # Calculate min and max

    weights_filtered = weights_all.replace([np.inf, -np.inf], np.nan).dropna()  # Replace inf values with NaN, then drop NaN values
    weights_temp = weights_filtered.drop(columns=['monkey', 'session', 'subregion'])

    ymax = weights_temp.mean().max()
    ymin = weights_temp.mean().min()

    # Visualization - mean weights with error bars and in the background plot individual sessions with lines connecting them
    # weights temp is a DataFrame with weights for each regressor
    weights = weights_temp.mean().values
    weights_err = weights_temp.std().values #/ np.sqrt(len(weights_temp))

    ax.errorbar(range(1, len(weights) + 1), weights, yerr=weights_err,
           fmt='o-', color=COLORS[subregion], 
           markeredgecolor='black', markeredgewidth=.5,
           label='session mean ± SEM', 
           markersize=6, capsize=2)
    # add stirplot
    if not show_data:
        for i, col in enumerate(weights_temp.columns):
            xpos = i+1 -.2
            jitter = np.random.normal(0, 0.05, size=len(weights_temp[col]))
            ax.scatter(xpos + jitter, weights_temp[col].values, color='black', alpha=0.2, zorder=0, s=1)
    else:
        for i, row in weights_temp.iterrows():
            ax.plot(range(1, len(weights) + 1), row.values, color='grey', alpha=0.1)


    # fit exponential and print params
    '''def exponential_fit(x, C, a, b):
        return C*(a * (1-a) ** (x - 1)) + b
    from scipy.optimize import curve_fit
    x_data = np.arange(1, len(weights) + 1)
    try:
        popt, _ = curve_fit(exponential_fit, x_data, weights, p0=(1, 0.5, 0))
        fit_line = exponential_fit(x_data, *popt)
        ax.plot(x_data, fit_line, color=COLORS[subregion], linestyle='--', label=f'{popt}', zorder=1)
    except Exception as e:
        print(f"Error fitting exponential for {monkey} {subregion}: {e}")
        fit_line = None'''



    # Plot the weights as a bar plot

    
    #ax.scatter(range(1, len(weights) + 1), weights, color='steelblue', alpha=0.7, s=80, edgecolor='k')
    ax.axhline(0, color='black', linestyle='dotted', alpha=0.3)
    ax.set_xlabel('Regressor Index')
    ax.set_ylabel('Regression weight')
    ax.set_xticks(range(1, len(weights) + 1))
    ax.set_xticklabels([f'R_{i}' for i in range(1, len(weights)+1)], rotation=45, ha='right')
    #ax.grid(True, alpha=0.3)
    #ax.legend(loc='upper right', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #ax.set_ylim(ymin - 0.1, ymax + 0.1)

    ax.set_title(f'{monkey} {subregion}')

    if axis_original is None:
        return fig, ax

def plot_cpds_history_neural_value(cpd_temp, null=None, ax=None, show_data=False):
    axis_original = ax
    if ax is None:
        h = 2
        w = 2.5
        fig, ax = plt.subplots(1, 1, figsize=(w, h))

    monkey, subregion = cpd_temp.monkey.unique()[0], cpd_temp.subregion.unique()[0]

    cpd_cols = [col for col in cpd_temp.columns if col.startswith('cpd_R_')]
    ymax = cpd_temp.groupby(['monkey', 'subregion'])[cpd_cols].mean().max().max()

    cpd_cols = [col for col in cpd_temp.columns if col.startswith('cpd_R_')]
    cpd_vals = cpd_temp[cpd_cols]

    cpd_means = cpd_vals.mean().values
    cpd_sems = cpd_vals.std().values #/ np.sqrt(len(cpd_vals))

    # Plot means + error bars
    ax.errorbar(range(1, len(cpd_means)+1), cpd_means, yerr=cpd_sems, 
           fmt='o-', color=COLORS[subregion], 
           markeredgecolor='black', markeredgewidth=.5,
           label='session mean ± SEM', 
           markersize=6, capsize=2)
    
    # Plot individual sessions
    # add stirplot
    if show_data:
        # show individual sessions with dots
        for i, col in enumerate(cpd_vals.columns):
            xpos = i+1 -.3
            jitter = np.random.normal(0, 0.05, size=len(cpd_vals[col]))
            ax.scatter(xpos + jitter, cpd_vals[col].values, color='black', alpha=.2, zorder=0, s=2)
        '''# show connected lines
        for _, row in cpd_vals.iterrows():
            ax.plot(range(1, len(cpd_cols)+1), row.values, color='grey', alpha=0.1)'''

    # Stars: compare each regressor to null
    
    for i, reg in enumerate(cpd_cols):
        if null is None:  # if there is no null, we assume that there is only one session in the data: print pvalue of CPD
            pcol = f'pval_{"_".join(reg.split("_")[1:])}'
            pval = cpd_temp[pcol].values[0]
        else:
            pval = np.sum(null >= cpd_means[i]) / null.shape[0]

        if pval < 0.05:
            ypos = cpd_means[i] + cpd_sems[i] + .5 if not np.isnan(cpd_sems[i]) else cpd_means[i] + 3
            ax.text(i+1, ypos, '*', color='black', fontsize=8, ha='center')

    ax.axhline(0, color='black', linestyle='dotted', alpha=0.3)
    ax.set_title(f'{monkey} {subregion}')
    ax.set_ylabel('CPD (%)')
    #ax.set_ylim(-5, ymax+10)
    ax.set_xticks(range(1, len(cpd_cols)+1))
    ax.set_xticklabels([f't-{i}' for i in range(1, len(cpd_cols)+1)], rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.legend(loc='upper right', fontsize=8)

    if axis_original is None:
        return fig, ax

def plot_across_time_decodability(decodability_matrix, projected_data=None, cmap='RdBu', vmin=None, vmax=None):
    """
    Plot across-time decodability matrix, simple decodability in time, and optionally projected data.
    
    Parameters:
    -----------
    decodability_matrix : xarray.DataArray
        Matrix of across-time decodability scores
    projected_data : xarray.DataArray, optional
        Projected neural data
    figsize : tuple
        Figure size
    cmap : str
        Colormap for heatmap
    vmin, vmax : float, optional
        Color scale limits
    """
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.size'] = 8

    #cmap = 'RdBu'
    # Determine number of subplots
    n_plots = 2
    
    h, w = 1.7, 5 * 1.7/2  # height, width in cm
    fig, axes = plt.subplots(1, n_plots, figsize=(w, h))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Decodability matrix
    ax = axes[plot_idx]
    #max_value = np.abs(decodability_matrix.values).max() - .5
    #vmin, vmax = .5-max_value, .5+max_value
    data = decodability_matrix#.where((decodability_matrix > .55) | (decodability_matrix < .45), np.nan)
    im = ax.imshow(data, aspect='auto', origin='lower', interpolation='gaussian',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[decodability_matrix.test_time.min(),
                           decodability_matrix.test_time.max(),
                           decodability_matrix.train_time.min(),
                           decodability_matrix.train_time.max()])
    

    plot_keypoints(ax, (-1, 0), axis='both', mark_event='Fb', xlabels='events')
    ax.grid(alpha=0.5, linestyle='--')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Test Time (s)')
    ax.set_ylabel('Train Time (s)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy')
    
    plot_idx += 1
    
    # Plot 2: Simple decodability in time (diagonal)
    ax = axes[plot_idx]
    
    # Extract diagonal values
    diagonal_values = projected_data.decodability.data
    times = projected_data.time.values  # or test_time, they should be the same for diagonal

    ax.plot(times, diagonal_values, linewidth=1, color='black')
    ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='Chance level')
    
    plot_keypoints(ax, (-1, 0), mark_event='Fb', xlabels='events')
    ax.set_ylim([.45, 1.01])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.5, linestyle='--', axis='x')
    ax.set_xlim([-5, 5])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.legend()

    
    #plt.tight_layout()
    return fig, axes


### Save figures
from matplotlib.backends.backend_pdf import PdfPages
import io
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def save_figures_split_layout(figures, filename="figures_split_layout.pdf", figsize=(15, 7), title=None):
    """
    Save multiple figures in a 2-row, 5-column layout with optional title:
    - figures[0] -> top-left (0, 0)
    - Remaining figures split:
        - First half go to top row (0, 1 to 4)
        - Second half go to bottom row (1, 2 to 4)
    - Bottom row cols 0 and 1 remain blank
    """
    if len(figures) < 2:
        raise ValueError("Need at least 2 figures for this layout.")

    fig_combined, axs = plt.subplots(2, 6, figsize=figsize, gridspec_kw={'width_ratios': [3, 1, 2, 1, 1, 1], 'height_ratios': [1, 1]})
    axs = np.array(axs)

    # Turn off all axes
    for ax in axs.flatten():
        ax.axis('off')

    # Add title if provided
    if title:
        fig_combined.suptitle(title, fontsize=16, y=1.02)

    # Place first figure
    _add_fig_to_ax(figures[0], axs[0, 0])

    # Split and place remaining figures
    rest = figures[1:]
    mid = len(rest) // 2 + len(rest) % 2
    top_rest = rest[:mid]
    bottom_rest = rest[mid:]

    for i, fig in enumerate(top_rest):
        col = i + 1
        if col < 6:
            _add_fig_to_ax(fig, axs[0, col])

    for i, fig in enumerate(bottom_rest):
        col = i + 1
        if col < 6:
            _add_fig_to_ax(fig, axs[1, col])

    fig_combined.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title
    fig_combined.savefig(filename, bbox_inches='tight')
    plt.close(fig_combined)
    print(f"Saved combined figure to {filename}")

def _add_fig_to_ax(fig, ax):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = np.array(Image.open(buf))
    ax.imshow(img)
    ax.axis('off')

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score, KFold
import xarray as xr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut

from popy.io_tools import load_behavior, load_neural_data
from popy.neural_data_tools import *
from popy.behavior_data_tools import *


# Data processing #
def _get_data_of_interest(neural_data, step_len):
    '''
    when we do not want to decode every time point, we can subsample the data to only include the time points of interest. This function does that.
    '''
    time_original = neural_data.time.data
    times_of_interest = np.arange(time_original[0], time_original[-1], step_len)
    # find times closest to times of interest
    times_idx = [np.argmin(np.abs(time_original - time)) for time in times_of_interest]
    times_of_interest = time_original[times_idx]
    return neural_data.isel(time=times_idx)


def _create_results_xr(conditions, time_vector, areas):
    xr_scores = xr.Dataset(
        {
            'scores': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
            'pvals': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
            'perm_mean': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
            'perm_std': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
            #'cv_mean': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
            #'cv_std': (['target', 'time', 'area'], np.full((len(conditions), len(time_vector), len(areas)), np.nan)),
        },
        coords={
            'target': np.array(conditions, dtype='U50'),  # Unicode string with max length 50
            'area': np.array(areas, dtype='U10'),         # Unicode string with max length 10
            'time': time_vector,
        }
    )

    return xr_scores


def _label_encoder(y):
    """
    Converts a vector of labels to numerical values.
    """
    # if there are nans
    assert not np.isnan(y).any(), "Label vector contains NaN values."

    # encode as integers
    le = LabelEncoder()
    le.fit(y)
    y_numerical = le.transform(y)

    return y_numerical
    
    """
    This part replaced nan values with nan values in the labels, but it is not necessary anymore since we remove nan trials in the first place.

    # array of zeros, set dtype to , dtype=object
    y_new = np.zeros(y_numerical.shape, dtype='float64')

    # reset values to nan if they were nan (LabelEncoder sets nan to some numerical value as well...)
    # Also, for some reason, labels are relabeled to start from 0...
    class_names = le.classes_ # get the class names
    for i, class_name in enumerate(class_names):
        if class_name == 'nan':
            y_new[np.where(y_numerical == i)] = np.nan
            print("nan s'est glisse dans les labels")
        else:
            y_new[np.where(y_numerical == i)] = i

    return y_new"""


def _preproc_data(neural_dataset_original, 
                  target_name, 
                  group_target_name=None
                  ):
    """
    Creates a matrix of regressors (X) and a vector of labels (y) from the neural data.
    
    Includes 3 steps:
    0. Creating data matrix and labels
    1. Remove nan trials
    2. If categorical, create numerical labels and balance the dataset
    
    Returns the data matrix and labels, ready to plug into the decoder.
    """

    neural_dataset = neural_dataset_original.copy(deep=True)
    y_groups = None

    # Step 0. Drop nan trials from labels (and groups)
    neural_dataset = remove_nan_labels(neural_dataset, target_name)  # drop trials with nan behavioural label
    if group_target_name is not None:
        neural_dataset = remove_nan_labels(neural_dataset, group_target_name)  
    
    # Step 1. Balance classes
    #X, y = _balance_categorical(X, y)
    n_classes = len(np.unique(neural_dataset[target_name].data))
    
    n_trials_before = len(neural_dataset)
    if n_classes <= 5 and group_target_name is None:  # if y is categorical and groups are not provided
        neural_dataset = balance_labels(neural_dataset, coords=target_name)
    elif n_classes <= 5 and group_target_name is not None:  # if y is categorical and groups are provided
        neural_dataset = balance_labels(neural_dataset, coords=[target_name, group_target_name])
    elif n_classes > 5 and group_target_name is not None:  # if y is continuous and groups are provided
        neural_dataset = balance_labels(neural_dataset, coords=group_target_name)

    n_trials_after = len(neural_dataset)
    print(f"Removed {n_trials_before - n_trials_after}/{n_trials_before} trials with unbalanced classes.\n")

    # Step 2. Creating data matrix and labels
    X = neural_dataset.data
    y = neural_dataset[target_name].data
    if group_target_name is not None:
        y_groups = neural_dataset[group_target_name].data

    # Step 3. Encode categoricaxl labels
    if len(np.unique(y)) <= 5:  # if y is categorical
        y = _label_encoder(y)  # relabel from 0

    if y_groups is not None:  # if groups are provided, encode them as well
        y_groups = _label_encoder(y_groups)

    return X, y, y_groups


# Decoding #


def linear_decoding(X, y, groups=None, K_fold=None, n_perm=1000, n_jobs=1):
    """
    Decoding the target variable 'y' from the neural data matrix 'X'.

    In case of continuous data, the decoder is a linear regression, and the scoring measure is 'r2'.
    In case of discrete data, the decoder is a logistic regression, and the scoring measure is 'accuracy'.
    """

    if K_fold is None:
        K_fold = 10
    ## DECODER SETUP


    # is data continuous or discrete?
    dtype = 'continuous' if len(np.unique(y)) > 5 else 'discrete'
    if dtype == 'continuous':  # in case of continuous data -> regression
        if groups is None:
            kf = KFold(n_splits=K_fold, shuffle=True)
        else:
            kf = LeaveOneGroupOut()  # use LeaveOneGroupOut decode the value across targets (trained on one target, tested on the other) 
        decoder = LinearRegression()
        #decoder = Lasso(alpha=.05)
        scoring_function = 'r2'
    elif dtype == 'discrete':  # in case of dicrete data -> classification
        if groups is None:
            kf = StratifiedKFold(n_splits=K_fold, shuffle=True)  # use stratified KFold to preserve the ratio of classes in each fold
        else:
            kf = LeaveOneGroupOut()
            #kf = LeaveOneGroupOut()
        decoder = LogisticRegression()
        scoring_function = 'accuracy'

    ## DECODING LOOP

    # scores of shape (K_fold, n_bins)
    scores = np.empty(X.shape[2])
    perm_scores_mean = np.empty(X.shape[2])
    perm_scores_std = np.empty(X.shape[2])
    pvals = np.empty(X.shape[2])
    #cv_mean_scores = np.empty(X.shape[2])
    #cv_std_scores = np.empty(X.shape[2])

    # loop through time points
    for i_train, t_train in enumerate(np.arange(X.shape[2])):
        X_temp = X[:, :, t_train]  # select (trial, unit) data at time t_train
        if groups is None:
            res = permutation_test_score(decoder, X_temp, y, scoring=scoring_function, cv=kf, n_permutations=n_perm, n_jobs=n_jobs)  # TODO maybe its not necessary to avoid passing groups=None?
        else:
            res = permutation_test_score(decoder, X_temp, y, groups=groups, scoring=scoring_function, cv=kf, n_permutations=n_perm, n_jobs=n_jobs)
    
        # unpack results and store them
        score, perm_scores, pvalue = res

        scores[i_train] = score
        pvals[i_train] = pvalue
        perm_scores_mean[i_train] = np.mean(perm_scores)
        perm_scores_std[i_train] = np.std(perm_scores)

        # cross-validation scores without permutation (sanity check)
        '''score = cross_val_score(decoder, X_temp, y, groups=groups, cv=kf, scoring=scoring_function)
        cv_mean_scores[i_train] = np.mean(score)
        cv_std_scores[i_train] = np.std(score)'''

    return scores, pvals, perm_scores_mean, perm_scores_std#, cv_mean_scores, cv_std_scores


def load_data_for_decoder(monkey, session, n_extra_trials=(-1, 1)):
    # 1. Behavior data

    # Load behavior data
    behav = load_behavior(monkey, session)
    behav = drop_time_fields(behav)

    # add behav vars to decode
    behav = add_value_function(behav)  # add value function for its decoding
    behav = add_shift_value(behav)  # add shift value for its decoding
    behav = add_switch_info(behav)  # add switch information for its decoding

    '''for alpha in np.linspace(.05, 1, 20):
        behav = add_shift_value(behav, alpha_ka=alpha, alpha_po=alpha)
        behav = behav.rename(columns={'shift_value': f'shift_value_{alpha:.2f}'})'''
    #behav = add_shift_value(behav)  # add shift value for its decoding

    behav['target_shuffled'] = behav['target'].copy()
    behav['target_shuffled'] = behav['target'].sample(frac=1).values

    behav['target_no_2'] = behav['target'].copy()
    behav['target_no_2'] = behav['target_no_2'].replace({2: np.nan})
    behav['target_no_2_shuffled'] = behav['target_no_2'].copy()
    behav['target_no_2_shuffled'] = behav['target_no_2_shuffled'].sample(frac=1).values

    # print number of targets
    for target in behav['target'].unique():
        n_targets = len(behav[behav['target'] == target])
        print(f"Target {target}: {n_targets} trials")
    
    # 2. Neural data

    # Load neural data
    neural_data = load_neural_data(monkey, session, hz=1000)
    n_units_all = len(neural_data['unit'].values)

    # remove some units
    neural_data = remove_low_fr_neurons(neural_data, 1, print_usr_msg=False)
    neural_data = remove_trunctuated_neurons(neural_data, mode='remove', delay_limit=10, print_usr_msg=False)
    n_units_kept = len(neural_data['unit'].values)
    if len(neural_data['unit'].values) == 0:
        raise ValueError(f"No neurons left for {monkey}_{session}")
    
    # process neural data
    neural_data = add_firing_rates(neural_data, drop_spike_trains=True, method='gauss', std=.05)
    neural_data = downsample_time(neural_data, 100)
    neural_data = scale_neural_data(neural_data)
    neural_data = time_normalize_session(neural_data)

    # 3. build neural dataset and merge with behavior
    neural_dataset = build_trial_dataset(neural_data, mode='full_trial', n_extra_trials=n_extra_trials)
    neural_dataset = merge_behavior(neural_dataset, behav)

    '''print(f"Monkey: {monkey}, Session: {session}\n",
          f"Removed {n_units_all - n_units_kept} / {n_units_all} neurons\n")'''

    return neural_dataset


### Decoding pipeline

def run_decoder(monkey, session, PARAMS, n_jobs=1):
    """
    Decoding the target variable from the neural data.

    Includes these steps:
    1. loads data, preprocess it, create labelled dataset
    2. run decoder on the labelled dataset

    Returns the scores of the decoder.

    """
    # if targets is not a list, make it a list (i.e. if only one target is passed as string)
    if not isinstance(PARAMS['conditions'], list): 
        PARAMS['conditions'] = [PARAMS['conditions']]

    # print log
    print(f"Running for monkey {monkey} and session {session}")

    ## init data
    neural_dataset = load_data_for_decoder(monkey, session, PARAMS['n_extra_trials'])

    # sanity: groups must be categorical
    if PARAMS['group_target'] is not None:
        groups = neural_dataset[PARAMS['group_target']].data
        assert len(np.unique(groups)) <= 5, "Groups must be categorical."
    
    # 0. get bins of interest
    neural_data = _get_data_of_interest(neural_dataset, PARAMS['step_len'])

    # Create an empty dataset with dimensions 'time' and 'area'
    xr_scores = _create_results_xr(
        PARAMS['conditions'], 
        time_vector= neural_data.time.data,
        areas=['LPFC', 'MCC'])
    
    for target in PARAMS['conditions']:
        for area in ['LPFC', 'MCC']:
            print(f"Decoding {target} in {area}")
            #try:
            neural_data_temp = neural_data.where(neural_data.area == area, drop=True)
            if len(neural_data_temp.unit) == 0:
                continue

            # 1. preprocess data -> create a matrix of regressors (X) and a vector of labels (y)

            X, y, groups = _preproc_data(neural_data_temp.firing_rates, target, PARAMS['group_target'])

            # 2. run decoder on the labelled dataset
            res = linear_decoding(X, y, groups, K_fold=PARAMS['K_fold'], n_perm=PARAMS['n_perm'], n_jobs=n_jobs)

            scores, pvals, perm_mean, perm_std = res#, cv_mean, cv_std = res

            # create xarray dataset, dimension is time, vars are scores
            xr_scores.scores.loc[target, :, area] = scores
            xr_scores.pvals.loc[target, :, area] = pvals
            xr_scores.perm_mean.loc[target, :, area] = perm_mean
            xr_scores.perm_std.loc[target, :, area] = perm_std
            #xr_scores.cv_mean.loc[target, :, area] = cv_mean
            #xr_scores.cv_std.loc[target, :, area] = cv_std

            """except Exception as e:
                print(f"Error decoding {target} in {area}: {e}")"""

    # add monkey and session information
    xr_scores = xr_scores.assign_coords(session=f'{monkey}_{session}')
    xr_scores = xr_scores.expand_dims('session')
    xr_scores = xr_scores.assign_coords(monkey=('session', [monkey]))
    for key, value in PARAMS.items():
        xr_scores.attrs[key] = str(value)

    session_log=[]  # Log messages of the internal run, not implemented yet
    return xr_scores, session_log     


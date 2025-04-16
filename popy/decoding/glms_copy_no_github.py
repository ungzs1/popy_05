import sys

sys.path.append("C:\ZSOMBI\OneDrive\PoPy")
sys.path.append("/Users/zsombi/OneDrive/PoPy")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import xarray as xr
import warnings

from popy.io_tools import load_behavior, load_neural_data
from popy.behavior_data_tools import add_value_function, add_switch_info
from popy.neural_data_tools import add_firing_rates, remove_trunctuated_neurons, remove_low_fr_neurons, remove_drift_neurons, remove_low_varance_neurons, set_off_recording_times_to_nan
from popy.neural_data_tools import time_normalize_session
from popy.decoding.population_decoders import build_dataset

from scipy.stats import permutation_test
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn import linear_model
from sklearn.metrics import mean_poisson_deviance


def statistics(a, b):
    return np.corrcoef(a, b)[0, 1]

def get_p_val(coeffs_perm, coeff_true):
    p_vals_pos = np.sum(coeffs_perm >=  coeff_true)
    p_vals_neg = np.sum(coeffs_perm <=  coeff_true)
    # get smaller p-value
    p_val = np.min([p_vals_pos, p_vals_neg])
    p_val = p_val / len(coeffs_perm)

    return p_val

def fit_eval_glm(y, X):
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    # get coefficients (z-values)
    coeffs = model.params
    tvals = model.tvalues
    
    # predict
    y_pred = model.predict(X)
    D = 1 - (mean_poisson_deviance(y, y_pred) / mean_poisson_deviance(y, np.asarray(y).mean() * np.ones_like(y)))  # get the coefficient of determination?

    return coeffs, D

def get_CPD(y_true, X_full, X_reduced): 
    # estimate D2 score for the model (full vs reduced)
    model_full = sm.GLM(y_true, X_full, family=sm.families.Poisson()).fit()
    model_reduced = sm.GLM(y_true, X_reduced, family=sm.families.Poisson()).fit()

    # predict
    y_pred_full = model_full.predict(X_full)
    y_pred_reduced = model_reduced.predict(X_reduced)

    CPD = 1 - (mean_poisson_deviance(y_true, y_pred_full) / mean_poisson_deviance(y_true, y_pred_reduced))

    return CPD


class SingleUnitAnalysis:
    def __init__(self, monkey, session, area=None):
        """
        endog: count data, 1D array, shape (n_trials)
        exog: predictors, 2D array, shape (n_trials, n_predictors)
        """
        #self.bin_size = .010  # in seconds
        self.step_time = .100  # in seconds, how much we want to slide the glm window

        self.monkey = monkey
        self.session = session
        self.area = area  # area of interest, e.g. 'MCC', 'ofc', 'LPFC'

        # glm parameters
        #self.linear_predictors = None  # condition of interest, e.g. ['target', 'feedback', 'value_function]
        #self.anova_predictors = None  # condition of interest, e.g. ['target', 'feedback', 'value_function]
        self.glm_predictors = [None]  # condition of interest, e.g. ['target', 'feedback', 'value_function]
        self.cpd_predictors = None
        self.cpd_targets = [None]  # target predictors for the CPD test, in a list, e.g. ['feedback', 'value_function']
        
        self.target_of_interest = None
        
        self.model = None  # model to run, 'linear_correlation' or 'anova' or 'poisson_glm' or 'value_glm'
        self.model_scoring_function = None  # measure of interest, e.g. 'D2', 'coeffs', 'tvals', 'p_vals'
        
        self.neural_data_type = None  # type of neural data, 'spike_counts' or 'firing_rates'
        self.value_type = None  # type of value function, 'continuous' or 'discrete'
        self.n_extra_trials = -1  # number of extra trials to add to the dataset
        
        ###self.glm_results = {'coeffs': None, 'tvals': None, 'p_vals': None}

        self.results = None  # save results
        self.log = []  # write log info in human readable format, e.g. errors, etc.


    def _get_time_vector(self, neural_dataset):
        time_vector_original = neural_dataset.time.data  # original time vector
        bin_size = time_vector_original[1] - time_vector_original[0]  # bin size in seconds
        step_time = self.step_time  # step time in seconds
        step_bins = int(step_time / bin_size)  # number of bins to step
        
        # create new time vector with step time
        return time_vector_original[::step_bins]


    def _create_results_xr(self, unit_ids, time_vector, scoring, p_vals):
        # create xarray with unit_ids as unit dimension and time_vector as time dimension
        attrs = self.__dict__.copy()
        # remove 'results', 'monkey', 'session' keys from the attributes...
        for key in ['results', 'monkey', 'session']:
            if key in attrs.keys():
                attrs.pop(key)
            
        data_xr = xr.Dataset(
            data_vars=dict(
                scores=(['unit', 'time'], scoring),
                p_vals=(['unit', 'time'], p_vals),
            ),
            coords=dict(
                unit= [f'{self.monkey}_{self.session}_{unit_id}' for unit_id in unit_ids],  # save unit ids
                time=time_vector,
                monkey=("unit", [self.monkey for _ in range(len(unit_ids))]),
                session=("unit", [self.session for _ in range(len(unit_ids))]),
                area=("unit", [unit_id.split('_')[0] for unit_id in unit_ids]),
                channel=("unit", [unit_id.split('_')[1] for unit_id in unit_ids]),
            ),
            attrs=attrs
        )
        
        for key, value in data_xr.attrs.items():
            data_xr.attrs[key] = str(value)
        
        return data_xr
    
        
    def measure_modulation(self, y, predictors):
        a = y
        b = predictors[self.linear_predictors].values.squeeze()
        
        res = permutation_test((a, b), statistics, n_resamples=1000)

        return res.statistic, res.pvalue


    def anova(self, y, predictors):
        x = predictors[self.anova_predictors].values.squeeze()
        datas = {value_level: y[x == value_level] for value_level in np.unique(x)}

        f_val, p_val = stats.f_oneway(*datas.values())

        return f_val, p_val
    

    @staticmethod
    def permutation_glm(y, X):
        # Prepare data
        X = sm.add_constant(X)
        
        # Fit the true model
        coeffs_ture, D_true = fit_eval_glm(y, X)
        
        # Run permutation test - fit shuffled models N times
        n_perm = 1000
        coeffs_perm = pd.DataFrame(index=X.columns, columns=range(n_perm))  # create pandas df to store coefficients: indices are columns of X, cols are the permuted coefficients per iteration
        D_perm = np.empty(n_perm)  # create empty array to store D2 values for permuted models
        for i in range(n_perm):
            X_shuffled = X.sample(frac=1)  # shuffle labels
            coeffs_perm_temp, D_perm_temp = fit_eval_glm(y, X_shuffled)
            coeffs_perm[i] = coeffs_perm_temp
            D_perm[i] = D_perm_temp

        # get the number of times the permuted coefficients are larger than the true coefficient and if smaller, then select the smaller one
        coeffs_pvals = pd.Series([get_p_val(coeffs_perm.loc[idx], coeffs_ture[idx]) for idx in coeffs_perm.index], index=coeffs_perm.index)
        D_pval = get_p_val(D_perm, D_true)  # TODO: extend this line to multiple predictors

        return coeffs_ture, coeffs_pvals, D_true, D_pval
    
    @staticmethod
    def permutation_glm_CPD(y_true, X_full, target_predictors):
        coeffs_ture = pd.Series(index=target_predictors)
        coeffs_pvals = pd.Series(index=target_predictors)
        
        for target_predictor in target_predictors:
            # Prepare data 
            X_reduced = X_full.drop(target_predictor, axis=1)  # drop target predictor from the full model
            X_full, X_reduced = sm.add_constant(X_full), sm.add_constant(X_reduced)  # add constant
            
            # Fit the true model, i.e. get cpd for the target predictor
            CPD_true = get_CPD(y_true, X_full, X_reduced)
            
            # Permutation test - fit shuffled models N times
            n_perm = 1000
            CPD_perm = np.zeros(n_perm)
            for i in range(n_perm):
                y_perm = np.random.permutation(y_true)
                CPD_perm[i] = get_CPD(y_perm, X_full, X_reduced)
                
            # save results
            coeffs_ture[target_predictor] = CPD_true
            coeffs_pvals[target_predictor] = np.sum(CPD_perm > CPD_true) / n_perm
        
        return coeffs_ture, coeffs_pvals
    
    
    def load_data_for_glm(self):
        """
        First load the all-session behav info.
            - X: behav info, shape of (n_trials)
            - y: spike count data, shape of (n_trials, n_unis, n_timebins)

        Then create dataset.
        """
        # 1. Behavior data
        session_data = load_behavior(self.monkey, self.session)
        if not self.value_type is None:
            if self.value_type == 'discrete':
                session_data = add_value_function(session_data, monkey=self.monkey, digitize=True)
            elif self.value_type == 'continuous':
                session_data = add_value_function(session_data, monkey=self.monkey, digitize=False)

        # add more info
        session_data = add_switch_info(session_data)
        
        '''session_data['previous_choice_at_switch'] = [session_data['target'].values[i-1] if session_data['switch'].values[i] else np.nan for i in range(len(session_data))]
        session_data['upcoming_choice_at_switch'] = [session_data['target'].values[i] if session_data['switch'].values[i] else np.nan for i in range(len(session_data))]
        
        session_data['last_fb'] = session_data['feedback'].shift(1)
        session_data['last_fb_cmw'] = ((session_data['target'].values == session_data['target'].shift(1).values).astype(int) -.5) * 2  # if target is the same as the previous one, 1, else -1
        session_data['last_fb_cmw_mixed'] = session_data['last_fb_cmw'] * session_data['last_fb']
        
        session_data['last_last_fb'] = session_data['feedback'].shift(2)
        session_data['last_last_fb_cmw'] = ((session_data['target'].values == session_data['target'].shift(2).values).astype(int) -.5) * 2
        session_data['last_last_fb_cmw_mixed'] = session_data['last_last_fb_cmw'] * session_data['last_last_fb']
        
        session_data['last_last_last_fb'] = session_data['feedback'].shift(3)
        session_data['last_last_last_fb_cmw'] = ((session_data['target'].values == session_data['target'].shift(3).values).astype(int) -.5) * 2
        session_data['last_last_last_fb_cmw_mixed'] = session_data['last_last_last_fb_cmw'] * session_data['last_last_last_fb']
        
        session_data['last_last_last_last_fb'] = session_data['feedback'].shift(4)
        session_data['last_last_last_last_fb_cmw'] = ((session_data['target'].values == session_data['target'].shift(4).values).astype(int) -.5) * 2
        session_data['last_last_last_last_fb_cmw_mixed'] = session_data['last_last_last_last_fb_cmw'] * session_data['last_last_last_last_fb']'''

        session_data = session_data.dropna()

        # set datatypes
        '''session_data['feedback'] = session_data['feedback'].astype(int)
        session_data['last_fb'] = session_data['last_fb'].astype(int)
        session_data['last_fb_cmw'] = session_data['last_fb_cmw'].astype(int)
        session_data['last_fb_cmw_mixed'] = session_data['last_fb_cmw_mixed'].astype(int)
        session_data['last_last_fb'] = session_data['last_last_fb'].astype(int)
        session_data['last_last_fb_cmw'] = session_data['last_last_fb_cmw'].astype(int)
        session_data['last_last_fb_cmw_mixed'] = session_data['last_last_fb_cmw_mixed'].astype(int)
        session_data['last_last_last_fb'] = session_data['last_last_last_fb'].astype(int)
        session_data['last_last_last_fb_cmw'] = session_data['last_last_last_fb_cmw'].astype(int)
        session_data['last_last_last_fb_cmw_mixed'] = session_data['last_last_last_fb_cmw_mixed'].astype(int)
        session_data['last_last_last_last_fb'] = session_data['last_last_last_last_fb'].astype(int)
        session_data['last_last_last_last_fb_cmw'] = session_data['last_last_last_last_fb_cmw'].astype(int)
        session_data['last_last_last_last_fb_cmw_mixed'] = session_data['last_last_last_last_fb_cmw_mixed'].astype(int)'''
                
        # remove all but target 1
        if not self.target_of_interest is None:
            if self.target_of_interest == 'random':
                # remove random 2/3rd of the trials
                ids_to_keep = np.random.choice(session_data.index, int(len(session_data) /3), replace=False)
                # sort ids to keep
                ids_to_keep = np.sort(ids_to_keep)
                # raise warning
                warnings.warn(f'leaving {len(ids_to_keep)}/{len(session_data)} trials!!!!')
                session_data = session_data.loc[ids_to_keep]
            else:
                warnings.warn(f'removing all but target {self.target_of_interest}, {np.sum(session_data["target"].values == self.target_of_interest)}\{len(session_data)}')
                session_data = session_data[session_data['target'] == self.target_of_interest]

        # 2. Neural data
        neural_data = load_neural_data(self.monkey, self.session, mode='spikes', return_dataset_format=True)
        
        # convert neural data to firing rates or binned spike counts
        if self.neural_data_type == 'spike_counts':
            neural_data = add_firing_rates(neural_data, method='count', win_len=0.200)
        elif self.neural_data_type == 'firing_rates':            
            neural_data = add_firing_rates(neural_data, method='gauss', win_len=0.050)
        else:
            raise ValueError('neural_data_type must be "spike_counts" or "firing_rates"')
        
        # filter neurons by area
        if self.area is not None:
            neural_data = neural_data.sel(unit=neu
        
        # neural dataset preprocessing
        neural_data = remove_trunctuated_neurons(neural_data, mode='set_nan', delay_limit=0)  # set off-recording neurons to nan
        # todo: high dispersion neurons should be handled here!!!
        #neural_data = remove_low_varance_neurons(neural_data, var_limit=.2, print_usr_msg=True)  # remove neurons with low variance
        #neural_data = remove_drift_neurons(neural_data, corr_limit=.2)  # remove neurons with drift
        #neural_data = remove_low_fr_neurons(neural_data, 1)  # remove low_firing units
        neural_data = neural_data.drop('spike_trains')  # remove spike_trains, as we only need firing rates or binnedspike counts
        
        # build dataset
        if self.neural_data_type == 'firing_rates':
            neural_data = time_normalize_session(neural_data)
            neural_dataset, predictors = build_dataset(neural_data, session_data, n_extra_trials=self.n_extra_trials)
        elif self.neural_data_type == 'spike_counts':
            # center on feedback pm 2sec
            neural_dataset, predictors = build_dataset(neural_data, session_data, n_extra_trials=None, center_on='feedback', center_window=[-1, 4])


        return neural_dataset, predictors  # save predictors


    def run(self, print_log=False):
        # print log
        print(f'Running {self.model} for {self.monkey} - {self.session}...')
        
        ## init data - the loading method should be customized in a child class to return the dataset and predictors of interest - but in a general way
        neural_dataset, predictors = self.load_data_for_glm()  # load data, inits self.neural_dataset and self.predictors

        # time vector, considering the step time
        time_vector = self._get_time_vector(neural_dataset)
        
        # init results container
        scoring = np.full((len(neural_dataset.unit), len(time_vector)), np.nan) 
        p_vals = np.full((len(neural_dataset.unit), len(time_vector)), np.nan)
        n_predictors = len(self.glm_predictors)  # number of predictors TODO: extend to other models as well
        coeffs = np.full((len(neural_dataset.unit), len(time_vector), n_predictors+1), np.nan)
        coeffs_pvals = np.full((len(neural_dataset.unit), len(time_vector), n_predictors+1), np.nan)
        
        for unit_id, unit_name in enumerate(neural_dataset.unit.data):  # loop over units
            if print_log: print(f'{unit_id}/{len(neural_dataset.unit.data)} - Running {unit_name}...')
            for t, time in enumerate(time_vector):  # loop over time bins
                y = neural_dataset[self.neural_data_type].sel(unit=unit_name, time=time).data  # get spike counts (or firing rates) for given unit and time bin
                
                # drop nan values
                non_nan_idx = ~np.isnan(y)
                y = y[non_nan_idx]  # remove nan values
                predictors_temp = predictors[non_nan_idx]  # remove nan values
                
                # skip if theres data neural dta for less than 50 trials 
                if len(y) < 50:
                    scoring[unit_id, t] = np.nan

                '''if self.model == 'linear_correlation':    
                    # Mesure 1: get correlation and p-value with permutation test (linear correlation)
                    coeff_temp, pval_temp= self.measure_modulation(y, predictors_temp)
                    tval_temp = coeff_temp / np.std(y)
                    scoring[unit_id, t] = tval_temp
                    p_vals[unit_id, t] = pval_temp

                elif self.model == 'anova':
                    try:
                        # Measure 2: run ANOVA
                        f_val_temp, pval_temp = self.anova(y, predictors_temp)
                        scoring[unit_id, t] = f_val_temp
                        p_vals[unit_id, t] = pval_temp
                    except:
                        scoring[unit_id, t] = np.nan
                        p_vals[unit_id, t] = np.nan'''
                                        
                if self.model == 'glm':
                    # Measure 3: fit glm and run permutation test
                    try:
                        predictors_temp = predictors_temp[self.glm_predictors]  # select predictors of interest
                        coeffs_temp, coeffs_pvals_temp, D, D_pval = self.permutation_glm(y, predictors_temp)  # run permutation test
                        
                        scoring[unit_id, t] = D
                        p_vals[unit_id, t] = D_pval
                        coeffs[unit_id, t, :] = coeffs_temp.values
                        coeffs_pvals[unit_id, t, :] = coeffs_pvals_temp.values
                    except:
                        self.log.append(f'Error in fitting GLM for {unit_name} at time {time}.')
                    
                elif self.model == 'glm_cpd':
                    # Measure 4: run D2 test to see value against feedback
                    try:
                        predictors_temp = predictors_temp[self.cpd_predictors]  # select predictors of interest for the full model
                        coeffs_temp, coeffs_pvals_temp = self.permutation_glm_CPD(y, predictors_temp, self.cpd_targets)  # run permutation test

                        coeffs[unit_id, t, :] = coeffs_temp.values
                        coeffs_pvals[unit_id, t, :] = coeffs_pvals_temp.values
                    except:
                        self.log.append(f'Error in fitting GLM for {unit_name} at time {time}.')
                        
        # create results xarray
        self.results = self._create_results_xr(unit_ids=neural_dataset.unit.values,
                                               time_vector=time_vector,
                                               scoring=scoring,
                                               p_vals=p_vals)
        # add coeffs one by one
        aaa = self.glm_predictors if self.model == 'glm' else self.cpd_targets  # TODO
        for i, predictor in enumerate(aaa):
            self.results[f'coeffs_{predictor}'] = (['unit', 'time'], coeffs[:, :, i])
            self.results[f'p_vals_{predictor}'] = (['unit', 'time'], coeffs_pvals[:, :, i])

        return self.results



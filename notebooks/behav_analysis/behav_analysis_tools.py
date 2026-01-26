import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score

### data analysis notebook helpers and tools




def predict_switch(X, y, measure='accuracy', cv_fold=10, return_weights=False):
    """
    Predicts switch based on history of feedback.
    
    """
    # convert y to 0 and 1
    X = X.astype(int)
    y = y.astype(int)

    # shuffle data to avoid bias
    X, y = shuffle(X, y)

    # balance classes by undersampling majority class
    X, y = RandomUnderSampler().fit_resample(X, y)

    # logistic regression, 5 fold cross validation
    clf = LogisticRegression()

    if measure == 'accuracy':
        if cv_fold == 1:
            clf.fit(X, y)
            scores = clf.score(X, y)
            weights = clf.coef_
            if return_weights:
                return scores, weights
            else:
                return scores
        else:
            if return_weights:
                return cross_val_score(clf, X, y, cv=cv_fold, scoring='accuracy'), clf.fit(X, y).coef_
            else:
                return cross_val_score(clf, X, y, cv=cv_fold)
    elif measure == 'roc_curve':
        clf.fit(X, y)
        # Generate predictions on the test set
        y_pred_proba = clf.predict_proba(X)[:, 1]

        # Calculate the false positive rate, true positive rate, and threshold values
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

        # Calculate the area under the ROC curve (AUC)
        auc = roc_auc_score(y, y_pred_proba)

        # interpolate fpr and tpr to have the same length
        new_length = 100
        fpr = np.interp(np.linspace(0, len(fpr) - 1, new_length), np.arange(len(fpr)), fpr)
        tpr = np.interp(np.linspace(0, len(tpr) - 1, new_length), np.arange(len(tpr)), tpr)

        return fpr, tpr, auc


def effect_of_rewards(session_data, num_max_trials, measure='accuracy', cv_fold=10):
    """
    Calculate the effect of rewards on a given session data.

    Args:
        session_data (DataFrame): The session data containing the history of feedback and switch values.
        num_max_trials (int): The maximum number of trials to consider.
        measure (str, optional): The performance measure to calculate. Defaults to 'accuracy'.
        cv_fold (int, optional): The number of cross-validation folds. Defaults to 10.

    Returns:
        dict: A dictionary containing the calculated scores for each performance measure.
    """
    if 'history_of_feedback' not in session_data.columns:
        raise ValueError("session_data must contain a column named 'history_of_feedback'")
    if 'switch' not in session_data.columns:
        raise ValueError("session_data must contain a column named 'switch'")

    all_scores = {'accuracy': [], 'fpr': [], 'tpr': [], 'auc': []}

    for num_trials in range(0, num_max_trials):
        X = session_data.history_of_feedback.values
        X = np.vstack(X)  # convert array of arrays to 2d array
        X = X[:, :num_trials+1]  # take only the last num_trials columns
        y = session_data.switch.values

        if measure == 'roc_curve':
            fpr, tpr, auc = predict_switch(X, y, cv_fold=cv_fold, measure=measure)
            all_scores['fpr'].append(fpr)
            all_scores['tpr'].append(tpr)
            all_scores['auc'].append(auc)
        elif measure == 'accuracy':
            all_scores['accuracy'].append(predict_switch(X, y, cv_fold=cv_fold, measure=measure))

    # convert all lists to numpy arrays
    return {key: np.array(val) for key, val in all_scores.items()}


def plot_roc_auc(scores, title=None):
    fprs = scores['fpr']
    tprs = scores['tpr']

    # plot ROC curves
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax = axs[0]
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        ax.plot(fpr, tpr, color='gray', alpha=(i+1)*(1/len(fprs)))
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')

    ax = axs[1]
    scores_auc = scores['auc']
    ax.plot(np.arange(1, len(scores_auc)+1), scores_auc, color='black', marker='o', label='mean (of n-fold CV))')
    ax.set_ylabel('ROC-AUC score (%)')
    ax.set_xlabel('Number of past outcomes used as predictors')
    ax.grid()
    ax.set_ylim([0.49, 1.01])

    if title is not None:
        fig.suptitle(title)

    plt.show()


def plot_scores(scores, title=None):
    # plot ROC curves
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    scores_auc = scores['accuracy']
    ax.plot(np.arange(1, len(scores_auc)+1), np.mean(scores_auc, axis=1), color='black', marker='o', label='mean (of n-fold CV))')
    ax.fill_between(np.arange(1, len(scores_auc)+1), np.mean(scores_auc, axis=1) - np.std(scores_auc, axis=1), np.mean(scores_auc, axis=1) + np.std(scores_auc, axis=1), alpha=0.2, color='gray')

    ax.set_ylabel('Accuracy (cross-val) (%)')
    ax.set_xlabel('Number of past outcomes used as predictors')
    ax.grid()
    ax.set_ylim([0.49, 1.01])

    if title is not None:
        ax.set_title(title)

    plt.show()

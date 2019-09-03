import numpy as np
import pandas as pd
import tqdm, datetime, time, logging
import shap
from tqdm import tqdm_notebook
from numba import jit
import os, gc, tqdm
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, KFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, plot_importance
from utils import time_split, load_merge_data, gen_sub, run_permutation_importance, save_lgb_importances

def covariate_shift_lgbm(train, test, cols_to_drop, NFOLDS, LGB_PARAMS, PARAMS, whitelist):
    '''
    concat train+test (balanced), create origin labels, get AUC metric and OOF predictions
    '''
    train, test = train.copy(), test.copy()
    train = train.drop('isFraud', axis=1)
    train['origin'] = 0
    test['origin'] = 1
    X = pd.concat([train, test], axis=0)
    y = X['origin'].values
    X = X.astype(dict(test.dtypes)).drop(cols_to_drop, axis=1)

    if whitelist:
        X = X.drop([c for c in X.columns if c not in whitelist], axis=1)

    kf = KFold(n_splits=NFOLDS, shuffle=True)
    folds = [(a, b) for a, b in kf.split(X)]
    y_oof = np.zeros(shape=X.shape[0])
    fold_aucs, bsts, bsts_folds = [], [], []
    for i, (train_ixs, val_ixs) in enumerate(folds):
        X_train, X_val, y_train, y_val = X.iloc[train_ixs], X.iloc[val_ixs], y[train_ixs], y[val_ixs]

        bst = LGBMClassifier(**LGB_PARAMS)
        bst.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric=['auc'],
                verbose=PARAMS['verbose'], early_stopping_rounds=PARAMS['early_stopping_rounds']
                )

        y_val_pred = bst.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_true=y_val, y_score=y_val_pred)
        print(f'*** Fold {i:d} AUC : {val_score:.4f} ***')

        y_oof[val_ixs] = y_val_pred

        bsts.append(bst)
        bsts_folds.append((train_ixs, val_ixs))
        fold_aucs.append(val_score)

    feat_names = X.columns
    return y_oof, fold_aucs, bsts, bsts_folds, feat_names, X
def run_covariate_shift_lgbm(path):
    '''
    :param path: path to save df containg covariate shift feature importance
    :return:
    '''

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    train = load_merge_data('train')
    test = load_merge_data('test')

    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 64,
        'min_child_samples': 200,
        'n_estimators': 200,
        'n_jobs': -1,
        'scale_pos_weight': train.shape[0] / test.shape[0]
    }
    PARAMS = {
        'verbose': 25,
        'early_stopping_rounds': 100,
    }
    cols_to_drop = ['TransactionID', 'TransactionDT', 'origin']

    y_oof_cov, fold_aucs_cov, bsts_cov, bsts_folds_cov, feat_names_cov, X_cov = covariate_shift_lgbm(
        train, test,
        cols_to_drop,
        NFOLDS=3,
        LGB_PARAMS=LGB_PARAMS, PARAMS=PARAMS,
        whitelist=[],
    )
    print(pd.Series(fold_aucs_cov).describe())

    # Save run info
    imps_cov = np.mean(np.vstack([bst.booster_.feature_importance('gain') for bst in bsts_cov]), axis=0)
    df_cov = pd.DataFrame(imps_cov, index=feat_names_cov, columns=['Importance']).sort_values(
        by='Importance', ascending=False).reset_index()
    df_cov.to_hdf(f'{path}/covariate_imps_{timestamp}.h5', mode='w', key='cov')
    np.save(f'{path}/covariate_weights_{timestamp}', y_oof_cov[:train.shape[0]]) # save train cov weights

if __name__ == '__main__':
    run_covariate_shift_lgbm(path='./covariate_shift_outputs')
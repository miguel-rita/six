import numpy as np
import pandas as pd
import tqdm, datetime, time, logging
from numba import jit
import os, gc, tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, plot_importance
from utils import time_split, load_merge_data, gen_sub, run_permutation_importance, save_lgb_importances

def train_predict_lgbm(train, test, cols_to_drop, NFOLDS, LGB_PARAMS, PARAMS, whitelist):
    y = train.isFraud.values
    X = train.drop(cols_to_drop, axis=1)
    X_test = test.drop([c for c in cols_to_drop if c != 'isFraud'], axis=1)

    if whitelist:
        X = X.drop([c for c in X.columns if c not in whitelist], axis=1)
        X_test = X_test.drop([c for c in X_test.columns if c not in whitelist], axis=1)

    folds = time_split(X, nfolds=NFOLDS)
    y_pred = np.zeros(shape=(test.shape[0], NFOLDS))
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
        y_pred[:, i] = bst.predict_proba(X_test)[:, 1]

        bsts.append(bst)
        bsts_folds.append((train_ixs, val_ixs))
        fold_aucs.append(val_score)

    # TODO: Review mean of folds- median, weighted, etc
    y_pred = np.mean(y_pred, axis=1)
    feat_names = X.columns
    return y_pred, fold_aucs, bsts, bsts_folds, feat_names, X

def stack_one():
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logging.basicConfig(filename=f'logs/run_{timestamp}.log', filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.DEBUG,
    )

    featlist = [
        # 'hwindow_feats_v2'
    ]
    wlist = [list(pd.read_hdf(f'features/{fl}_train.h5').columns) for fl in featlist]
    train = load_merge_data('train', features=featlist)
    test = load_merge_data('test', features=featlist)

    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 32,
        'min_child_samples': 500,
        'n_estimators': 2000,
        'n_jobs': -1,
    }
    PARAMS = {
        'verbose':100,
        'early_stopping_rounds':100,
    }
    cols_to_drop = ['TransactionID', 'isFraud', 'TransactionDT']

    logging.info(f'LGB parameters:')
    for k,v in LGB_PARAMS.items(): logging.info(f'     {k}:{v}')
    logging.info(f'Other parameters:')
    for k, v in PARAMS.items(): logging.info(f'     {k}:{v}')

    BFEATS = list(pd.read_hdf('perm_imps/pimps_2019-09-01 20:08:59.h5').index)[:70]

    y_pred, fold_aucs, bsts, bsts_folds, feat_names, X = train_predict_lgbm(train, test,
        cols_to_drop, NFOLDS=5, LGB_PARAMS=LGB_PARAMS, PARAMS=PARAMS,
        whitelist=BFEATS,#+wlist[0],
    )
    print('CV:', pd.Series(fold_aucs).describe())
    logging.info(pd.Series(fold_aucs).describe())
    logging.info(feat_names)


    save_lgb_importances(bsts=bsts, feat_names=feat_names, path='./lgb_imps', imptype='split', ts=timestamp)
    # run_permutation_importance(X, train['isFraud'].values, bsts, bsts_folds, bst_n=-1, ts=timestamp)
    gen_sub(y_pred, subname=f'meanCV{np.mean(fold_aucs):.4f}_{timestamp}')

if __name__ == '__main__':
    stack_one()
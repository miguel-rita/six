import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns, tqdm

def merge_base_df(name):
    '''
    Run once per project
    :param name:
    :return:
    '''
    p = './data/'
    print(f'Reading {name} source dfs ...')
    i, t = pd.read_csv(f'{p}/{name}_identity.csv'), pd.read_csv(f'{p}/{name}_transaction.csv')
    print(f'Merging {name} source dfs ...')
    d = t.merge(i, on='TransactionID', how='left')

    # Set categorical dtype and other casts to reduce mem
    print(f'Setting {name} df dtypes and categoricals ...')
    dd = {k: 'float32' for k, v in dict(d.dtypes).items() if v == 'float64'}
    cat_feats = ['ProductCD', 'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2', 'DeviceType', 'DeviceInfo'] + \
                [f'card{i:d}' for i in np.arange(1, 7)] + [f'M{i:d}' for i in np.arange(1, 10)] + [f'id_{i:d}' for i in
                                                                                                   np.arange(12, 39)]
    for cf in cat_feats:
        dd[cf] = 'category'
    d = d.astype(dd)
    d.to_hdf(f'data/{name}.h5', mode='w', key=name, format='table')

def load_merge_data(name, features=None):
    p = './data/'
    print(f'Reading {name} csv ...')
    d = pd.read_hdf(f'{p}/{name}.h5')

    if features is not None: # Merge additional features
        print(f'Merging additional features to {name} df ...')
        fs = [pd.read_hdf(f'features/{f}_{name}.h5') for f in features]
        for f in tqdm.tqdm(fs, total=len(fs)): d = d.merge(f, on='TransactionID', how='left')

    print(f'... Done loading {name} df')
    return d.astype({'TransactionID':str})

def time_split(X, nfolds):
    kf = KFold(n_splits=nfolds, shuffle=False)
    folds=[(a,b) for a,b in kf.split(X)]
    return folds

def permutation_importance(bst, X_val, y_val):
    # Baseline score
    METRIC = 'AUC'
    N = X_val.shape[1]
    preds = bst.predict_proba(X_val)[:, 1]
    base_metric = roc_auc_score(y_val, preds)
    print(f'Num of original features : {N:d} with {METRIC} = {base_metric:.4f}')
    imps = []

    # Feature permutation scores
    for i, feat in enumerate(X_val.columns):
        X_ = X_val.copy()
        prev_type = X_[feat].dtype
        X_[feat] = pd.Series(np.random.permutation(X_[feat].values), dtype=prev_type)
        p = bst.predict_proba(X_)[:, 1]
        col_metric = roc_auc_score(y_val, p)
        pimp = base_metric - col_metric
        imps.append(pimp)
        print(f'* {feat} permutation importance = {pimp:.6f} *')

    # Scores keeping only top x, y, ... features
    pimps = pd.DataFrame(np.array(imps), index=X_val.columns, columns=['pimp']).sort_values(
        by='pimp', ascending=False)

    return pimps

def run_permutation_importance(X, y_tgt, bsts, bsts_folds, bst_n, ts):
    '''
    Run perm imp algorithm, and save pimps to disk
    :param bsts: list of fitted bsts
    :param bsts_folds: list containing cv fold ixs
    :param bst_n: num. of bst used for pimp calc
    :param ts: timestamp
    :return:
    '''
    #TODO - validate function
    val_ixs = bsts_folds[bst_n][1]
    X_val = X.iloc[val_ixs, :]
    y_val = y_tgt[val_ixs]
    permutation_importance(bsts[bst_n], X_val, y_val).to_hdf(f'./perm_imps/pimps_{ts}.h5', mode='w', key='pimps')

def load_top_pimps(full_path, t):
    '''
    return list of feats with pimp above t
    :param full_path:
    :param t:
    :return:
    '''
    pimps = pd.read_hdf(full_path).reset_index()
    return list(pimps.loc[pimps.pimp > t, 'index'])

def gen_sub(y_pred, subname):
    sub = pd.read_csv(f'./data/sample_submission.csv').astype({'isFraud':np.float32})
    sub['isFraud'] = y_pred
    sub.to_csv(f'./subs/{subname}.csv', float_format='%.5f', index=False)

def save_lgb_importances(bsts, feat_names, path, imptype, ts):
    c = f'Importance_{imptype}'
    imps = np.mean(np.vstack([bst.booster_.feature_importance(imptype) for bst in bsts]), axis=0)
    df = pd.DataFrame(imps, index=feat_names, columns=[c]).sort_values(
        by=c,ascending=False).reset_index()
    f, a = plt.subplots(1, 1, figsize=(10, len(feat_names) * 0.1))
    p = sns.barplot(x=c, y='index', data=df, ax=a)
    plt.tight_layout()
    p.figure.savefig(f'{path}/imps_{imptype}_{ts}.png')

if __name__ == '__main__':
    merge_base_df('train')
    merge_base_df('test')
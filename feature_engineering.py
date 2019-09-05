import numpy as np
import pandas as pd
import tqdm, datetime, time, logging
from numba import jit
import os, gc, tqdm, functools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, plot_importance
from utils import time_split, load_merge_data, gen_sub, run_permutation_importance, save_lgb_importances

def gen_lookups(df, gb_col):
    '''
    return a cumsum array starting at 0 and a dict to map gb_col to its positions
    '''
    ss = df.groupby(gb_col).size().cumsum()
    ssx = np.zeros(len(ss)+1).astype(int)
    ssx[1:] = ss.values
    ssdict = {}
    for i, col in enumerate(ss.index):
        ssdict[col] = i
    return ssx, ssdict

def build_hashes_df(df, col_combs):
    df_hashes = pd.DataFrame()
    df_hashes['TransactionID'] = df['TransactionID'].copy()
    names = [''.join(c) for c in col_combs]
    for h, name in zip(col_combs, names) :
        hs = [df[c].astype(str) for c in h]
        df_hashes[name] = functools.reduce(lambda a,b : a+b, hs)
    return df_hashes.astype(str)
def run_hash_target_encoding(path, featset_name):

    train = load_merge_data('train')#.sample(n=2000)
    test = load_merge_data('test')#.sample(n=2000)

    cs = [
        ['card1'],
        ['card1', 'addr1'],
        ['card1', 'P_emaildomain'],
        ['card1', 'R_emaildomain'],
        [f'card{i:d}' for i in range(1,7)],
        [f'card{i:d}' for i in range(1,7)] + ['addr1','addr2'],
    ]
    cs_ = [c for c in cs if len(c)>1]
    names = [''.join(c) for c in cs]
    train_hashes = build_hashes_df(train, cs_)
    test_hashes = build_hashes_df(test, cs_)

    train = train.merge(train_hashes, on='TransactionID', how='left').loc[:, ['TransactionID', 'isFraud']+names].astype({c:str for c in names})
    test = test.merge(test_hashes, on='TransactionID', how='left').loc[:, ['TransactionID', 'isFraud']+names].astype({c:str for c in names})
    folds = time_split(train, nfolds=5)

    for h in tqdm.tqdm(names):
        # build placeholder cols
        for p in ['ct_', 'freq_']:
            train[p+h] = np.zeros(train.shape[0])
            test[p+h] = np.zeros(test.shape[0])
        # fill per fold to avoid leakage
        for ts, vs in folds:
            s = train.iloc[ts,:].groupby(h)['isFraud'].agg(['sum', 'count']).reset_index()
            s['freq'] = 0
            s['freq'].loc[s['sum']>0] = s.loc[s['sum']>0]['sum']/s.loc[s['sum']>0]['count']
            ct_encode = dict(zip(s[h], s['sum']))
            freq_encode = dict(zip(s[h], s['freq']))

            ce = lambda v : ct_encode[str(v)] if str(v) in ct_encode.keys() else np.nan
            fe = lambda v : freq_encode[str(v)] if str(v) in freq_encode.keys() else np.nan
            
            raw = train[h].iloc[vs]
            train['ct_'+h].values[vs] = raw.apply(ce).values
            train['freq_'+h].values[vs] = raw.apply(fe).values
            raw = test[h].iloc[vs]
            test['ct_'+h].values[vs] = raw.apply(ce).values
            test['freq_'+h].values[vs] = raw.apply(fe).values

    dp = lambda df : df[['TransactionID'] + [c for c in df.columns if ('ct_' in c or 'freq_' in c)]]
    train, test = dp(train), dp(test)
    train.to_hdf(f'{path}/{featset_name}_train.h5', mode='w', key='hash_encoding')
    test.to_hdf(f'{path}/{featset_name}_test.h5', mode='w', key='hash_encoding')

@jit(nopython=True)
def amt_window_feats(arr, ws):
    '''
    arr - array corresponding to hash transactions
    ws - window sizes in secs
    '''
    NSTATS = 5  # min, max, mean, std, count
    M = ws.size * NSTATS  # num feats
    N = arr.shape[0]
    r = np.empty((N, M), dtype=np.float32)
    r.fill(np.nan)

    for i in range(N):
        carr = arr[:i + 1, :]  # transaction up to this date
        for k, w in enumerate(ws):
            deltas = np.abs(carr[:, 0] - carr[-1, 0])
            amts = carr[deltas <= w, 1]  # window applied
            r[i, k * NSTATS + 0] = np.min(amts)
            r[i, k * NSTATS + 1] = np.max(amts)
            r[i, k * NSTATS + 2] = np.mean(amts)
            r[i, k * NSTATS + 3] = np.std(amts)
            r[i, k * NSTATS + 4] = amts.size
    return r
def window_features(df_name, path, featset_name):
    '''
    build window feeatures for a given df
    :return:
    '''

    H = 3600
    D, W, M = 24*H, 7*24*H, 30*24*H
    WS = np.array([H, D, W, 2*W, M, 3*M, 6*M], dtype=np.float32)
    w_names = ['hour', 'day', 'week', '2_week', 'month', '3_month', '6_month']
    s_names = ['min', 'max', 'mean', 'std', 'count']
    HLEN = len(w_names) * len(s_names)
    hashes = ['C1_hash']#['card_hash', 'card_mail_hash', 'card_mail_addr_hash']
    f_names = []
    for h in hashes:
        for w in w_names: f_names.extend([f'h{h}_w{w}_{s}' for s in s_names])

    df = load_merge_data(df_name)
    hs = build_hashes_df(df)
    df = df.merge(hs, on='TransactionID', how='left')[hashes+['TransactionID', 'TransactionDT', 'TransactionAmt']]

    final_arr = np.zeros(shape=(df.shape[0], len(f_names)))

    for kk, h in enumerate(hashes):
        cixs, cdict = gen_lookups(df, gb_col=h)
        hdf = df.drop([h_ for h_ in hashes if h_ != h], axis=1).sort_values(
            by=[h, 'TransactionDT']).values[:,2:].astype(np.float32)  # 2: to exclude hash and id

        for k in tqdm.tqdm(cdict.keys()): # For each hash_id
            hix = cdict[k]
            arr = hdf[cixs[hix]:cixs[hix + 1]]
            feats = amt_window_feats(arr, ws=WS)
            final_arr[cixs[hix]:cixs[hix + 1], kk*HLEN:(kk+1)*HLEN] = feats

    final_df = pd.DataFrame(data=final_arr, columns=f_names)
    final_df['TransactionID'] = df['TransactionID']
    final_df.to_hdf(f'{path}/{featset_name}.h5', mode='w', key='hfeats')
def run_window_features():
    for d in ['train', 'test']: window_features(df_name=d, path='./features', featset_name=f'hwindow_feats_v2_{d}')

if __name__ == '__main__':
    # run_window_features()
    run_hash_target_encoding(path='./features', featset_name=f'hash_encoding')
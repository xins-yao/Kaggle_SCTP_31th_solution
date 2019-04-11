import gc
import sys
import time
import warnings
import multiprocessing

import numpy as np
import pandas as pd
import lightgbm as lgb

from os import path, makedirs
from tqdm import tqdm
from utils import Logger
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ======================================================================= Method
def load_dataframe(dataset):
    return pd.read_csv(dataset)


def augment(x, y, t=2):
    xs, xn = [], []
    for i in range(t):
        mask = y > 0
        x1 = x[mask].copy()
        for c in range(200):
            val = x1[:, [c, c+200, c+400]]
            np.random.shuffle(val)
            x1[:, [c, c+200, c+400]] = val
        xs.append(x1)

    for i in range(t//2):
        mask = y == 0
        x1 = x[mask].copy()
        for c in range(200):
            val = x1[:, [c, c+200, c+400]]
            np.random.shuffle(val)
            x1[:, [c, c+200, c+400]] = val
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x, xs, xn])
    y = np.concatenate([y, ys, yn])
    return x, y


# ======================================================================= Main
if __name__ == '__main__':
    gc.enable()
    pd.set_option('max_rows', None)
    pd.set_option('max_columns', None)
    warnings.simplefilter('ignore', UserWarning)

    # =================================================================== Params
    top_folder = './output'

    today = datetime.today()
    now = today.strftime('%m%d-%H%M')
    log_name = now + '.txt'
    sys.stdout = Logger(path.join(top_folder, log_name))

    seed_np = 1011
    np.random.seed(seed_np)
    print('numpy seed: {}'.format(seed_np))

    # =================================================================== Load Data
    start = time.time()
    with multiprocessing.Pool() as pool:
        train, test = pool.map(load_dataframe, ['./input/train.csv', './input/test.csv'])

    # === fake sample
    df_test = test.drop(columns=['ID_code']).values

    unique_samples = []
    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    idx_score = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    idx_synthetic = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    synthetic = test.loc[idx_synthetic]
    test = test.loc[idx_score]

    raw = pd.concat([train, test], axis=0, sort=False, ignore_index=True)

    # ============================== Extra Feature
    len_train = len(train)
    col_var = list(raw.columns[2:])

    # === replace value(frequency=1) to NA
    mask = pd.DataFrame(np.ones([raw.shape[0], len(col_var)]), columns=col_var)
    for col in tqdm(col_var):
        cnt = raw[col].value_counts()
        val = cnt[cnt == 1].index
        mask.loc[np.isin(raw[col], val), col] = 0
    col_repeat = [col + '_repeat_2' for col in col_var]
    raw[col_repeat] = raw[col_var][mask.astype(bool)]

    # === replace value(frequency=1/2) to NA
    mask = pd.DataFrame(np.ones([raw.shape[0], len(col_var)]), columns=col_var)
    for col in tqdm(col_var):
        cnt = raw[col].value_counts()
        val = cnt[np.isin(cnt, [1, 2])].index
        mask.loc[np.isin(raw[col], val), col] = 0
    col_repeat = [col + '_repeat_3' for col in col_var]
    raw[col_repeat] = raw[col_var][mask.astype(bool)]

    raw = pd.concat([raw, synthetic], axis=0, sort=False, ignore_index=True)

    # === logging
    print('data: {}'.format(raw.shape))
    print('elapsed time: {:.1f} min'.format((time.time() - start)/60))

    # =================================================================== PreProcess
    feats = [col for col in raw.columns.values if col not in ['ID_code', 'target']]

    # =================================================================== Model
    train = raw[:len_train]
    test = raw[len_train:].copy()

    x_train = train[feats]
    y_train = train['target']
    x_test = test[feats]

    print('trn_x: {}'.format(x_train.shape))
    print('x_test: {}'.format(x_test.shape))

    param = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': 11,
        'random_state': 1993,
        'learning_rate': 0.01,

        'num_leaves': 8,
        'max_depth': -1,
        'feature_fraction': 0.05,
        'bagging_freq': 5,
        'bagging_fraction': 0.4,
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
    }
    print('model params:\n{}'.format(pd.Series(list(param.values()), index=list(param.keys()))))

    seed_fold = 26
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_fold)
    print('StratifiedKFold seed: {}'.format(seed_fold))

    round_max = 30000
    round_early_stopping = 3000
    print('num_round: {}'.format(round_max))
    print('early_stopping_round: {}'.format(round_early_stopping))

    # === training
    oof = np.zeros(len(x_train))
    predictions = np.zeros(len(x_test))

    start = time.time()
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, y_train.values)):
        print("fold n°{}".format(fold_))

        trn_x, trn_y = x_train.iloc[trn_idx], y_train.iloc[trn_idx]
        val_x, val_y = x_train.iloc[val_idx], y_train.iloc[val_idx]

        N = 5
        for i in range(N):
            X_t, y_t = augment(trn_x.values, trn_y.values)
            X_t = pd.DataFrame(X_t, columns=feats)

            trn_data = lgb.Dataset(X_t, label=y_t)
            val_data = lgb.Dataset(val_x, label=val_y)

            evals_result = {}
            clf = lgb.train(param,
                            trn_data,
                            round_max,
                            valid_sets=[trn_data, val_data],
                            early_stopping_rounds=round_early_stopping,
                            verbose_eval=1000,
                            evals_result=evals_result)

            oof[val_idx] += clf.predict(val_x, num_iteration=clf.best_iteration) / N
            predictions += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / N

        fold_score = roc_auc_score(val_y, oof[val_idx])
        print('fold {} auc score: {:.5f}'.format(fold_, fold_score))

    cv_score = roc_auc_score(y_train, oof)
    print('elapsed time: {:.1f} min'.format((time.time() - start)/60))
    print('auc score: {:.5f}'.format(cv_score))

    # =================================================================== Saving File
    sub_folder = path.join(top_folder, 'cv_' + now + '_' + str(np.round(cv_score, 5)))
    makedirs(sub_folder, exist_ok=True)

    test['target'] = predictions
    test[['ID_code', 'target']].to_csv(path.join(sub_folder, 'submission.csv'), index=False)

    raw['oof'] = np.concatenate([oof, predictions], axis=0)
    raw[['ID_code', 'oof']].to_csv(path.join(sub_folder, 'oof.csv'), index=False)




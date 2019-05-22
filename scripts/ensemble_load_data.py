# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def load_data(eval_path, model_name, split='val'):
    Xs, ys = [], []

    for m_name in model_name.split('_'):
        X, y = _load_data(eval_path, m_name, split)
        Xs.append(X)
        ys.append(y)

    X = {}
    y = {}

    for diagnosis in ['abnormal', 'acl', 'meniscus']:
        X[diagnosis] = np.hstack(x[diagnosis] for x in Xs)

        n_models = len(ys)
        if n_models > 1:
            for i in range(1, n_models):
                assert(
                    np.array_equal(ys[i - 1][diagnosis], ys[i][diagnosis])
                )

        y[diagnosis] = ys[0][diagnosis]

    return X, y


def _load_data(eval_path, model_name, split='val'):
    path = os.path.join(eval_path, model_name, f'{split}_preds.csv')

    log.info(f'Loading data from {path}')
    df = pd.read_csv(path)

    X = {}
    y = {}

    for diagnosis in ['abnormal', 'acl', 'meniscus']:
        X_cols = [
            f'axial_{diagnosis}_pred',
            f'coronal_{diagnosis}_pred',
            f'sagittal_{diagnosis}_pred'
        ]
        y_cols = [
            f'axial_{diagnosis}_label',
            f'coronal_{diagnosis}_label',
            f'sagittal_{diagnosis}_label'
        ]
        X[diagnosis] = df[X_cols].values

        assert(np.all(df[y_cols[0]] == df[y_cols[1]]))
        assert(np.all(df[y_cols[1]] == df[y_cols[2]]))
        y[diagnosis] = df[y_cols[0]].values

    return X, y


# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pickle
import pprint

import pandas as pd
from sklearn.linear_model import LogisticRegression


log = logging.getLogger(__name__)


def load_data(eval_path, model_name, split='val'):
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
        y[diagnosis] = df[y_cols].values

    return X, y


def train(seed, X, y):
    return LogisticRegression(random_state=seed).fit(X, y)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('evals_path')
    parser.add_argument('output_path')
    parser.add_argument('--seed', type=int, default=42)

    verbosity_help = 'Verbosity level (default: %(default)s)'
    choices = [
        logging.getLevelName(logging.DEBUG),
        logging.getLevelName(logging.INFO),
        logging.getLevelName(logging.WARN),
        logging.getLevelName(logging.ERROR)
    ]

    parser.add_argument(
        '-v',
        '--verbosity',
        choices=choices,
        help=verbosity_help,
        default=logging.getLevelName(logging.INFO)
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the logging to console level
    logging.basicConfig(level=args.verbosity)

    return args


if __name__ == '__main__':
    args = parse_args()

    log.info(pprint.pformat(args.__dict__))

    output_path = args.output_path
    model_name = args.model_name

    os.makedirs(output_path, exist_ok=True)

    val_X, val_y = load_data(args.evals_path, model_name, split='val')

    ensembles = {'name': model_name}
    for diagnosis in ['abnormal', 'acl', 'meniscus']:
        log.info(f'Training {diagnosis} model')
        ensembles[diagnosis] = train(
            args.seed, val_X[diagnosis], val_y[diagnosis]
        )

    # save the models
    model_path = os.path.join(output_path, f'{model_name}_ensembles.pkl')
    log.info(f'Saving models to {model_path}')
    with open(model_path, 'wb') as fout:
        pickle.dump(ensembles, fout)

# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pickle
import pprint

import numpy as np
import pandas as pd
import sklearn.metrics as metrics


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

        assert(np.all(df[y_cols[0]] == df[y_cols[1]]))
        assert(np.all(df[y_cols[1]] == df[y_cols[2]]))
        y[diagnosis] = df[y_cols[0]].values

    return X, y


def print_eval(preds, truth, threshold=0.5):
    int_preds = (preds > threshold).astype(int)
    int_truth = np.array(truth).astype(int)

    conf = metrics.confusion_matrix(int_truth, int_preds)

    TN = float(conf[0][0])
    FN = float(conf[1][0])
    TP = float(conf[1][1])
    FP = float(conf[0][1])

    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    fpr, tpr, threshold = metrics.roc_curve(truth, preds)
    auc = metrics.auc(fpr, tpr)

    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Accuracy: {accuracy}')
    print(f'AUC: {auc}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('output_path')

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

    os.makedirs(output_path, exist_ok=True)

    test_X, test_y = load_data(args.evals_path, args.model_name, split='test')

    model_path = args.model_path
    log.info(f'Loading moel from {model_path}')
    with open(model_path, 'rb') as fin:
        ensembles = pickle.load(fin)

    model_name = ensembles['name']
    log.info(f'Found {model_name} ensembles')

    preds = {}

    for diagnosis in ['abnormal', 'acl', 'meniscus']:
        print(diagnosis)
        preds[diagnosis] = (
            ensembles[diagnosis].predict_proba(test_X[diagnosis])[:, 1].ravel()
        )

        print_eval(preds[diagnosis], test_y[diagnosis])

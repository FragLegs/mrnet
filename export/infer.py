# -*- coding: utf-8 -*-
import argparse
import collections
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from model_choice import MODELS

log = logging.getLogger(__name__)

MODEL_NAMES = ['MRNet', 'MRNet-Squeeze', 'MRNet-Attend', 'MRNet-SqueezeAttend']
SERIES_NAMES = ['axial', 'coronal', 'sagittal']
DIAGNOSIS_NAMES = ['abnormal', 'acl', 'meniscus']


def load_models(models_path):
    models = {}

    for model_name in MODEL_NAMES:
        models[model_name] = {}
        for series in SERIES_NAMES:
            models[model_name][series] = {}
            for diagnosis in DIAGNOSIS_NAMES:
                model_path = os.path.join(
                    models_path, f'{diagnosis}-{model_name}-{series}'
                )
                print(f'Loading {model_name} from {model_path}')
                model = MODELS[model_name](pretrained=False)
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                models[model_name][series][diagnosis] = model

    ensemble_path = os.path.join(models_path, 'final_ensembles.pkl')
    print(f'Loading ensembles from {ensemble_path}')
    with open(ensemble_path, 'rb') as fin:
        models['ensemble'] = pickle.load(fin)

    return models


def load_data(data_path):
    data = collections.defaultdict(dict)

    for line in open(data_path):
        line = line.strip()
        elems = line.split('/')
        case, _ = elems[-1].split('.')
        series = elems[-2]
        data[case][series] = line

    return data


def load_volume(path):

    INPUT_DIM = 224
    MAX_PIXEL_VAL = 255
    MEAN = 58.09
    STDDEV = 49.73

    vol = np.load(path)

    # crop middle
    pad = int((vol.shape[2] - INPUT_DIM) / 2)
    vol = vol[:, pad:-pad, pad:-pad]

    # standardize
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

    # normalize
    vol = (vol - MEAN) / STDDEV

    # convert to RGB
    vol = np.stack((vol,) * 3, axis=1)

    return torch.FloatTensor(vol)


def predict_diagnosis(models, case, diagnosis):
    preds = []
    for model_name in MODEL_NAMES:
        for series in SERIES_NAMES:
            vol = Variable(load_volume(case[series]))
            logit = models[model_name][series][diagnosis].forward(vol)
            preds.append(torch.sigmoid(logit))

    preds = np.array(preds).reshape(1, -1)
    prob = models['ensemble'][diagnosis].predict_proba(preds)[0, 1]

    return prob


def predict_case(models, case):
    return [predict_diagnosis(models, case, d) for d in DIAGNOSIS_NAMES]


def main(data_path, pred_path, models_path):
    print(f'Loading data from {data_path}')
    data = load_data(data_path)

    print(f'Loading models from {models_path}')
    models = load_models(models_path)

    preds = []

    print('Found {} cases'.format(len(data)))

    for case in sorted(list(data.keys())):
        print(f'Predicting case {case}')
        preds.append(predict_case(models, data[case]))

    df = pd.DataFrame(data=np.array(preds))

    print('Writing {} cases to {}'.format(len(df), pred_path))
    df.to_csv(pred_path, index=False, header=False)


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Predict on MRNet data'
    parser = argparse.ArgumentParser(description=desc)

    data_path_help = 'Where the volume filenames are'
    parser.add_argument(
        'data_path',
        type=str,
        help=data_path_help
    )

    pred_path_help = 'Where to write the predictions'
    parser.add_argument(
        'pred_path',
        type=str,
        help=pred_path_help
    )

    models_path_help = 'Folder where models live'
    parser.add_argument(
        '--models_path',
        type=str,
        default='final_models',
        help=models_path_help
    )

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
    print(args)
    main(args.data_path, args.pred_path, args.models_path)

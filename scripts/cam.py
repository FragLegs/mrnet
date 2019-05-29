# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch

from loader import load_data
from model_choice import MODELS

log = logging.getLogger(__name__)


def get_weights(model):
    return model.classifier.weight.numpy().reshape(-1)  # n_channel


def get_features(model, volume):
    x = torch.squeeze(volume, dim=0)  # only batch size 1 supported
    features = model.model.features(x)
    x = model.gap(features).view(features.size(0), -1)

    name = model.__class__.__name__

    if 'Attention' in name:
        m = torch.softmax(model.attention(x), dim=0).data.cpu().numpy()
        idx = np.argmax(m)
        print(idx)
    else:
        a = torch.argmax(x, 0).view(-1).data.cpu().numpy()
        idx = Counter(a)[0][0]
        print(idx)

    return features.data.cpu().numpy(), idx


def get_CAM(model, volume):
    features, idx = get_features(model, volume)  # n_seq, n_channel, w, h
    weights = get_weights(model)

    n_seq, n_channel, width, height = features.shape
    features = features.reshape(n_seq, n_channel, width * height)

    cams = weights @ features  # n_seq, w * h

    cam = cams[idx]
    cam = cam.reshape(width, height)
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = np.uint8(255 * cam)
    return cv2.resize(cam, (256, 256)), idx


def get_data(diagnosis, series, gpu):
    # load the paths dataframe
    paths = pd.read_csv(
        '/mnt/mrnet-image-paths.csv', header=None, names=['path']
    ).path.values

    # load the labels dataframe
    label_df = pd.read_csv('/mnt/mrnet-labels-3way.csv', index_col=0)

    _, _, test_loader = load_data(
        paths=paths, series=series, label_df=label_df,
        diagnosis=diagnosis, use_gpu=gpu, is_full=False,
        augment=False
    )

    return test_loader


def get_model(model_name, model_path, gpu):
    model = MODELS[model_name]()
    state_dict = torch.load(
        model_path, map_location=(None if gpu else 'cpu')
    )
    model.load_state_dict(state_dict)

    if gpu:
        model = model.cuda()

    return model


def main(model_name,
         model_path,
         diagnosis,
         series,
         gpu,
         output_path,
         **kwargs):

    os.makedirs(output_path, exist_ok=True)

    test_loader = get_data(diagnosis, series, gpu)
    model = get_model(model_name, model_path, gpu)

    for batch in test_loader:
        vol, label, case = batch
        cam, idx = get_CAM(model, vol)
        img = vol[idx]
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        colored = 0.3 * heatmap + 0.5 * img

        img_path = 'result-{}-{}.jpg'.format(case[:-len('.npy')], label)
        img_path = os.path.join(output_path, img_path)
        cv2.imwrite(img_path, colored)
        break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('model_path')
    parser.add_argument(
        '-d',
        '--diagnosis',
        choices=['abnormal', 'acl', 'meniscus', 'all'],
        default='all'
    )
    parser.add_argument(
        '-s',
        '--series',
        choices=['axial', 'coronal', 'sagittal', 'all'],
        default='all'
    )
    parser.add_argument('--gpu', action='store_true')
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
    main(**parse_args().__dict__)

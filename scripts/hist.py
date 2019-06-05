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
    return model.classifier.weight.data.numpy().reshape(-1)  # n_channel


def get_features(model, volume):
    x = torch.squeeze(volume, dim=0)  # only batch size 1 supported
    features = model.model.features(x)
    x = model.gap(features).view(features.size(0), -1)

    name = model.__class__.__name__

    if 'Attention' in name:
        m = torch.softmax(model.attention(x), dim=0).data.cpu().numpy()
        print(m.shape)
        idx = np.argmax(m)
    else:
        a = torch.argmax(x, 0).view(-1).data.cpu().numpy()
        print(a.shape)
        idx = Counter(a).most_common(1)[0][0]

    return features.data.cpu().numpy(), idx


def get_CAM(model, volume, idx=None):
    final_dim = (volume.shape[-2], volume.shape[-1])
    features, max_idx = get_features(model, volume)  # n_channel, w, h
    weights = get_weights(model)

    if idx is None:
        idx = max_idx
    features = features[idx]

    n_channel, width, height = features.shape
    features = features.reshape(n_channel, width * height)

    cam = weights @ features  # w * h
    cam = cam.reshape(width, height)
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = np.uint8(255 * cam)
    return cv2.resize(cam, final_dim), idx


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


def get_model_path(model_name, diagnosis, series):
    models_dir = f'/mnt/runs/{model_name}/{series}/{diagnosis}'
    most_recent = sorted(os.listdir(models_dir))[-1]
    most_recent_path = os.path.join(models_dir, most_recent)
    model_paths = sorted([
        fn for fn in os.listdir(most_recent_path)
        if fn.startswith('val')
    ])
    model_path = os.path.join(most_recent_path, model_paths[0])

    return model_path


def denorm(img, mean=58.09, std=49.73):
    img = img * std
    img = img + mean
    return img


def create_and_save_CAM(model,
                        batch,
                        output_path,
                        diagnosis,
                        series,
                        idx=None):
    vol, label, case = batch
    case = case[0][:-len('.npy')]
    label = int(label.view(-1).data.numpy()[0])
    cam, idx = get_CAM(model, vol, idx)

    _, n_seq, n_channel, width, height = vol.shape

    img = vol.view(n_seq, n_channel, width, height).data.numpy()[idx]
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img = denorm(np.moveaxis(img, 0, 2))
    colored = 0.3 * heatmap + 0.5 * img

    pred = torch.sigmoid(model.forward(vol)).data.cpu().numpy()[0][0]

    img_path = 'result-{}-{}-{}-t{}-p{:.3f}-i{}.jpg'.format(
        case, diagnosis, series, label, pred, idx
    )
    img_path = os.path.join(output_path, img_path)
    cv2.imwrite(img_path, colored)
    return img_path


def main(#model_name,
         diagnosis,
         series,
         gpu,
         output_path,
         idx,
         **kwargs):

    output_path = os.path.join(output_path, model_name)
    os.makedirs(output_path, exist_ok=True)

    test_loader = get_data(diagnosis, series, gpu)
    model_path = get_model_path(model_name, diagnosis, series)
    model = get_model(model_name, model_path, gpu)

    for batch in test_loader:
        create_and_save_CAM(model, batch, output_path, diagnosis, series)
        break


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('model_name')
    parser.add_argument('output_path')
    parser.add_argument(
        '-i',
        '--idx',
        type=int,
        default=88
    )
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

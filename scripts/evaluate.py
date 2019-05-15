import argparse
from datetime import datetime
import logging
# import matplotlib.pyplot as plt
import os
import pprint

import numpy as np
import pandas as pd
import torch
# import wandb

from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm

from loader import load_data
from model import MRNet


log = logging.getLogger(__name__)


def run_model(model, loader, train=False, optimizer=None, log_every=25):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0
    cases = []

    for batch in tqdm(loader):
        if train:
            optimizer.zero_grad()

        vol, label, case = batch
        cases.append(case)
        if loader.dataset.use_gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)
        label = Variable(label)

        logit = model.forward(vol)

        loss = loader.dataset.weighted_loss(logit, label)
        loss_val = loss.item()
        total_loss += loss_val

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels, cases


def load_model(model_path, use_gpu):
    model = MRNet()
    state_dict = torch.load(
        model_path, map_location=(None if use_gpu else 'cpu')
    )
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    return model


def evaluate(loader, model):

    loss, auc, preds, labels, cases = run_model(model, loader)

    log.info(f'loss: {loss:0.4f}')
    log.info(f'AUC: {auc:0.4f}')

    return preds, labels, cases


def parse_args():
    parser = argparse.ArgumentParser()
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    args.now = datetime.now().strftime('%m-%d_%H-%M')

    args.rundir = args.rundir.format(**args.__dict__)

    os.makedirs(args.rundir, exist_ok=True)

    log.info(pprint.pformat(args.__dict__))

    # load the paths dataframe
    paths = pd.read_csv(
        '/mnt/mrnet-image-paths.csv', header=None, names=['path']
    ).path.values

    # load the labels dataframe
    label_df = pd.read_csv('/mnt/mrnet-labels-3way.csv', index_col=0)

    _, valid_loader, test_loader = load_data(
        paths=paths, series=args.series, label_df=label_df,
        diagnosis=args.diagnosis, use_gpu=args.gpu, is_full=False
    )

    log.info('Loading model from {}'.format(args.model_path))
    model = load_model(args.model_path, args.gpu)

    log.info('Validation')

    val_preds, val_labels, val_cases = evaluate(valid_loader, model)

    log.info('Test')

    test_preds, test_labels, test_cases = evaluate(test_loader, model)

    model_dir = os.path.dirname(args.model_path)

    val = pd.DataFrame()
    val['case'] = val_cases
    val['label'] = val_labels
    val['pred'] = val_preds
    val.to_csv(os.path.join(model_dir, 'val_preds.csv'), index=False)

    test = pd.DataFrame()
    test['case'] = test_cases
    test['label'] = test_labels
    test['pred'] = test_preds
    test.to_csv(os.path.join(model_dir, 'test_preds.csv'), index=False)

import argparse
import collections
import logging
# import matplotlib.pyplot as plt
import os
import pprint

import pandas as pd
import torch
# import wandb

from sklearn import metrics
from torch.autograd import Variable

from loader import load_data
from model import MRNet


log = logging.getLogger(__name__)


def run_model(model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0
    cases = []

    for batch in loader:
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
    parser.add_argument('model_name')
    parser.add_argument('output_path')
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

    log.info(pprint.pformat(args.__dict__))

    output_path = os.path.join(args.output_path, args.model_name)

    os.makedirs(output_path, exist_ok=True)

    # load the paths dataframe
    paths = pd.read_csv(
        '/mnt/mrnet-image-paths.csv', header=None, names=['path']
    ).path.values

    # load the labels dataframe
    label_df = pd.read_csv('/mnt/mrnet-labels-3way.csv', index_col=0)

    model_name = args.model_name

    val_cases = collections.defaultdict(dict)
    test_cases = collections.defaultdict(dict)

    for series in ['axial', 'coronal', 'sagittal']:
        for diagnosis in ['abnormal', 'acl', 'meniscus']:
            log.info(f'{series}/{diagnosis}')
            models_dir = f'runs/{model_name}/{series}/{diagnosis}'
            most_recent = sorted(os.listdir(models_dir))[-1]
            most_recent_path = os.path.join(models_dir, most_recent)
            model_paths = sorted([
                fn for fn in os.listdir(most_recent_path)
                if fn.startswith('val')
            ])
            model_path = os.path.join(most_recent_path, model_paths[0])

            _, valid_loader, test_loader = load_data(
                paths=paths, series=series, label_df=label_df,
                diagnosis=diagnosis, use_gpu=args.gpu, is_full=False
            )

            log.info('Loading model from {}'.format(model_path))
            model = load_model(model_path, args.gpu)

            log.info('Validation')

            for pred, label, case in zip(*evaluate(valid_loader, model)):
                val_cases[case]['case'] = case
                val_cases[case][f'{series}_{diagnosis}_label'] = label
                val_cases[case][f'{series}_{diagnosis}_pred'] = pred

            log.info('Test')

            if test_loader is not None:
                for pred, label, case in zip(*evaluate(test_loader, model)):
                    test_cases[case]['case'] = case
                    test_cases[case][f'{series}_{diagnosis}_label'] = label
                    test_cases[case][f'{series}_{diagnosis}_pred'] = pred

    val = pd.DataFrame.from_records(list(val_cases.values()))
    test = pd.DataFrame.from_records(list(test_cases.values()))

    val_path = os.path.join(output_path, 'val_preds.csv')
    log.info(f'Writing validation predictions to {val_path}')
    val.to_csv(val_path, index=False)

    test_path = os.path.join(output_path, 'test_preds.csv')
    log.info(f'Writing test predictions to {test_path}')
    test.to_csv(test_path, index=False)

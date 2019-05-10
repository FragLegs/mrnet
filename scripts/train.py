import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import pprint

import numpy as np
import pandas as pd
import torch
import wandb

# from sklearn import metrics

from evaluate import run_model
from loader import load_data
from model import MRNet


log = logging.getLogger(__name__)


MODELS = {
    'MRNet': MRNet
}


def train(model_name,
          rundir,
          diagnosis,
          series,
          epochs,
          learning_rate,
          weight_decay,
          max_patience,
          factor,
          full,
          gpu,
          log_interval,
          **kwargs):

    # load the paths dataframe
    paths = pd.read_csv(
        '/mnt/mrnet-image-paths.csv', header=None, names=['path']
    ).path.values

    # load the labels dataframe
    label_df = pd.read_csv('/mnt/mrnet-labels-3way.csv', index_col=0)

    train_loader, valid_loader, _ = load_data(
        paths=paths, series=series, label_df=label_df,
        diagnosis=diagnosis, use_gpu=gpu, is_full=full
    )

    model = MODELS[model_name]()  # MRNet()
    wandb.watch(model)

    if gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max_patience, factor=factor, threshold=1e-4
    )

    best_val_loss = float('inf')

    start_time = datetime.now()

    for epoch in range(epochs):
        change = datetime.now() - start_time
        print(
            'starting epoch {}. time passed: {}'.format(epoch + 1, str(change))
        )

        train_loss, train_auc, _, _ = run_model(
            model, train_loader, train=True,
            optimizer=optimizer, log_every=log_interval
        )
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')

        wandb.log({
            'train_loss': train_loss,
            'train_auc': train_auc,
            'valid_loss': val_loss,
            'valid_auc': val_auc,
            'epoch': epoch
        })

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            file_name = (
                f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch + 1}'
            )
            save_path = Path(rundir) / file_name
            torch.save(model.state_dict(), save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument(
        '--rundir',
        type=str,
        default='runs/{model_name}/{series}/{diagnosis}/{now}'
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
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--log-interval', default=25, type=int)

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

    log.debug(pprint.pformat(args.__dict__))
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    name = '{}--{}'.format(args.rundir, )

    wandb.init(
        name=args.rundir,
        config=args,
        project='mrnet',
        dir=args.rundir
    )

    train(**args.__dict__)

# -*- coding: utf-8 -*-
import argparse
import logging
import os

import pandas as pd
import seaborn as sns


log = logging.getLogger(__name__)


def load_data(model):
    dfs = []
    for series in ['axial', 'coronal', 'sagittal']:
        for diagnosis in ['abnormal', 'acl', 'meniscus']:
            dfs.append(load_and_shape(model, series, diagnosis))

    return pd.concat(dfs, ignore_index=True)


def load_and_shape(model, series, diagnosis):
    base_dir = '/mnt/runs/{}/{}/{}'.format(model, series, diagnosis)

    subruns = sorted(os.listdir(base_dir))
    most_recent = subruns[-1]

    metrics_path = os.path.join(base_dir, most_recent, 'metrics.csv')
    log.debug('Reading {}'.format(metrics_path))

    metrics = pd.read_csv(metrics_path)

    recs = []
    for value in ['loss', 'auc']:
        for _, row in metrics.iterrows():
            recs.append({
                'epoch': row.epoch,
                value: row['train_{}'.format(value)],
                'split': 'train',
                'model': model,
                'series': series,
                'diagnosis': diagnosis
            })
            recs.append({
                'epoch': row.epoch,
                value: row['val_{}'.format(value)],
                'split': 'val',
                'model': model,
                'series': series,
                'diagnosis': diagnosis
            })
    return pd.DataFrame.from_records(recs)


def plot(metrics, value):
    recs = []
    for _, row in metrics.iterrows():
        recs.append({
            'epoch': row.epoch,
            value: row['train_{}'.format(value)],
            'split': 'train'
        })
        recs.append({
            'epoch': row.epoch,
            value: row['val_{}'.format(value)],
            'split': 'val'
        })
    df = pd.DataFrame.from_records(recs)

    return sns.line_plot(
        x='epoch',
        y=value,
        hue='split',
        data=df,
        hue_order=['train', 'val']
    )


def save_plot(path, metrics, y_axis):
    facets = sns.FacetGrid(metrics, row='diagnosis', col='series', hue='split')
    loss_graph = facets.map(sns.lineplot, 'epoch', y_axis)
    loss_graph.add_legend()
    loss_graph.savefig(path)


def plot_data(model, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    metrics = load_data(model)

    loss_path = os.path.join(output_dir, '{}_loss.png'.format(model))
    save_plot(loss_path, metrics, 'loss')

    auc_path = os.path.join(output_dir, '{}_auc.png'.format(model))
    save_plot(auc_path, metrics, 'auc')


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Plot loss and AUC for each series/diagnosis for a single model'
    parser = argparse.ArgumentParser(description=desc)

    model_help = 'The name of the model'
    parser.add_argument(
        'model',
        type=str,
        help=model_help
    )

    output_dir_help = 'Where to save the figure'
    parser.add_argument(
        'output_dir',
        type=str,
        help=output_dir_help
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
    plot_data(**parse_args().__dict__)

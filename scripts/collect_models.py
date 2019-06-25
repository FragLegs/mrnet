# -*- coding: utf-8 -*-
import argparse
import logging
import os
import shutil
from zipfile import ZipFile

log = logging.getLogger(__name__)


def collect_models(output_path):
    output_dir = os.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    names = ['MRNet', 'MRNet-Attend', 'MRNet-Squeeze', 'MRNet-SqueezeAttend']
    with ZipFile(output_path, 'w') as fout:
        for model_name in names:
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

                    out_model_name = f'{diagnosis}-{model_name}-{series}'
                    out_model_path = os.path.join(output_dir, out_model_name)

                    log.info(
                        f'Copying model from {model_path} to {out_model_path}'
                    )
                    shutil.copyfile(model_path, out_model_path)

                    log.info(f'Adding {out_model_path} to {output_path}')
                    fout.write(out_model_path)


def parse_args():
    """
    Parses the arguments from the command line

    Returns
    -------
    argparse.Namespace
    """
    desc = 'Collect best models'
    parser = argparse.ArgumentParser(description=desc)

    output_path_help = 'Path to the zip file being created'
    parser.add_argument(
        '--output_path',
        type=str,
        default='final_models/final-models.zip',
        help=output_path_help
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
    collect_models(args.output_path)

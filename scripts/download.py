# -*- coding: utf-8 -*-
import logging

import boto3

log = logging.getLogger(__name__)


if __name__ == '__main__':
    s3 = boto3.client('s3')
    print('Downloading...')
    s3.download_file(
        Bucket='mrnet-data',
        Key='MRNet-v1.0.tar.gz',
        Filename='/domino/datasets/local/mrnet-data/MRNet-v1.0.tar.gz'
    )
    print('Done!')

#!/bin/bash
cd /domino/datasets/output/mrnet-data
python /repos/mrnet/scripts/download.py
echo "Extracting..."
tar xvzf MRNet-v1.0.tar.gz
echo "Done!"

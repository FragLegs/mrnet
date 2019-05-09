#!/bin/bash
cd /domino/datasets/output/mrnet-data
python download.py
echo "Extracting..."
tar xvzf MRNet-v1.0.tar.gz
echo "Done!"

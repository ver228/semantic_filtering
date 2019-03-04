#!/bin/bash

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/bgnd_removal/scripts


python -W ignore train_model.py --batch_size 32  --data_type 'sBBBC026-fibroblasts' --loss_type 'l1smooth' \
--lr 32e-5 --num_workers 4  --n_epochs 500 --save_frequency 25
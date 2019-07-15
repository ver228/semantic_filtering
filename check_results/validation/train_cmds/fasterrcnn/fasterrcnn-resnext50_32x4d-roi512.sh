#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

source activate pytorch-1.0
cd $HOME/GitLab/bgnd_removal/check_results/validation

python -W ignore train_fasterrcnn.py \
--backbone 'resnext50_32x4d' \
--batch_size 12 \
--lr 1e-4 \
--roi_size 512 \
--num_epochs 10000
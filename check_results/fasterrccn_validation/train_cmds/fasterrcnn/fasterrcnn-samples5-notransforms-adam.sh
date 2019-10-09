#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

source activate pytorch-1.0
cd $HOME/GitLab/semantic_filtering/check_results/validation

python -W ignore train_fasterrcnn.py \
--backbone 'resnet50' \
--roi_size 512 \
--lr 0.005 \
--max_samples 5 \
--use_transforms False \
--num_epochs 1000
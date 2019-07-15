#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

source activate pytorch-1.0
cd $HOME/GitLab/bgnd_removal/check_results/validation

python -W ignore train_fasterrcnn.py \
--backbone 'resnet50' \
--roi_size 512 \
--optimizer_name 'sgd' \
--lr_scheduler_name 'stepLR-250-0.1' \
--lr 0.0005 \
--momentum 0.9 \
--weight_decay 5e-4 \
--max_samples 100 \
--only_flip_transforms True \
--batch_size 6 \
--num_epochs 1000
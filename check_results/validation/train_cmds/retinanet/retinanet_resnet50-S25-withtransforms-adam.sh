#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

source activate pytorch-1.0
cd $HOME/GitLab/bgnd_removal/check_results/validation


python -W ignore train_fasterrcnn.py \
--model_name 'retinanet' \
--backbone 'resnet50' \
--roi_size 512 \
--optimizer_name 'adam' \
--lr 0.0001 \
--num_epochs 750 \
--max_samples 25 \
--batch_size 4 \
--only_flip_transforms
#--use_transforms False \
#--batch_size 12 \
#--weight_decay 1e-4 \
#--lr_scheduler_name 'stepLR-250-0.1' \

#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

source activate pytorch-1.0
cd $HOME/GitLab/bgnd_removal/check_results/validation

python -W ignore train_fasterrcnn.py \
--model_name 'retinanet' \
--backbone 'resnet50' \
--optimizer_name 'adam' \
--lr_scheduler_name 'stepLR-40-0.1' \
--lr 1e-4 \
--batch_size 4 \
--use_transforms False \
--num_epochs 130
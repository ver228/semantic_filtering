#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

source activate pytorch-1.0
cd $HOME/GitLab/bgnd_removal/check_results/validation

python -W ignore train_fasterrcnn.py \
--model_name 'retinanet' \
--backbone 'resnet50' \
--optimizer_name 'adam' \
--lr 1e-5 \
--batch_size 12 \
--roi_size 512 \
--num_epochs 130

#--lr_scheduler_name 'stepLR-40-0.1' \
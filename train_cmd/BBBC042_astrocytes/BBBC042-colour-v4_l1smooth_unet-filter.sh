#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/semantic_filtering/scripts


python -W ignore train_model.py \
--batch_size 32  \
--data_type 'BBBC042-colour-v4' \
--loss_type 'l1smooth' \
--model_name 'unet-filter' \
--lr 32e-5 \
--num_workers 1  \
--n_epochs 101 \
--save_frequency 5 \
--is_preloaded True
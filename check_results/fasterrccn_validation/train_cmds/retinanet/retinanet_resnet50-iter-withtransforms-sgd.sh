#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

source activate pytorch-1.0
cd $HOME/GitLab/semantic_filtering/check_results/validation

n_samples=( 5 10 25 100)

for i in "${n_samples[@]}"
do
	echo $i
python -W ignore train_fasterrcnn.py \
--model_name 'retinanet' \
--backbone 'resnet50' \
--roi_size 512 \
--optimizer_name 'sgd' \
--lr_scheduler_name 'stepLR-250-0.1' \
--lr 0.0005 \
--momentum 0.9 \
--weight_decay 5e-4 \
--batch_size 12 \
--num_epochs 750 \
--max_samples $i

done
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
--backbone 'resnet50' \
--roi_size 512 \
--optimizer_name 'adam' \
--lr_scheduler_name 'stepLR-40-0.1' \
--lr 1e-4 \
--max_samples $i \
--use_transforms False \
--batch_size 6 \
--num_epochs 750
done
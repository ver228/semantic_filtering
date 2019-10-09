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
--optimizer_name 'adam' \
--lr_scheduler_name 'stepLR-120-0.1' \
--lr 1e-4 \
--batch_size 4 \
--max_samples $i \
--only_hflip True \
--num_epochs 450
done
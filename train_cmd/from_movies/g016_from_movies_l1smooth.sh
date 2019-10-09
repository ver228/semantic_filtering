#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd /users/rittscher/avelino/GitLab/semantic_filtering/scripts


python train_model.py --batch_size 8 --data_type 'from-movies-gap-16' --model_name 'unet-ch3' --loss_type 'l1smooth' --lr 8e-5  --num_workers 8
#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd /users/rittscher/avelino/GitLab/bgnd_removal/scripts


python -W ignore train_fmnist_v2.py --batch_size 32 --data_type 'fmnist-v2' --loss_type 'l1smooth' --lr 32e-5  --num_workers 8
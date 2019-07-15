#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/bgnd_removal/scripts


python -W ignore train_microglia.py --batch_size 12  --data_type 'microglia-v2-separated' --loss_type 'l1smooth' \
--lr 12e-5 --num_workers 4 --n_epochs 1000 --save_frequency 50
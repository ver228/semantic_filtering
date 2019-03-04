#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/bgnd_removal/scripts


python -W ignore train_BBBC042.py --batch_size 32  --data_type 'BBBC042' --loss_type 'l1smooth' \
--lr 32e-5 --num_workers 8  --n_epochs 500 --save_frequency 25 \
--data_root_dir '/tmp/avelino' --init_model_path 'BBBC042-v3_unet_l1smooth_20190227_153240_adam_lr0.00032_wd0.0_batch32/checkpoint.pth.tar'
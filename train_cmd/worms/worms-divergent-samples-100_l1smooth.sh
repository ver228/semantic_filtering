#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd /users/rittscher/avelino/GitLab/bgnd_removal/scripts

for i in {1..5};
do
echo 'Loop='$i
python train_model.py --batch_size 36  --data_type 'worms-divergent-samples-100' --loss_type 'l1smooth' \
--n_epochs 600 --samples_per_epoch 2790 --lr 1e-4 --num_workers 8;
done

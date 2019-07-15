#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from bgnd_removal.trainer import train, get_device, log_dir_root_dflt
from bgnd_removal.flow import BasicFlow
from bgnd_removal.models import UNet, get_loss

def train_N2N(
            data_root_dir,
            log_dir_root = log_dir_root_dflt,
            loss_type = 'l1smooth',
            cuda_id = 0,
            n_epochs = 2000,
            samples_per_epoch = 2790,
            batch_size = 16,
            num_workers = 1,
            lr = 16e-5, 
            weight_decay = 0.0,
            cropping_size = 256,
            **argkws
            ):
    
    save_prefix = f'N2N_unet_{loss_type}'
    log_dir = log_dir_root / 'microglia'

    gen = BasicFlow(root_dir = data_root_dir, 
                        is_log_transform = False, 
                        scale_int = (0, 255),
                        samples_per_epoch = 2790,
                        cropping_size = cropping_size
                        )

    n_ch_in, n_ch_out = 1, 1
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    device = get_device(cuda_id)

    criterion = get_loss(loss_type)

    
    train(save_prefix,
        model,
        device,
        gen,
        criterion,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        **argkws
        )

if __name__ == '__main__':
    import fire
    
    fire.Fire(train_N2N)
    
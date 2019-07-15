#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from bgnd_removal.trainer import train, get_device, log_dir_root_dflt
from bgnd_removal.flow import FluoMergedFlow
from bgnd_removal.models import UNet, get_loss

def train_fmnist_v2(
                data_type = 'microglia-fluo',
                model_name = 'unet',
                loss_type = 'l1smooth',
                cuda_id = 0,
                log_dir_root = log_dir_root_dflt,
                batch_size = 16,
                num_workers = 1,
                **argkws
                ):
    
    save_prefix = f'{data_type}_{model_name}_{loss_type}'
    log_dir = log_dir_root / 'fmnist_v2'
    
    is_separated_output = '-separated' in data_type
    
    n_ch_in  = 1
    if is_separated_output:
        n_ch_out = 3
    else:
        n_ch_out = 1
    
    
    if 'fmnist-v2' in data_type:
        cells1_prefix = 'foreground'
        cells2_prefix = 'background_crops'
        bgnd_prefix = 'background'
    else:
        raise ValueError(data_type)
        
    data_root_dir = Path.home() / 'workspace/denoising/data/MNIST_fashion/train'
    gen = FluoMergedFlow(epoch_size = 20480,
                        root_dir = data_root_dir,
                        bgnd_prefix = bgnd_prefix,
                            cells1_prefix = cells1_prefix,
                            cells2_prefix = cells2_prefix,
                            crop_size = (128, 128),
                             is_log_transform = False,
                             int_scale = (0, 255),
                             img_ext = '*.png',
                             is_timeseries_dir = False,
                             n_cells_per_crop = 5,
                             n_bgnd_per_crop = 10,
                             int_factor = (0.9, 1.1),
                             bgnd_sigma_range = (0., 1.2),
                             frac_crop_valid = 0.25,
                             zoom_range = (0.9, 1.1),
                             noise_range = (0., 5.),
                             rotate_range = (0, 90),
                             max_overlap = 0.5,
                             is_separated_output = is_separated_output
                         )  


    valid_dat_types = ['fmnist-v2', 'fmnist-v2-separated'] 
    if not any([data_type == x for x in valid_dat_types]):
        raise ValueError(data_type)
    
    
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
    
    fire.Fire(train_fmnist_v2)
    